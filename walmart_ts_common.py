"""
walmart_ts_common.py
=====================
Shared infrastructure for the three time-series models (PatchTST, DLinear,
Prophet) in the Walmart Store Sales five-model comparison.

Why this file exists
--------------------
The whole point of a five-model comparison is that any difference in WMAE must
come from the *model*, not from differences in how the data was split, cleaned,
or scored. This module is the single source of truth for all three of those
things, so PatchTST, DLinear and Prophet are provably evaluated on identical
data.

It deliberately reuses the *lightweight* parts of the existing
``preprocessor.py`` (the one the XGBoost / LightGBM notebooks use):

    merge_and_preprocess  -> identical table merge
    make_masks            -> identical train / val / test split (last 8 weeks)
    clean_target          -> identical IQR target cleaning

It does NOT run the 250-feature engineering pipeline, because the channel-
independent deep models want a clean univariate sales series, not a wide
feature table. The pieces above are all that define the shared *contract*.

Public surface used by the three notebooks
------------------------------------------
    build_panel()               -> tidy long panel (the shared contract)
    SeriesStore(panel)          -> per-series windowing / naive fallbacks
    score_val_predictions(...)  -> join preds onto real val rows + WMAE
    wmae_breakdown(merged)      -> {val_wmae, holiday / non-holiday split, ...}
    ExperimentTracker(name)     -> log(), summary(), diff(), to_csv()
    build_submission(...)       -> competition submission frame
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. Constants that mirror the preprocessor's evaluation contract             #
# --------------------------------------------------------------------------- #
VAL_WEEKS: int = 8          # must match preprocessor.VAL_WEEKS_DEFAULT
HOLIDAY_WEIGHT: float = 5.0  # must match preprocessor w_val mapping {1:5, 0:1}


# --------------------------------------------------------------------------- #
# 1. Build the canonical panel (reuses preprocessor.py verbatim)              #
# --------------------------------------------------------------------------- #
def build_panel(
    base_path: str = "/kaggle/input/competitions/"
                     "walmart-recruiting-store-sales-forecasting/",
    val_weeks: int = VAL_WEEKS,
) -> pd.DataFrame:
    """Return one tidy long panel that every TS model consumes.

    Columns: Store, Dept, Date, IsHoliday, y, split

    * ``y``     -- ``Weekly_Sales`` AFTER ``clean_target``.  This means the
                   train region is IQR-clipped (exactly what XGB/LGBM trained
                   on) while the val region keeps its TRUE, unclipped values
                   (exactly what XGB/LGBM was scored against).  So a single
                   column already encodes "clipped where train, true where val".
    * ``split`` -- 'train' | 'val' | 'test'  (identical boundaries to XGB/LGBM)
    """
    # Imported lazily so the rest of this module is testable without the
    # competition data / the preprocessor utility script attached.
    from walmart_preprocessing import (          # noqa: E402  (Kaggle utility script)
        load_raw_data,
        merge_and_preprocess,
        make_masks,
        clean_target,
    )

    train, test, stores, features_df = load_raw_data(base_path)
    df = merge_and_preprocess(train, test, stores, features_df)
    tr_mask, val_mask, te_mask = make_masks(df, val_weeks)
    df = clean_target(df, tr_mask)  # adds Weekly_Sales_raw, clips train target

    split = np.full(len(df), "test", dtype=object)
    split[tr_mask.to_numpy()] = "train"
    split[val_mask.to_numpy()] = "val"

    panel = pd.DataFrame(
        {
            "Store": df["Store"].to_numpy(),
            "Dept": df["Dept"].to_numpy(),
            "Date": pd.to_datetime(df["Date"]).to_numpy(),
            "IsHoliday": df["IsHoliday"].astype(float).to_numpy(),
            "y": df["Weekly_Sales"].to_numpy(),
            "split": split,
        }
    )
    panel = panel.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------- #
# 2. The one and only scoring function (mirrors preprocessor w_val exactly)   #
# --------------------------------------------------------------------------- #
def compute_wmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
    holiday_weight: float = HOLIDAY_WEIGHT,
) -> float:
    """Weighted Mean Absolute Error.

    w = holiday_weight where IsHoliday==1 else 1.  Identical to the
    ``{1.0: 5.0, 0.0: 1.0}`` mapping the preprocessor builds for ``w_val``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.where(np.asarray(is_holiday) == 1, holiday_weight, 1.0)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


# --------------------------------------------------------------------------- #
# 3. Series construction + windowing (shared by PatchTST AND DLinear)         #
# --------------------------------------------------------------------------- #
class SeriesStore:
    """Reindexes every (Store, Dept) series onto one regular weekly grid.

    Decisions baked in here (documented once, reused by both deep models):

    * **Regular grid** -- windows are position-based, so a series with a gap
      would silently window across non-adjacent weeks. We therefore reindex
      each series onto the union weekly grid and time-interpolate short gaps
      for *input* construction. Training *targets* are still required to be
      originally observed, so the model is never asked to fit a fabricated
      value.
    * **Per-series stats** -- mean/std computed on the TRAIN region only
      (leak-free) and per (Store, Dept), because a small-format store and a
      supercenter live on totally different scales; one global scale would let
      big series dominate the loss.
    """

    def __init__(self, panel: pd.DataFrame):
        tv = panel[panel["split"].isin(["train", "val"])]
        self.grid: np.ndarray = np.sort(tv["Date"].unique())
        self.val_dates: np.ndarray = np.sort(
            panel.loc[panel["split"] == "val", "Date"].unique()
        )
        self.train_dates: np.ndarray = np.sort(
            panel.loc[panel["split"] == "train", "Date"].unique()
        )
        self.test_dates: np.ndarray = np.sort(
            panel.loc[panel["split"] == "test", "Date"].unique()
        )
        self._date_pos = {d: i for i, d in enumerate(self.grid)}
        self.n_val = len(self.val_dates)
        self.first_val_pos = self._date_pos[self.val_dates[0]]

        # grid-aligned per-week holiday flag (1 if any row that week is a
        # holiday). Lets us weight train-window WMAE the same 5x way val is
        # weighted, so train_wmae and val_wmae are on the same footing.
        self.holiday_by_pos: np.ndarray = (
            panel[panel["split"].isin(["train", "val"])]
            .groupby("Date")["IsHoliday"].max()
            .reindex(self.grid).fillna(0.0).to_numpy().astype(np.float32)
        )

        self.series: Dict[Tuple[int, int], dict] = {}
        self._build(panel)

    def _build(self, panel: pd.DataFrame) -> None:
        tv = panel[panel["split"].isin(["train", "val"])]
        for (store, dept), g in tv.groupby(["Store", "Dept"], sort=False):
            g = g.sort_values("Date")
            s = pd.Series(g["y"].to_numpy(), index=g["Date"].to_numpy())
            s = s.reindex(self.grid)                       # regular grid
            observed = s.notna().to_numpy()
            # interpolate short gaps for INPUT continuity only
            y_filled = (
                s.interpolate(method="linear", limit_direction="both")
                .to_numpy()
            )

            train_slice = slice(0, self.first_val_pos)
            train_obs = observed[train_slice]
            train_vals = y_filled[train_slice][train_obs]
            mu, sd = self._mu_sd(train_vals)

            # full-history stats (train+val) used only for the submission path
            full_vals = y_filled[observed]
            fmu, fsd = self._mu_sd(full_vals)

            self.series[(store, dept)] = {
                "y_filled": y_filled.astype(np.float32),
                "observed": observed,
                "mu": mu,
                "sd": sd,
                "full_mu": fmu,
                "full_sd": fsd,
                "n_train_obs": int(train_obs.sum()),
            }

    @staticmethod
    def _mu_sd(vals: np.ndarray) -> Tuple[float, float]:
        if vals.size >= 2:
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
        else:
            mu, sd = 0.0, 1.0
        return mu, (sd if sd > 1e-6 else 1.0)

    # -- training windows (pooled across ALL series = global model) --------- #
    def make_training_windows(
        self, input_len: int, pred_len: int, return_meta: bool = False
    ):
        """Return X (N, input_len) and Y (N, pred_len), per-series normalized.

        Only windows whose entire target lies inside the TRAIN region are kept
        (no val leakage), and whose target weeks were all originally observed.
        Every window is normalized by ITS OWN series' train mean/std, which is
        what makes this a channel-independent global model: each sample is one
        univariate window, pooled across all 3,000+ series into one model.

        With ``return_meta=True`` it also returns ``(mus, sds, hol)`` per window
        — the series mean/std used to normalize it, and the target weeks'
        holiday flags — so callers can inverse-transform predictions back to
        dollars and compute a holiday-weighted train WMAE.
        """
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        mus: List[float] = []
        sds: List[float] = []
        hols: List[np.ndarray] = []
        last_target_start = self.first_val_pos - pred_len  # exclusive of val
        for meta in self.series.values():
            y = meta["y_filled"]
            obs = meta["observed"]
            mu, sd = meta["mu"], meta["sd"]
            for i in range(0, last_target_start - input_len + 1):
                t0 = i + input_len
                t1 = t0 + pred_len
                if not obs[t0:t1].all():        # require observed targets
                    continue
                xs.append((y[i:t0] - mu) / sd)
                ys.append((y[t0:t1] - mu) / sd)
                if return_meta:
                    mus.append(mu)
                    sds.append(sd)
                    hols.append(self.holiday_by_pos[t0:t1])
        if not xs:
            X = np.empty((0, input_len), np.float32)
            Y = np.empty((0, pred_len), np.float32)
            if return_meta:
                return (X, Y, np.empty(0, np.float32), np.empty(0, np.float32),
                        np.empty((0, pred_len), np.float32))
            return X, Y
        X, Y = np.asarray(xs, np.float32), np.asarray(ys, np.float32)
        if return_meta:
            return (X, Y, np.asarray(mus, np.float32), np.asarray(sds, np.float32),
                    np.asarray(hols, np.float32))
        return X, Y

    # -- validation inputs (one window per predictable series) -------------- #
    def make_val_inputs(
        self, input_len: int
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, np.ndarray]:
        """Return (X_val, keys, mus, sds) for series that HAVE enough history.

        X_val[k] is the normalized ``input_len`` window ending right before the
        first val week for series ``keys[k]``. Inverse-transform predictions
        with mus[k]/sds[k]. Series without a full observed input window are
        omitted here and handled by :meth:`naive_val` instead.
        """
        xs, keys, mus, sds = [], [], [], []
        start = self.first_val_pos - input_len
        for key, meta in self.series.items():
            if start < 0:
                continue
            obs = meta["observed"][start:self.first_val_pos]
            if not obs.all():
                continue
            y = meta["y_filled"]
            mu, sd = meta["mu"], meta["sd"]
            xs.append((y[start:self.first_val_pos] - mu) / sd)
            keys.append(key)
            mus.append(mu)
            sds.append(sd)
        X = (np.asarray(xs, np.float32) if xs
             else np.empty((0, input_len), np.float32))
        return X, keys, np.asarray(mus, np.float32), np.asarray(sds, np.float32)

    def make_val_targets(
        self, keys: Sequence[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalized val targets + observed mask, aligned to ``keys``.

        Returns (Y_norm, mask), both (len(keys), n_val). ``Y_norm`` is each
        series' true val weeks normalized by ITS OWN train mean/std (so it lines
        up with a model's normalized output), and ``mask`` is 1 where the val
        week was actually observed. Used to compute a per-epoch val loss in the
        same units as the training loss, giving a train-vs-val overfitting curve.
        """
        vp = slice(self.first_val_pos, self.first_val_pos + self.n_val)
        Y, M = [], []
        for key in keys:
            meta = self.series[key]
            y = meta["y_filled"][vp]
            obs = meta["observed"][vp]
            Y.append((y - meta["mu"]) / meta["sd"])
            M.append(obs.astype(np.float32))
        if not Y:
            z = np.empty((0, self.n_val), np.float32)
            return z, z.copy()
        return np.asarray(Y, np.float32), np.asarray(M, np.float32)

    # -- naive fallback (same-week-last-year) ------------------------------- #
    def naive_val(self, key: Tuple[int, int]) -> np.ndarray:
        """Same-week-last-year forecast for the 8 val weeks of one series.

        Prediction for val week t = value 52 weeks earlier if observed, else
        the series' train mean. This is the SINGLE fallback method applied
        (identically) by all three TS notebooks for any series they cannot
        model natively, so the comparison always covers the same val rows.
        """
        meta = self.series[key]
        y, obs, mu = meta["y_filled"], meta["observed"], meta["mu"]
        out = np.empty(self.n_val, np.float32)
        for j in range(self.n_val):
            pos = self.first_val_pos + j
            ly = pos - 52
            if ly >= 0 and obs[ly]:
                out[j] = y[ly]
            else:
                out[j] = mu
        return out

    # ---------------- submission path (final iteration only) --------------- #
    # These use the FULL labeled history (train+val) as context and forecast
    # the genuinely-unknown test weeks. No leakage guard is needed because
    # nothing here is held out for scoring -- we simply use everything we know.
    def make_full_windows(
        self, input_len: int, pred_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Training windows over the ENTIRE labeled grid, full-history scaled."""
        xs, ys = [], []
        n = len(self.grid)
        for meta in self.series.values():
            y, obs = meta["y_filled"], meta["observed"]
            mu, sd = meta["full_mu"], meta["full_sd"]
            for i in range(0, n - pred_len - input_len + 1):
                t0, t1 = i + input_len, i + input_len + pred_len
                if not obs[t0:t1].all():
                    continue
                xs.append((y[i:t0] - mu) / sd)
                ys.append((y[t0:t1] - mu) / sd)
        if not xs:
            return (np.empty((0, input_len), np.float32),
                    np.empty((0, pred_len), np.float32))
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32)

    def make_forecast_inputs(
        self, input_len: int
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, np.ndarray]:
        """Last ``input_len`` observed window per series (full-history scaled)."""
        xs, keys, mus, sds = [], [], [], []
        n = len(self.grid)
        start = n - input_len
        for key, meta in self.series.items():
            if start < 0 or not meta["observed"][start:n].all():
                continue
            y = meta["y_filled"]
            mu, sd = meta["full_mu"], meta["full_sd"]
            xs.append((y[start:n] - mu) / sd)
            keys.append(key)
            mus.append(mu)
            sds.append(sd)
        X = (np.asarray(xs, np.float32) if xs
             else np.empty((0, input_len), np.float32))
        return X, keys, np.asarray(mus, np.float32), np.asarray(sds, np.float32)

    def naive_submit(self, key: Tuple[int, int]) -> np.ndarray:
        """Same-week-last-year forecast across all test weeks for one series."""
        meta = self.series[key]
        y, obs, mu = meta["y_filled"], meta["observed"], meta["full_mu"]
        n = len(self.grid)
        out = np.empty(len(self.test_dates), np.float32)
        for j in range(len(self.test_dates)):
            ly = n + j - 52          # position 52 weeks before this test week
            out[j] = y[ly] if (0 <= ly < n and obs[ly]) else mu
        return out


def build_submission(
    panel: pd.DataFrame,
    store: "SeriesStore",
    pred_by_key: Dict[Tuple[int, int], np.ndarray],
    floor_zero: bool = True,
) -> pd.DataFrame:
    """Map per-series test forecasts to the competition submission format.

    ``pred_by_key[(Store, Dept)]`` has length ``len(store.test_dates)`` aligned
    to ``store.test_dates``. Returns a DataFrame with columns ``Id`` (of the
    form ``Store_Dept_Date``) and ``Weekly_Sales``, covering exactly the test
    rows present in ``panel``.
    """
    test = panel[panel["split"] == "test"][["Store", "Dept", "Date"]].copy()
    date_of = {i: d for i, d in enumerate(store.test_dates)}
    rows = []
    for (s, d), preds in pred_by_key.items():
        for j, p in enumerate(preds):
            rows.append((s, d, date_of[j], float(p)))
    pred_df = pd.DataFrame(rows, columns=["Store", "Dept", "Date", "Weekly_Sales"])
    out = test.merge(pred_df, on=["Store", "Dept", "Date"], how="left")
    out["Weekly_Sales"] = out["Weekly_Sales"].fillna(0.0)
    if floor_zero:
        out["Weekly_Sales"] = out["Weekly_Sales"].clip(lower=0.0)
    ds = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out["Id"] = (out["Store"].astype(str) + "_" + out["Dept"].astype(str)
                 + "_" + ds)
    return out[["Id", "Weekly_Sales"]]


# --------------------------------------------------------------------------- #
# 4. Assemble a scored val frame from per-series predictions                  #
# --------------------------------------------------------------------------- #
def score_val_predictions(
    panel: pd.DataFrame,
    store: "SeriesStore",
    pred_by_key: Dict[Tuple[int, int], np.ndarray],
    subset: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[float, pd.DataFrame]:
    """Join per-series 8-week predictions onto the REAL val rows and score.

    ``pred_by_key[(Store, Dept)]`` is an array of length ``store.n_val`` aligned
    to ``store.val_dates``. We join by (Store, Dept, Date) onto the actual val
    rows (true y + IsHoliday) so WMAE is computed on exactly the rows XGB/LGBM
    used. Returns (wmae, merged_frame).

    ``subset`` restricts scoring to a given set of (Store, Dept) keys. This is
    what Prophet needs: it tunes on a *fixed sample* of series and must score
    WMAE on exactly that sample, not on the whole population (otherwise every
    un-sampled val row would join to a missing prediction and be counted as a
    zero-forecast, dwarfing the signal). PatchTST / DLinear predict every series
    (model + naive fallback), so they leave ``subset`` at ``None`` = score all.
    """
    val = panel[panel["split"] == "val"][
        ["Store", "Dept", "Date", "IsHoliday", "y"]
    ].copy()
    val = val.rename(columns={"y": "y_true"})

    if subset is not None:
        keep_keys = set(map(tuple, subset))
        pairs = zip(val["Store"].to_numpy(), val["Dept"].to_numpy())
        mask = np.fromiter((k in keep_keys for k in pairs),
                           dtype=bool, count=len(val))
        val = val[mask]

    rows = []
    date_of = {i: d for i, d in enumerate(store.val_dates)}
    for (s, d), preds in pred_by_key.items():
        for j, p in enumerate(preds):
            rows.append((s, d, date_of[j], float(p)))
    pred_df = pd.DataFrame(rows, columns=["Store", "Dept", "Date", "y_pred"])

    merged = val.merge(pred_df, on=["Store", "Dept", "Date"], how="left")
    # any val row with no prediction (unseen series) -> 0 as last resort
    merged["y_pred"] = merged["y_pred"].fillna(0.0)
    wmae = compute_wmae(
        merged["y_true"].to_numpy(),
        merged["y_pred"].to_numpy(),
        merged["IsHoliday"].to_numpy(),
    )
    return wmae, merged


# --------------------------------------------------------------------------- #
# 5. WMAE breakdown — overall + holiday / non-holiday split                   #
# --------------------------------------------------------------------------- #
def regression_metrics(
    y_true,
    y_pred,
    is_holiday=None,
    prefix: str = "",
) -> Dict[str, float]:
    """MAE, RMSE, R^2 (and WMAE when holiday flags are given) with a name prefix.

    Used to produce the ``train_*`` and ``val_*`` metric families a trial logs:
    ``regression_metrics(..., prefix="val_")`` -> ``val_mae/val_wmae/val_rmse/
    val_r2``. R^2 is the usual 1 - SS_res/SS_tot on the flattened targets; it can
    be negative if the model is worse than predicting the global mean, which for
    a sparse holiday-weighted panel is informative rather than a bug.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0:
        keys = ["mae", "rmse", "r2"] + (["wmae"] if is_holiday is not None else [])
        return {f"{prefix}{k}": float("nan") for k in keys}
    err = yt - yp
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    out = {f"{prefix}mae": round(mae, 2),
           f"{prefix}rmse": round(rmse, 2),
           f"{prefix}r2": round(r2, 4)}
    if is_holiday is not None:
        out[f"{prefix}wmae"] = round(
            compute_wmae(yt, yp, np.asarray(is_holiday).ravel()), 2)
    return out


def wmae_breakdown(merged: pd.DataFrame) -> Dict[str, float]:
    """Turn a scored ``merged`` frame into the metrics dict a trial logs.

    ``merged`` is exactly what :func:`score_val_predictions` returns: columns
    ``y_true``, ``y_pred``, ``IsHoliday``. We report:

    * ``val_wmae``      -- the headline number (5x holiday weight), the value
                           every leaderboard sorts on.
    * ``mae_holiday``   -- plain MAE on holiday rows only. With the 5x weight,
                           this is where most WMAE movement lives, so watching
                           it apart from ``mae_nonholiday`` tells you whether a
                           knob helped the (heavily weighted) holiday weeks or
                           the ordinary ones.
    * ``mae_nonholiday``-- plain MAE on the ordinary rows.

    Both group numbers are *unweighted* MAE (within a group the weight is
    constant, so a weighted mean equals the plain mean) — they decompose the
    error by regime; ``val_wmae`` recombines them with the competition weight.
    """
    yt = merged["y_true"].to_numpy(dtype=float)
    yp = merged["y_pred"].to_numpy(dtype=float)
    ish = merged["IsHoliday"].to_numpy() == 1

    w = np.where(ish, HOLIDAY_WEIGHT, 1.0)
    val_wmae = float(np.sum(w * np.abs(yt - yp)) / np.sum(w))

    def _mae(mask: np.ndarray) -> float:
        return float(np.abs(yt[mask] - yp[mask]).mean()) if mask.any() else float("nan")

    return {
        "val_wmae": round(val_wmae, 2),
        "mae_holiday": round(_mae(ish), 2),
        "mae_nonholiday": round(_mae(~ish), 2),
        "n_holiday": int(ish.sum()),
        "n_rows": int(len(merged)),
    }


# --------------------------------------------------------------------------- #
# 6. ExperimentTracker — a running, self-describing leaderboard               #
# --------------------------------------------------------------------------- #
class ExperimentTracker:
    """Accumulates ``(cfg, metrics)`` rows and makes the sweep legible.

    Design goals (why this exists rather than a bare list of dicts):

    * **One-knob methodology is enforced by the display, not by hand.** Start
      from a ``BASE`` config, change a single field per trial, and
      :meth:`summary` shows *only the columns that actually varied* — so "what
      changed" is literally the columns on screen, and a constant like
      ``pred_len`` never clutters the table.
    * **Every trial is judged against the best so far.** :meth:`log` prints the
      new row next to the incumbent, so you see immediately whether a change
      helped.
    * **Attribution is one call.** :meth:`diff` takes two trial ids and reports
      exactly which field moved and what each metric did — the payoff of the
      change-one-thing discipline.

    Trials are numbered from 1 in the order they are logged, so ``diff(1, k)``
    always means "baseline vs trial k".
    """

    def __init__(
        self,
        name: str,
        sort_key: str = "val_wmae",
        mlflow_experiment: Optional[str] = None,
    ):
        self.name = name
        self.sort_key = sort_key
        self.records: List[dict] = []           # flat rows for the DataFrame
        self._cfgs: Dict[int, dict] = {}        # trial id -> cfg
        self._metrics: Dict[int, dict] = {}     # trial id -> metrics
        self._notes: Dict[int, str] = {}
        self._history: Dict[int, list] = {}
        self._next_id = 1

        # -- optional MLflow mirror ---------------------------------------- #
        # If ``mlflow_experiment`` is set, every logged trial is ALSO pushed to
        # MLflow as its own run (params = cfg, metrics = metrics dict). Set the
        # tracking backend first with :func:`setup_dagshub_mlflow`. Logging is
        # best-effort: a network/MLflow failure prints a warning and never
        # crashes the sweep, so you always keep the local leaderboard.
        self.mlflow_experiment = mlflow_experiment
        self._mlflow = None
        if mlflow_experiment:
            try:
                import mlflow  # noqa: E402  (only imported when actually logging)
                mlflow.set_experiment(mlflow_experiment)
                self._mlflow = mlflow
                print(f"[{name}] MLflow ON -> experiment '{mlflow_experiment}'")
            except Exception as e:
                # No backend / no auth / offline: disable MLflow but keep the
                # local leaderboard fully functional.
                print(f"[{name}] MLflow OFF (setup failed: {type(e).__name__}: {e}) "
                      f"-- continuing with the local tracker only")

    # ---- logging ---------------------------------------------------------- #
    def log(self, cfg: dict, metrics: dict, note: str = "",
            history: Optional[List[dict]] = None) -> int:
        """Record one trial. ``metrics`` are the trial's summary scalars.

        ``history`` (optional) is a list of per-epoch dicts like
        ``{"epoch": e, "train_loss": ..., "val_loss": ...}``; when present and
        MLflow is on, each numeric field is logged with ``step=epoch`` so DagsHub
        renders it as a curve. Prophet passes ``history=None`` (no epochs).
        """
        tid = self._next_id
        self._next_id += 1
        self._cfgs[tid] = dict(cfg)
        self._metrics[tid] = dict(metrics)
        self._notes[tid] = note
        self._history[tid] = history or []
        self.records.append({"trial": tid, **cfg, **metrics, "note": note})

        if self._mlflow is not None:
            self._log_mlflow_run(tid, cfg, metrics, note, history)

        score = metrics.get(self.sort_key, float("nan"))
        best_id = min(self._metrics,
                      key=lambda t: self._metrics[t].get(self.sort_key, float("inf")))
        best = self._metrics[best_id].get(self.sort_key, float("nan"))
        hol = metrics.get("mae_holiday")
        non = metrics.get("mae_nonholiday")
        split = f" (hol {hol} / non {non})" if hol is not None else ""
        tag = "  <- new best" if best_id == tid else f"  | best: trial {best_id} @ {best}"
        note_s = f"  note='{note}'" if note else ""
        print(f"[{self.name} · trial {tid}] {self.sort_key}={score}{split}{tag}{note_s}")
        return tid

    # ---- leaderboard ------------------------------------------------------ #
    def _varied_cfg_keys(self) -> List[str]:
        all_keys = sorted({k for c in self._cfgs.values() for k in c})
        varied = []
        for k in all_keys:
            seen = {repr(self._cfgs[t].get(k)) for t in self._cfgs}
            if len(seen) > 1:
                varied.append(k)
        return varied

    def summary(self) -> pd.DataFrame:
        """Leaderboard: best ``sort_key`` first, only the knobs that varied."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        cfg_keys = sorted({k for c in self._cfgs.values() for k in c})
        # metric columns = everything that isn't a cfg key / trial / note,
        # kept in first-seen order for a stable, readable layout.
        metric_cols, seen = [], set()
        for rec in self.records:
            for k in rec:
                if k in ("trial", "note") or k in cfg_keys or k in seen:
                    continue
                seen.add(k)
                metric_cols.append(k)
        cols = ["trial"] + self._varied_cfg_keys() + metric_cols + ["note"]
        cols = [c for c in cols if c in df.columns]
        out = df[cols]
        if self.sort_key in out.columns:
            out = out.sort_values(self.sort_key)
        return out.reset_index(drop=True)

    # ---- attribution ------------------------------------------------------ #
    def diff(self, a: int, b: int) -> pd.DataFrame:
        """Print (and return) what changed in cfg between trials a and b and
        what each metric did. Metric rows show ``b - a`` deltas."""
        if a not in self._cfgs or b not in self._cfgs:
            raise KeyError(f"unknown trial id(s): {a}, {b} "
                           f"(have {sorted(self._cfgs)})")
        ca, cb = self._cfgs[a], self._cfgs[b]
        ma, mb = self._metrics[a], self._metrics[b]

        cfg_rows = []
        for k in sorted(set(ca) | set(cb)):
            if ca.get(k) != cb.get(k):
                cfg_rows.append({"field": k, f"trial_{a}": ca.get(k),
                                 f"trial_{b}": cb.get(k), "delta": ""})

        print(f"[{self.name}] diff  trial {a} ('{self._notes.get(a,'')}')"
              f"  ->  trial {b} ('{self._notes.get(b,'')}')")
        if not cfg_rows:
            print("  config: identical")
        else:
            print("  config changes:")
            for r in cfg_rows:
                print(f"    {r['field']:<24} {r[f'trial_{a}']!r:>14}  ->  {r[f'trial_{b}']!r}")

        metric_rows = []
        print("  metric changes (trial b - trial a):")
        for k in mb:
            va, vb = ma.get(k), mb.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = round(vb - va, 2)
                arrow = "▼" if d < 0 else ("▲" if d > 0 else "=")
                print(f"    {k:<24} {va!r:>12}  ->  {vb!r:<12}  Δ={d:+.2f} {arrow}")
                metric_rows.append({"field": k, f"trial_{a}": va,
                                    f"trial_{b}": vb, "delta": d})
        return pd.DataFrame(metric_rows)

    # ---- MLflow mirror ---------------------------------------------------- #
    @staticmethod
    def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
        """Keep only finite numeric metrics (MLflow rejects NaN / non-numbers).

        ``mae_holiday`` can be NaN when a scored subset happens to contain no
        holiday weeks, so this guard matters in practice.
        """
        out: Dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                out[k] = float(v)
        return out

    @staticmethod
    def _clean_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                for k, v in cfg.items()}

    def _log_mlflow_run(self, tid: int, cfg: dict, metrics: dict, note: str,
                        history: Optional[List[dict]] = None) -> None:
        mlflow = self._mlflow
        safe_note = note.replace(" ", "_").replace("(", "").replace(")", "")
        run_name = f"{self.name}_trial{tid:02d}" + (f"_{safe_note}" if safe_note else "")
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({"model": self.name, "trial": tid, "note": note})
                mlflow.log_params(self._clean_params(cfg))
                mlflow.log_metrics(self._clean_metrics(metrics))
                # per-epoch curves: one point per epoch -> DagsHub plots a line
                for h in (history or []):
                    step = int(h.get("epoch", 0))
                    for k, v in h.items():
                        if k == "epoch":
                            continue
                        if isinstance(v, (int, float)) and not (
                                isinstance(v, float) and math.isnan(v)):
                            mlflow.log_metric(k, float(v), step=step)
        except Exception as e:                                    # never kill the sweep
            print(f"  [mlflow warn] trial {tid} not logged: {type(e).__name__}: {e}")

    def log_artifacts_run(
        self,
        run_name: str,
        params: Optional[dict] = None,
        metrics: Optional[dict] = None,
        artifacts: Optional[Sequence[str]] = None,
        texts: Optional[Dict[str, str]] = None,
        tags: Optional[dict] = None,
    ) -> None:
        """Open one summary MLflow run for the final iteration: best config as
        params, full-population metrics, and files (submission.csv, the trained
        model, the experiment log) as artifacts. Mirrors how Cimbir logs a
        ``*_Best`` run. No-op with a note if MLflow is off."""
        if self._mlflow is None:
            print(f"[{self.name}] MLflow off -> skipping artifacts run '{run_name}'")
            return
        mlflow = self._mlflow
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({"model": self.name, "stage": "final", **(tags or {})})
                if params:
                    mlflow.log_params(self._clean_params(params))
                if metrics:
                    mlflow.log_metrics(self._clean_metrics(metrics))
                for fname, content in (texts or {}).items():
                    mlflow.log_text(content, fname)
                for path in (artifacts or []):
                    if path and os.path.exists(path):
                        mlflow.log_artifact(path)
                    else:
                        print(f"  [mlflow warn] artifact not found, skipped: {path}")
            print(f"[{self.name}] logged final artifacts run '{run_name}'")
        except Exception as e:
            print(f"  [mlflow warn] artifacts run '{run_name}' failed: "
                  f"{type(e).__name__}: {e}")

    # ---- persistence ------------------------------------------------------ #
    def to_csv(self, path: str) -> None:
        pd.DataFrame(self.records).to_csv(path, index=False)
        print(f"[{self.name}] wrote {len(self.records)} trials -> {path}")


# --------------------------------------------------------------------------- #
# 7. DagsHub / MLflow backend setup (headless-friendly for Kaggle & Colab)     #
# --------------------------------------------------------------------------- #
def setup_dagshub_mlflow(
    owner: str,
    repo: str,
    token: Optional[str] = None,
    experiment: Optional[str] = None,
) -> str:
    """Point MLflow at a DagsHub-hosted tracking server and return its URI.

    Two auth paths, chosen automatically:

    * **Token (preferred on Kaggle — no browser).** Pass ``token`` directly, or
      leave it ``None`` and set env ``DAGSHUB_TOKEN`` / ``MLFLOW_TRACKING_PASSWORD``.
      On Kaggle, store the token under *Add-ons -> Secrets* and read it with
      ``UserSecretsClient().get_secret("DAGSHUB_TOKEN")``. We then set the MLflow
      basic-auth env vars and the tracking URI directly.
    * **Interactive OAuth (Colab).** With no token available we fall back to
      ``dagshub.init(...)``, which opens the usual browser auth flow.

    ``owner``/``repo`` are YOUR DagsHub repo (create it once on dagshub.com and
    connect it) — not someone else's. The MLflow URI is
    ``https://dagshub.com/<owner>/<repo>.mlflow``.
    """
    import mlflow

    uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    token = token or os.environ.get("DAGSHUB_TOKEN") \
        or os.environ.get("MLFLOW_TRACKING_PASSWORD")

    if token:
        os.environ["MLFLOW_TRACKING_URI"] = uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(uri)
        auth = "token"
    else:
        import dagshub
        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)  # OAuth
        mlflow.set_tracking_uri(uri)
        auth = "oauth"

    if experiment:
        mlflow.set_experiment(experiment)
    print(f"MLflow -> {uri}  (auth: {auth})"
          + (f"  | experiment: {experiment}" if experiment else ""))
    return uri
