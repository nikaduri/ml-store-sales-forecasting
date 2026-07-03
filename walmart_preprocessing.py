from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", "{:.4f}".format)
np.random.seed(42)


VAL_WEEKS_DEFAULT: int = 8

HOLIDAYS = {
    "SuperBowl":    ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "LaborDay":     ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "Thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "Christmas":    ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}

MD_COLS    = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
MACRO_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]

NON_FEATURES = {
    "Date", "Weekly_Sales", "Weekly_Sales_raw", "is_train",
    "Type", "is_outlier",
}

NAN_THRESHOLD         = 0.60
CORR_THRESHOLD        = 0.97
FINAL_CORR_THRESHOLD  = 0.92
TARGET_CORR_THRESHOLD = 0.90
VARIANCE_THRESHOLD    = 1e-4

LEAKY_HIGH_CORR: List[str] = ["roll_max_52w"]


def load_raw_data(base_path: str) -> Tuple[pd.DataFrame, ...]:
    train    = pd.read_csv(base_path + "train.csv.zip")
    test     = pd.read_csv(base_path + "test.csv.zip")
    stores   = pd.read_csv(base_path + "stores.csv")
    features = pd.read_csv(base_path + "features.csv.zip")
    return train, test, stores, features


def merge_and_preprocess(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    stores: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    train = train.copy()
    test  = test.copy()

    train["is_train"]    = 1
    test["is_train"]     = 0
    test["Weekly_Sales"] = np.nan

    df = pd.concat([train, test], axis=0, ignore_index=True)
    df = df.merge(stores,      on="Store",                        how="left")
    df = df.merge(features_df, on=["Store", "Date", "IsHoliday"], how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    return df


def make_masks(
    df: pd.DataFrame,
    val_weeks: int = VAL_WEEKS_DEFAULT,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    max_train_date = df.loc[df["is_train"] == 1, "Date"].max()
    val_cutoff     = max_train_date - pd.Timedelta(weeks=val_weeks)

    tr_mask  = (df["is_train"] == 1) & (df["Date"] <= val_cutoff)
    val_mask = (df["is_train"] == 1) & (df["Date"] >  val_cutoff)
    te_mask  =  df["is_train"] == 0
    return tr_mask, val_mask, te_mask


def _iqr_bounds(s: pd.Series, k: float = 3.0) -> Tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean_target(df: pd.DataFrame, tr_mask: pd.Series) -> pd.DataFrame:
    df = df.copy()

    lo_map = (
        df[tr_mask]
        .groupby(["Store", "Dept"])["Weekly_Sales"]
        .apply(lambda s: _iqr_bounds(s)[0])
        .rename("iqr_lo")
    )
    hi_map = (
        df[tr_mask]
        .groupby(["Store", "Dept"])["Weekly_Sales"]
        .apply(lambda s: _iqr_bounds(s)[1])
        .rename("iqr_hi")
    )

    df["iqr_lo"] = df.set_index(["Store", "Dept"]).index.map(lo_map)
    df["iqr_hi"] = df.set_index(["Store", "Dept"]).index.map(hi_map)

    df["is_outlier"] = 0
    df.loc[tr_mask & (df["Weekly_Sales"] < df["iqr_lo"]), "is_outlier"] = -1
    df.loc[tr_mask & (df["Weekly_Sales"] > df["iqr_hi"]), "is_outlier"] =  1

    df["Weekly_Sales_raw"] = df["Weekly_Sales"].copy()

    df.loc[tr_mask, "Weekly_Sales"] = df.loc[tr_mask, "Weekly_Sales"].clip(
        lower=df.loc[tr_mask, "iqr_lo"],
        upper=df.loc[tr_mask, "iqr_hi"],
    )
    p01 = df.loc[tr_mask, "Weekly_Sales"].quantile(0.001)
    df.loc[tr_mask, "Weekly_Sales"] = df.loc[tr_mask, "Weekly_Sales"].clip(lower=p01)

    df.drop(columns=["iqr_lo", "iqr_hi"], inplace=True)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]     = df["Date"].dt.quarter
    df["WeekOfMonth"] = (df["Date"].dt.day - 1) // 7 + 1

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Week_sin"]  = np.sin(2 * np.pi * df["Week"]  / 52)
    df["Week_cos"]  = np.cos(2 * np.pi * df["Week"]  / 52)

    t0 = df["Date"].min()
    df["Weeks_elapsed"] = ((df["Date"] - t0).dt.days / 7).astype(int)
    
    df["Is_MonthStart"] = (df["Date"].dt.day <= 7).astype(int)
    df["Is_MonthEnd"]   = (df["Date"].dt.days_in_month - df["Date"].dt.day <= 7).astype(int)
    
    return df


def _signed_days_to_nearest(d: pd.Timestamp, hol_dates: pd.DatetimeIndex) -> int:
    return min([(d - h).days for h in hol_dates], key=abs)


def _easter_date(year: int) -> pd.Timestamp:
    a = year % 19; b = year // 100; c = year % 100
    d = b // 4;    e = b % 4;       f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4;    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day   = ((h + l - 7*m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for name, dates in HOLIDAYS.items():
        hol_dates = pd.to_datetime(dates)
        df[f"Days_to_{name}"]  = df["Date"].apply(
            _signed_days_to_nearest, hol_dates=hol_dates)
        df[f"Weeks_to_{name}"] = df[f"Days_to_{name}"] // 7
        df[f"Is_{name}"]       = (df[f"Days_to_{name}"] == 0).astype(int)
        df[f"Pre_{name}_2w"]   = (
            (df[f"Days_to_{name}"] >= -14) & (df[f"Days_to_{name}"] < 0)).astype(int)
        df[f"Post_{name}_1w"]  = (
            (df[f"Days_to_{name}"] > 0) & (df[f"Days_to_{name}"] <= 7)).astype(int)

    df["Is_BackToSchool"]   = df["Week"].between(31, 35).astype(int)
    df["Is_PreHalloween"]   = df["Week"].between(41, 43).astype(int)
    df["Is_ValentinesWeek"] = df["Week"].between(6,  7).astype(int)

    easter_dates = [_easter_date(y) for y in range(2010, 2014)]
    df["Days_to_Easter"] = df["Date"].apply(
        lambda d: min(abs((d - e).days) for e in easter_dates))
    df["Is_EasterWeek"]  = (df["Days_to_Easter"] <= 7).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g  = df.groupby(["Store", "Dept"])["Weekly_Sales_raw"]

    for w in [1, 2, 3, 4, 8, 13, 26, 39, 51, 52, 53, 56, 104]:
        df[f"lag_{w}w"] = g.shift(w)

    roll4    = g.shift(1).groupby([df["Store"], df["Dept"]]).transform(
        lambda x: x.rolling(4, min_periods=1).mean())
    roll4_ly = g.shift(53).groupby([df["Store"], df["Dept"]]).transform(
        lambda x: x.rolling(4, min_periods=1).mean())
    df["yoy_ratio_4w"] = roll4 / (roll4_ly + 1e-8)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df      = df.copy()
    shifted = df.groupby(["Store", "Dept"])["Weekly_Sales_raw"].shift(1)

    for window in [4, 8, 13, 26, 52]:
        base = shifted.groupby([df["Store"], df["Dept"]])
        df[f"roll_mean_{window}w"] = base.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f"roll_std_{window}w"]  = base.transform(
            lambda x: x.rolling(window, min_periods=1).std())
        df[f"roll_max_{window}w"]  = base.transform(
            lambda x: x.rolling(window, min_periods=1).max())
        df[f"roll_min_{window}w"]  = base.transform(
            lambda x: x.rolling(window, min_periods=1).min())

    df["roll_cv_13w"]    = df["roll_std_13w"]  / (df["roll_mean_13w"]  + 1e-8)
    df["momentum_4_26"]  = df["roll_mean_4w"]  / (df["roll_mean_26w"]  + 1e-8)
    df["momentum_13_52"] = df["roll_mean_13w"] / (df["roll_mean_52w"]  + 1e-8)
    return df


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in MD_COLS:
        df[f"{col}_present"] = df[col].notna().astype(int)
    for col in MD_COLS:
        df[col] = df[col].fillna(0)

    df["MD_total"]        = df[MD_COLS].sum(axis=1)
    df["MD_count"]        = df[[f"{c}_present" for c in MD_COLS]].sum(axis=1)
    df["MD_max"]          = df[MD_COLS].max(axis=1)
    df["MD_nonzero_mean"] = df[MD_COLS].replace(0, np.nan).mean(axis=1).fillna(0)
    df["MD_total_log"]    = np.log1p(df["MD_total"].clip(lower=0))

    df["MD_x_holiday"]      = df["MD_total"] * df["IsHoliday"].astype(int)
    df["MD3_x_thanksgiving"] = df["MarkDown3"] * df["Is_Thanksgiving"]
    df["MD3_x_Size"]        = df["MarkDown3"] * df["Size"]
    df["MD_x_christmas"]    = df["MD_total"] * df["Is_Christmas"]
    return df


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["store_mean"]           = df.groupby("Store")["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().mean())
    df["dept_mean"]            = df.groupby("Dept")["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().mean())
    df["type_mean"]            = df.groupby("Type")["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().mean())
    df["store_dept_mean"]      = df.groupby(["Store", "Dept"])["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().mean())
    df["store_dept_median"]    = df.groupby(["Store", "Dept"])["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().median())
    df["store_dept_std"]       = df.groupby(["Store", "Dept"])["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().std())
    df["dept_week_mean"]       = df.groupby(["Dept", "Week"])["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().mean())
    df["store_week_mean"]      = df.groupby(["Store", "Week"])["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().mean())
    df["store_dept_week_mean"] = df.groupby(["Store", "Dept", "Week"])["Weekly_Sales_raw"].transform(
                                     lambda x: x.shift(1).expanding().mean())

    df["dept_store_ratio"]     = df["store_dept_mean"] / (df["store_mean"] + 1e-8)
    df["dept_week_vs_annual"]  = df["dept_week_mean"]  / (df["dept_mean"]  + 1e-8)
    df["store_week_vs_annual"] = df["store_week_mean"] / (df["store_mean"] + 1e-8)
    df["sales_per_sqft"]       = df["store_dept_mean"] / (df["Size"]       + 1e-8)
    
    df["Thanksgiving_x_sdmean"] = df["Is_Thanksgiving"] * df["store_dept_mean"]
    df["SuperBowl_x_sdmean"]    = df["Is_SuperBowl"] * df["store_dept_mean"]
    return df


def add_macro_features(df: pd.DataFrame, tr_mask: pd.Series) -> pd.DataFrame:
    df = df.copy()

    for col in MACRO_COLS:
        df[col] = df.groupby("Store")[col].transform(lambda x: x.ffill())
        tr_med  = df.loc[tr_mask, col].median()
        df[col] = df[col].fillna(tr_med)

    macro_by_date = (
        df[["Store", "Date"] + MACRO_COLS]
        .drop_duplicates(subset=["Store", "Date"])
        .sort_values(["Store", "Date"])
        .copy()
    )
    for col in MACRO_COLS:
        macro_by_date[f"{col}_roll4"] = macro_by_date.groupby("Store")[col].transform(
            lambda x: x.rolling(4, min_periods=1).mean())
        macro_by_date[f"{col}_yoy_delta"] = (
            macro_by_date[col]
            - macro_by_date.groupby("Store")[col].shift(52)
        )

    roll_cols = [c for c in macro_by_date.columns
                 if c not in ["Store", "Date"] + MACRO_COLS]
    df = df.merge(macro_by_date[["Store", "Date"] + roll_cols],
                  on=["Store", "Date"], how="left")

    df["Unemp_x_MD"] = df["Unemployment"] * df["MD_total"]

    _, bins = pd.qcut(
        df.loc[tr_mask, "Temperature"], q=4, retbins=True, duplicates="drop")
    bins[0], bins[-1] = -np.inf, np.inf
    df["Temp_quartile"] = pd.cut(df["Temperature"], bins=bins, labels=False)

    return df


def _rolling_slope(series: pd.Series, window: int = 52) -> pd.Series:
    def _slope(y: np.ndarray) -> float:
        if len(y) < 3:
            return 0.0
        x   = np.arange(len(y), dtype=float)
        xm, ym = x.mean(), y.mean()
        den = ((x - xm) ** 2).sum()
        return ((x - xm) * (y - ym)).sum() / den if den != 0 else 0.0

    return series.rolling(window, min_periods=3).apply(_slope, raw=True)


def add_seasonal_and_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dept_annual          = df.groupby("Dept")["Weekly_Sales_raw"].transform(
                               lambda x: x.shift(1).expanding().mean())
    dept_week_expanding  = df.groupby(["Dept", "Week"])["Weekly_Sales_raw"].transform(
                               lambda x: x.shift(1).expanding().mean())
    df["dept_seasonal_idx"] = dept_week_expanding / (dept_annual + 1e-8)

    store_annual         = df.groupby("Store")["Weekly_Sales_raw"].transform(
                               lambda x: x.shift(1).expanding().mean())
    store_week_expanding = df.groupby(["Store", "Week"])["Weekly_Sales_raw"].transform(
                               lambda x: x.shift(1).expanding().mean())
    df["store_seasonal_idx"] = store_week_expanding / (store_annual + 1e-8)

    df["store_dept_trend"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales_raw"]
        .transform(lambda x: _rolling_slope(x.shift(1)))
        .fillna(0.0)
    )
    df["store_dept_trend_norm"] = df["store_dept_trend"] / (df["store_dept_mean"] + 1e-8)
    return df


def add_christmas_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Christmas_dow"]      = df["Date"].apply(
        lambda d: pd.Timestamp(year=d.year, month=12, day=25).dayofweek)
    df["Weeks_to_Christmas"] = df["Date"].apply(
        lambda d: (pd.Timestamp(year=d.year, month=12, day=25) - d).days // 7)
    df["In_Christmas_Zone"]  = (
        (df["Weeks_to_Christmas"] >= -4) & (df["Weeks_to_Christmas"] <= 1)
    ).astype(int)
    df["XmasZone_x_sdmean"]  = df["In_Christmas_Zone"] * df["store_dept_mean"]
    df["Type_enc"]           = df["Type"].map({"A": 0, "B": 1, "C": 2})
    return df


def clean_features(
    df: pd.DataFrame,
    tr_mask: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    candidates = [c for c in df.columns if c not in NON_FEATURES]

    nan_rates  = df.loc[tr_mask, candidates].isnull().mean()
    high_nan   = nan_rates[nan_rates > NAN_THRESHOLD].index.tolist()
    candidates = [c for c in candidates if c not in high_nan]

    tr_sub         = df.loc[tr_mask, candidates]
    stds           = tr_sub.std(numeric_only=True)
    constant_cols  = stds[stds == 0].index.tolist()
    near_const     = [
        c for c in candidates
        if c in stds.index and stds[c] > 0
        and tr_sub[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.99
    ]
    drop_const = list(set(constant_cols + near_const))
    candidates = [c for c in candidates if c not in drop_const]

    leaky      = [c for c in candidates if "Weekly_Sales_raw" in c]
    candidates = [c for c in candidates if c not in leaky]

    sample = df.loc[tr_mask, candidates].sample(
        n=min(20_000, int(tr_mask.sum())), random_state=42)
    num_cols    = sample.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = sample[num_cols].corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop_corr: set = set()
    for col in upper.columns:
        for partner in upper.index[upper[col] > CORR_THRESHOLD].tolist():
            if partner not in to_drop_corr:
                to_drop_corr.add(partner)
    candidates = [c for c in candidates if c not in to_drop_corr]

    fill_cols = [c for c in candidates if df[c].isnull().any()]
    lag_roll  = [
        c for c in fill_cols
        if c.startswith("lag_") or c.startswith("roll_")
        or "yoy" in c or "momentum" in c
    ]

    for col in lag_roll:
        df[col]    = df.groupby(["Store", "Dept"])[col].transform(lambda x: x.ffill())
        global_med = df.loc[tr_mask, col].median()
        df[col]    = df[col].fillna(global_med if not pd.isna(global_med) else 0)

    for col in [c for c in fill_cols if c not in lag_roll]:
        med    = df.loc[tr_mask, col].median()
        df[col] = df[col].fillna(med if not pd.isna(med) else 0)

    return df, candidates


def select_features(
    df: pd.DataFrame,
    tr_mask: pd.Series,
    features: List[str],
) -> List[str]:
    X_tr = df.loc[tr_mask, features].fillna(0)
    y_tr = df.loc[tr_mask, "Weekly_Sales"]

    vt      = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    vt.fit(X_tr)
    low_var  = [f for f, keep in zip(features, vt.get_support()) if not keep]
    features = [f for f in features if f not in low_var]

    corr_with_target = X_tr[features].corrwith(y_tr).abs().sort_values(ascending=False)
    high_target_corr = corr_with_target[
        corr_with_target > TARGET_CORR_THRESHOLD].index.tolist()
    if high_target_corr:
        print(f"[select_features] |r| > {TARGET_CORR_THRESHOLD} vs target — review:")
        for f in high_target_corr:
            print(f"  {f:45s}  r={corr_with_target[f]:.4f}")
    features = [f for f in features if f not in LEAKY_HIGH_CORR]

    sample_sel = df.loc[tr_mask, features].sample(
        n=min(20_000, int(tr_mask.sum())), random_state=0).fillna(0)
    corr_mat  = sample_sel.corr().abs()
    upper_sel = corr_mat.where(
        np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop_sel = {
        p
        for col in upper_sel.columns
        for p in upper_sel.index[upper_sel[col] > FINAL_CORR_THRESHOLD]
    }
    features = [f for f in features if f not in to_drop_sel]

    return features


@dataclass
class WalmartDataset:
    X_tr:     pd.DataFrame
    y_tr:     pd.Series
    X_val:    pd.DataFrame
    y_val:    pd.Series
    X_test:   pd.DataFrame
    w_val:    pd.Series
    features: List[str]
    df:       pd.DataFrame
    tr_mask:  pd.Series
    val_mask: pd.Series
    te_mask:  pd.Series


def build_dataset(
    base_path: str = "/kaggle/input/competitions/walmart-recruiting-store-sales-forecasting/",
    val_weeks: int = VAL_WEEKS_DEFAULT,
    verbose:   bool = True,
) -> WalmartDataset:
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log("Step 1/7 · Loading raw data…")
    train, test, stores, features_df = load_raw_data(base_path)
    _log(f"  train={train.shape}  test={test.shape}  "
         f"stores={stores.shape}  features={features_df.shape}")

    _log("Step 2/7 · Merging tables…")
    df = merge_and_preprocess(train, test, stores, features_df)
    _log(f"  Combined shape: {df.shape}")

    _log("Step 3/7 · Creating train / val / test masks…")
    tr_mask, val_mask, te_mask = make_masks(df, val_weeks)
    _log(f"  Train={tr_mask.sum():,}  Val={val_mask.sum():,}  Test={te_mask.sum():,}")

    _log("Step 4/7 · Cleaning target (train only)…")
    df = clean_target(df, tr_mask)
    _log(f"  Post-clean range: "
         f"{df.loc[tr_mask, 'Weekly_Sales'].min():,.1f} – "
         f"{df.loc[tr_mask, 'Weekly_Sales'].max():,.1f}")

    _log("Step 5/7 · Engineering features…")
    df = add_date_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_markdown_features(df)
    df = add_target_encoding(df)
    df = add_macro_features(df, tr_mask)
    df = add_seasonal_and_trend_features(df)
    df = add_christmas_features(df)
    _log(f"  Columns after feature engineering: {df.shape[1]}")

    _log("Step 6/7 · Cleaning features…")
    df, candidate_features = clean_features(df, tr_mask)
    _log(f"  Features after cleaning: {len(candidate_features)}")

    _log("Step 7/7 · Selecting features…")
    FEATURES = select_features(df, tr_mask, candidate_features)
    _log(f"  Final feature count: {len(FEATURES)}")

    X_tr   = df.loc[tr_mask,  FEATURES].reset_index(drop=True)
    y_tr   = df.loc[tr_mask,  "Weekly_Sales"].reset_index(drop=True)
    X_val  = df.loc[val_mask, FEATURES].reset_index(drop=True)
    y_val  = df.loc[val_mask, "Weekly_Sales"].reset_index(drop=True)
    X_test = df.loc[te_mask,  FEATURES].reset_index(drop=True)
    w_val  = (df.loc[val_mask, "IsHoliday"].astype(float)
              .map({1.0: 5.0, 0.0: 1.0})
              .reset_index(drop=True))

    _log(f"\n  X_tr={X_tr.shape}  X_val={X_val.shape}  X_test={X_test.shape}")
    _log("  Done ✓")

    return WalmartDataset(
        X_tr=X_tr,   y_tr=y_tr,
        X_val=X_val, y_val=y_val,
        X_test=X_test, w_val=w_val,
        features=FEATURES,
        df=df,
        tr_mask=tr_mask, val_mask=val_mask, te_mask=te_mask,
    )
