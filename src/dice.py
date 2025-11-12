# src/dice_cf.py
import numpy as np
import pandas as pd
import streamlit as st
import dice_ml

# ---------- Defaults you can override per call ----------
DEFAULT_ONEHOT_GROUPS = {
    "Contract": ["Contract_One year", "Contract_Two year"],
    "PaymentMethod": [
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
    ],
    "InternetService": [
        "InternetService_Fiber optic",
        "InternetService_No",   # add "InternetService_DSL" in datasets that have it
    ],
}

# ---------- One-hot helpers ----------
def _drop_by_prefixes(df: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    if df is None or df.empty or not prefixes:
        return df
    keep = [c for c in df.columns if not any(c.startswith(p) for p in prefixes)]
    return df[keep]

def _fix_onehot_row(row: pd.Series, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in row.index]
    if not present:
        return row
    vals = row[present].astype(float).values
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return row
    j = int(np.nanargmax(vals))
    row[present] = 0
    row[present[j]] = 1
    return row

def _sanitize_onehots(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for _, cols in (groups or {}).items():
        out = out.apply(lambda r: _fix_onehot_row(r, cols), axis=1)
    return out

def _assert_onehot_valid(df: pd.DataFrame, cols: list[str], name: str):
    present = [c for c in cols if c in df.columns]
    if not present:
        return
    bad = (df[present].astype(float).sum(axis=1) > 1.0 + 1e-8)
    if bad.any():
        raise ValueError(f"{name}: rows with >1 active category: {bad[bad].index.tolist()}")

def _drop_unchanged_onehots(cf_df, base_row, groups, atol=1e-8):
    out = cf_df.copy()
    for _, cols in (groups or {}).items():
        present = [c for c in cols if c in out.columns and c in base_row.index]
        if not present:
            continue
        changed = any(
            (~np.isclose(out[c].astype(float).values, float(base_row[c]), atol=atol, rtol=0.0)).any()
            for c in present
        )
        if not changed:
            out.drop(columns=present, inplace=True, errors="ignore")
    return out

def _apply_group_immutability(features_to_vary, immutable_features, groups):
    """Remove entire one-hot groups from features_to_vary if any column in the group is immutable."""
    if not isinstance(features_to_vary, (list, tuple)):
        return features_to_vary
    fv = set(features_to_vary) - set(immutable_features)
    for _, group_cols in (groups or {}).items():
        if any(c in immutable_features for c in group_cols):
            fv -= set(group_cols)
    return list(fv)

# ---------- Scaling helpers ----------
def _scale_bounds_from_scaler(permitted_range_raw, scaler, feature_names_in_order):
    if permitted_range_raw is None or scaler is None:
        return permitted_range_raw
    name_to_idx = {name: i for i, name in enumerate(feature_names_in_order)}
    scaled = {}
    for feat, (raw_min, raw_max) in permitted_range_raw.items():
        if feat not in name_to_idx:
            continue
        idx = name_to_idx[feat]
        mu  = float(scaler.mean_[idx])
        sd  = float(scaler.scale_[idx])
        zmin = (raw_min - mu) / sd
        zmax = (raw_max - mu) / sd
        scaled[feat] = [min(zmin, zmax), max(zmin, zmax)]
    return scaled

def _inverse_scale_cols(df, scaler, numeric_feature_names_in_scaler_order, cols_to_unscale=None):
    """Inverse-scale selected columns using StandardScaler stats."""
    if scaler is None or numeric_feature_names_in_scaler_order is None:
        return df.copy()
    df_out = df.copy()
    scaler_cols = list(numeric_feature_names_in_scaler_order)
    if cols_to_unscale is None:
        cols = [c for c in df_out.columns if c in scaler_cols]
    else:
        cols = [c for c in cols_to_unscale if c in df_out.columns and c in scaler_cols]
    idx = {name: i for i, name in enumerate(scaler_cols)}
    for c in cols:
        i = idx[c]
        mu = float(scaler.mean_[i])
        sd = float(scaler.scale_[i])
        if not pd.api.types.is_numeric_dtype(df_out[c]):
            continue
        df_out[c] = df_out[c].astype(float) * sd + mu
    return df_out

# ---------- Display table helpers ----------
def make_raw_cf_table(cf_df_scaled, base_row_scaled, *, scaler, numeric_feature_names_in_scaler_order,
                      only_changed=True, show_deltas=True, round_to=2):
    """Scaled CFs → RAW space; keep only changed columns; optional Δ columns."""
    cf_raw   = _inverse_scale_cols(cf_df_scaled, scaler, numeric_feature_names_in_scaler_order)
    base_raw = _inverse_scale_cols(base_row_scaled, scaler, numeric_feature_names_in_scaler_order)

    if not only_changed and not show_deltas:
        return cf_raw.round(round_to)

    base = base_raw.iloc[0]
    changed_cols = []
    atol = 1e-8
    for c in cf_raw.columns:
        if c not in base.index:
            changed_cols.append(c)
            continue
        a = cf_raw[c]
        b = base.get(c)
        if pd.isna(b):
            if a.notna().any():
                changed_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(a) and isinstance(b, (int, float, np.number)):
            if (~np.isclose(a.values.astype(float), float(b), atol=atol, rtol=0.0)).any():
                changed_cols.append(c)
        else:
            if (a.astype(object) != b).any():
                changed_cols.append(c)

    cols_to_show = changed_cols if only_changed else list(cf_raw.columns)
    out = cf_raw[cols_to_show].copy()

    if show_deltas:
        for c in cols_to_show:
            if pd.api.types.is_numeric_dtype(out[c]) and c in base.index:
                out[c + "_Δ"] = (out[c] - float(base[c]))

    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(round_to)
    return out

def _deduplicate_rows(df, tol=1e-6):
    if df.empty:
        return df
    df_num_rounded = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df_num_rounded[c] = df[c].round(6)
    return df_num_rounded.drop_duplicates(ignore_index=True)

def _normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def cf_to_dataframe(e, original_row, show_only_changes=True, atol: float = 1e-8):
    """Robustly extract CFs as a DataFrame without IPython dependency."""
    cf_df = None
    try:
        cf_df = e.cf_examples_list[0].final_cfs_df.copy()
    except Exception:
        pass
    if cf_df is None:
        try:
            cf_df = e.final_cfs_df.copy()
        except Exception:
            pass
    if cf_df is None or len(cf_df) == 0:
        return pd.DataFrame()

    # align to original row order where possible
    if isinstance(original_row, pd.DataFrame) and original_row.shape[0] == 1:
        base_cols = list(original_row.columns)
        inter = [c for c in base_cols if c in cf_df.columns]
        extras = [c for c in cf_df.columns if c not in inter]
        cf_df = cf_df[inter + extras]
    else:
        cf_df.columns = [str(c) for c in cf_df.columns]

    if not show_only_changes:
        return cf_df

    base = original_row.iloc[0]
    changed_cols = []
    for c in cf_df.columns:
        if c not in base.index:
            changed_cols.append(c); continue
        a = cf_df[c]; b = base.get(c)
        if np.issubdtype(a.dtype, np.number) and isinstance(b, (int, float, np.number)):
            if (~np.isclose(a.values.astype(float), float(b), atol=atol, rtol=0.0)).any():
                changed_cols.append(c)
        else:
            if (a.astype(object) != b).any():
                changed_cols.append(c)
    return cf_df[changed_cols] if changed_cols else cf_df

# ---------- Model helpers ----------
def _align_to_feature_order(df, feature_order, outcome_name):
    df2 = df.drop(columns=[outcome_name], errors="ignore")
    return df2.reindex(columns=feature_order)

def _model_predict_proba(model, X):
    """Return Nx2 proba for positive class in col 1. Supports sklearn + XGBoost Booster."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 1:
            p = np.vstack([1 - p, p]).T
        return p
    # bare XGBoost Booster fallback
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(X)
        p1 = model.predict(dm)  # assumes binary:logistic objective => probs
        return np.column_stack([1 - p1, p1])
    except Exception as ex:
        raise AttributeError(
            "Model does not support predict_proba. "
            "Wrap your XGBoost Booster with XGBBoosterProba (see bottom of this file)."
        ) from ex

def _append_prediction_columns(df_raw: pd.DataFrame,
                               cf_scaled_filled: pd.DataFrame,
                               base_row_scaled: pd.DataFrame,
                               model,
                               *,
                               feature_order,
                               outcome_name="Churn",
                               round_to: int = 3) -> pd.DataFrame:
    """Append prediction_score and prediction_change at the end of the RAW CF table."""
    if df_raw.empty:
        return df_raw

    cf_aligned   = _align_to_feature_order(cf_scaled_filled, feature_order, outcome_name)
    base_aligned = _align_to_feature_order(base_row_scaled, feature_order, outcome_name)

    cf_scores  = _model_predict_proba(model, cf_aligned)[:, 1]
    base_score = float(_model_predict_proba(model, base_aligned)[:, 1])

    out = df_raw.copy()
    out["prediction_score"]  = pd.Series(cf_scores, index=out.index).round(round_to)
    out["prediction_change"] = (out["prediction_score"] - base_score).round(round_to)
    return out

# ---------- DiCE explainer ----------
def build_dice_explainer(model_for_dice, X_train, y_train, outcome_name="Churn",
                         continuous_features=None, method="random"):
    if continuous_features is None:
        raise ValueError("Please provide `continuous_features` (list of truly continuous columns).")
    missing = sorted(set(continuous_features) - set(X_train.columns))
    if missing:
        raise ValueError(f"continuous_features not found in X_train: {missing}\nAvailable: {list(X_train.columns)}")

    data = dice_ml.Data(
        dataframe=pd.concat([X_train.reset_index(drop=True),
                             y_train.reset_index(drop=True).rename(outcome_name)], axis=1),
        continuous_features=list(continuous_features),
        outcome_name=outcome_name
    )
    model_dice = dice_ml.Model(model=model_for_dice, backend="sklearn")
    return dice_ml.Dice(data, model_dice, method=method)

# ---------- Main entry ----------
def get_counterfactual_analysis(
    y_test, X_test, X_train, y_train, model, continuous_features, counterfactual_tab, *,
    outcome_name="Churn",
    total_CFs=6,
    features_to_vary="all",
    permitted_range=None,
    scaler=None,
    numeric_feature_names_in_scaler_order=None,
    smoke_test_first=True,
    immutable_features=None,
    onehot_groups=None,
):
    """
    Creates counterfactuals for 1 predicted non-churner and 1 predicted churner.
    Displays CFs in RAW units with prediction deltas. Works for scaled or encoded data.
    """
    immutable_features = set(immutable_features or [])
    groups = onehot_groups or DEFAULT_ONEHOT_GROUPS

    # 0) Validate + scale bounds if provided in RAW units
    if continuous_features is None or not isinstance(continuous_features, (list, tuple)):
        raise ValueError("continuous_features must be a list/tuple of column names.")
    missing_cf = sorted(set(continuous_features) - set(X_train.columns))
    if missing_cf:
        raise ValueError(f"continuous_features not found in X_train columns: {missing_cf}\nAvailable: {list(X_train.columns)}")

    scaled_permitted_range = _scale_bounds_from_scaler(
        permitted_range_raw=permitted_range,
        scaler=scaler,
        feature_names_in_order=(numeric_feature_names_in_scaler_order or [])
    )

    # 1) Build explainer
    explainer = build_dice_explainer(
        model_for_dice=model,
        X_train=X_train,
        y_train=y_train,
        outcome_name=outcome_name,
        continuous_features=continuous_features,
        method="random"
    )

    # 1a) Remove immutable/grouped features from features_to_vary if list-like
    if isinstance(features_to_vary, (list, tuple)):
        features_to_vary = _apply_group_immutability(features_to_vary, immutable_features, groups)

    # 2) Choose one row per class at your current threshold (wrapped or plain)
    yhat = (_model_predict_proba(model, _align_to_feature_order(X_test, list(X_train.columns), outcome_name))[:, 1] >= 0.5).astype(int)
    idx0 = y_test.index[yhat == 0]
    idx1 = y_test.index[yhat == 1]
    if len(idx0) == 0 or len(idx1) == 0:
        st.error("Need at least one model-predicted 0 and one 1 at the current threshold.")
        return None
    i_non, i_churn = idx0[0], idx1[0]

    def _run(row_df):
        pred = int((_model_predict_proba(model, _align_to_feature_order(row_df, list(X_train.columns), outcome_name))[:, 1] >= 0.5)[0])
        desired = 1 - pred
        try:
            return explainer.generate_counterfactuals(
                query_instances=row_df,
                total_CFs=total_CFs,
                desired_class=desired,
                features_to_vary=features_to_vary,
                permitted_range=scaled_permitted_range
            )
        except Exception:
            if not smoke_test_first:
                raise
            return explainer.generate_counterfactuals(
                query_instances=row_df,
                total_CFs=max(total_CFs, 8),
                desired_class=desired,
                features_to_vary="all",
                permitted_range=None
            )

    # 3) Generate
    q_non   = X_test.loc[[i_non]]
    q_churn = X_test.loc[[i_churn]]

    e_non   = _run(q_non)
    e_churn = _run(q_churn)

    # 4) Compact scaled CFs
    df_non_scaled   = cf_to_dataframe(e_non,   q_non,   show_only_changes=False)
    df_churn_scaled = cf_to_dataframe(e_churn, q_churn, show_only_changes=False)

    # 5) Normalize, fill, sanitize, drop unchanged one-hot groups
    # non-churn branch
    df_non_scaled   = _normalize_cols(df_non_scaled)
    q_non_norm      = _normalize_cols(q_non)
    df_non_filled   = _fill_missing_cats_with_base(df_non_scaled, q_non_norm.iloc[0])
    df_non_filled   = _sanitize_onehots(df_non_filled, groups)
    _assert_onehot_valid(df_non_filled, groups.get("InternetService", []), "InternetService")
    _assert_onehot_valid(df_non_filled, groups.get("Contract", []), "Contract")
    df_non_clean    = _drop_unchanged_onehots(df_non_filled, q_non_norm.iloc[0], groups)

    # churn branch
    df_churn_scaled = _normalize_cols(df_churn_scaled)
    q_churn_norm    = _normalize_cols(q_churn)
    df_churn_filled = _fill_missing_cats_with_base(df_churn_scaled, q_churn_norm.iloc[0])
    df_churn_filled = _sanitize_onehots(df_churn_filled, groups)
    _assert_onehot_valid(df_churn_filled, groups.get("InternetService", []), "InternetService")
    _assert_onehot_valid(df_churn_filled, groups.get("Contract", []), "Contract")
    df_churn_clean  = _drop_unchanged_onehots(df_churn_filled, q_churn_norm.iloc[0], groups)

    # 6) Convert to RAW for display
    df_non_raw = make_raw_cf_table(
        cf_df_scaled=df_non_clean,
        base_row_scaled=q_non_norm,
        scaler=scaler,
        numeric_feature_names_in_scaler_order=numeric_feature_names_in_scaler_order,
        only_changed=True, show_deltas=False, round_to=2
    )
    df_churn_raw = make_raw_cf_table(
        cf_df_scaled=df_churn_clean,
        base_row_scaled=q_churn_norm,
        scaler=scaler,
        numeric_feature_names_in_scaler_order=numeric_feature_names_in_scaler_order,
        only_changed=True, show_deltas=False, round_to=2
    )

    # 7) Predictions on *scaled/encoded* versions; align to training order
    feature_order = list(X_train.columns)
    df_non_raw = _append_prediction_columns(
        df_raw=df_non_raw,
        cf_scaled_filled=df_non_filled,
        base_row_scaled=q_non_norm,
        model=model,
        feature_order=feature_order,
        outcome_name=outcome_name,
        round_to=3
    )
    df_churn_raw = _append_prediction_columns(
        df_raw=df_churn_raw,
        cf_scaled_filled=df_churn_filled,
        base_row_scaled=q_churn_norm,
        model=model,
        feature_order=feature_order,
        outcome_name=outcome_name,
        round_to=3
    )

    # 8) Optional: drop immutable cols from display and deduplicate
    def _drop_immutables(df):
        cols = [c for c in df.columns if c not in immutable_features]
        return df[cols]
    df_non_raw   = _deduplicate_rows(_drop_immutables(df_non_raw))
    df_churn_raw = _deduplicate_rows(_drop_immutables(df_churn_raw))

    HIDE_PREFIXES = ("bill_bucket_", "fail_count_bucket_")

    df_non_raw   = _drop_by_prefixes(df_non_raw, HIDE_PREFIXES)
    df_churn_raw = _drop_by_prefixes(df_churn_raw, HIDE_PREFIXES)

    # 9) Show
    with counterfactual_tab:
        col1, _ = st.columns([3.5, 1])
        col1.subheader("Non-churn example")
        col1.dataframe(df_non_raw, hide_index=True)
        col1.subheader("Churn example")
        col1.dataframe(df_churn_raw, hide_index=True)

    return {
        "index_non_churn_example": i_non,
        "index_churn_example": i_churn,
        "counterfactuals_for_non_churn": df_non_raw,
        "counterfactuals_for_churn": df_churn_raw
    }

# ---------- Utilities you may want ----------

def _fill_missing_cats_with_base(cf_df, base_row):
    """Fill NaNs & missing categorical/one-hot columns in the CF with base values."""
    out = cf_df.copy()
    for c in base_row.index:
        if c not in out.columns:
            out[c] = base_row[c]
    for c in out.columns:
        if c in base_row.index:
            out[c] = out[c].fillna(base_row[c])
    return out

# Wrap a raw XGBoost Booster to expose predict_proba
class XGBBoosterProba:
    """
    Usage:
        booster = xgb.train(...).  # or load_model(...)
        proba_model = XGBBoosterProba(booster, feature_order=list(X_train.columns))
        # optionally wrap with your ThresholdedModel
    """
    def __init__(self, booster, feature_order: list[str]):
        self.booster = booster
        self.feature_order = feature_order

    def predict_proba(self, X: pd.DataFrame):
        import xgboost as xgb
        X = X.reindex(columns=self.feature_order, copy=False)
        d = xgb.DMatrix(X, feature_names=self.feature_order)
        p = self.booster.predict(d)  # assumes binary:logistic
        return np.column_stack([1 - p, p])
