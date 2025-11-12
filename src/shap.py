import shap
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
import json, os, tempfile
import re
from src.customer import get_customers_for_explanation
from scipy.special import expit
from sklearn.metrics import roc_auc_score, auc
from scipy.stats import spearmanr, kendalltau
from joblib import Parallel, delayed
from src.helpers_shap import load_shap

def make_get_attribs_lr(lr_model, X_background, model_output="probability"):
    expl = shap.LinearExplainer(lr_model, X_background, model_output=model_output)

    def get_attribs(X):
        sv = expl(X)
        vals = getattr(sv, "values", sv)
        # if SHAP returns (n, f, 2) for binary, take positive class
        if vals.ndim == 3 and vals.shape[-1] == 2:
            vals = vals[:, :, 1]
        return np.asarray(vals)  # shape (n_samples, n_features)
    return get_attribs

def make_get_attribs_tree(model, X_background, model_output="probability"):
    expl = shap.TreeExplainer(model, data=X_background, model_output=model_output)
    bg_cols = list(X_background.columns)

    def get_attribs(X):
        # ensure DataFrame with training/background schema
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=bg_cols)
        else:
            # reindex to same schema just in case
            X = X.reindex(columns=bg_cols)
        sv = expl(X, check_additivity=False)
        vals = getattr(sv, "values", sv)
        if vals.ndim == 3 and vals.shape[-1] == 2:
            vals = vals[:, :, 1]
        return np.asarray(vals)
    return get_attribs

def plot_precomputed_shap(dataset: str, model: str, shap_tab, run_id="latest", top_n=15):
    values, base, cols, idx = load_shap(dataset, model, run_id)
    shap_df = pd.DataFrame(values, columns=cols, index=idx)
    top_feats = shap_df.abs().mean().sort_values(ascending=False).head(top_n).index
    long = (shap_df[top_feats].stack().rename("shap")
            .reset_index().rename(columns={"level_0":"row","level_1":"feature"}))
    order = long.groupby("feature")["shap"].apply(lambda s: s.abs().mean()).sort_values().index
    fig = px.strip(long, x="shap", y="feature", orientation="h",
                   hover_data={"row": True, "shap":":.4f"},
                   title=f"{dataset} · {model} — Global SHAP (precomputed)")
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=order))
    fig.update_traces(jitter=0.35, marker={"opacity":0.55, "size":4})
    with shap_tab:
        st.plotly_chart(fig, use_container_width=True)

        shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["SHAP", "LIME", "Counterfactuals"])
        plot_precomputed_shap("Telco", "LR", shap_tab)
        
def collect_attributions_for_tests(
    *,
    # --- Logistic Regression (scaled) ---
    lr_model=None,
    X_train_scaled_bg: pd.DataFrame | None = None,   # small background, same scaling as training
    X_val_scaled: pd.DataFrame | None = None,        # validation set (scaled)

    # --- Random Forest / XGB sklearn (encoded/unscaled) ---
    rf_model=None,
    xgb_clf=None,
    X_train_enc: pd.DataFrame | None = None,         # full train (encoded); we'll subsample for background
    X_val_enc: pd.DataFrame | None = None,           # validation (encoded)

    # --- XGB native Booster (optional) ---
    xgb_booster=None,                                # e.g., xgb_clf.get_booster() or a Booster you loaded
    use_streamlit: bool = True,                      # print shapes in Streamlit if available
    bg_sample: int = 200,
):
    """
    Build SHAP attribution callables and compute attributions for each model.

    Returns:
      dict with keys:
        'lr': np.ndarray or None                # shape (n_val, n_features_scaled)
        'rf': np.ndarray or None                # shape (n_val, n_features_encoded)
        'xgb': np.ndarray or None               # shape (n_val, n_features_encoded)
        'xgb_booster': np.ndarray or None       # shape (n_val, n_features_encoded) in Δ log-odds
      and a 'meta' sub-dict with basic info.
    """

    results = {"lr": None, "rf": None, "xgb": None, "xgb_booster": None, "meta": {}}

    # -------- LR (scaled) --------
    if lr_model is not None and X_train_scaled_bg is not None and X_val_scaled is not None:
        get_attr_lr = make_get_attribs_lr(lr_model, X_train_scaled_bg, model_output="probability")
        A_lr = get_attr_lr(X_val_scaled)  # shape (n_val, n_features_scaled)
        results["lr"] = A_lr
        results["meta"]["lr_shape"] = A_lr.shape

    # -------- RF (encoded) --------
    if rf_model is not None and X_train_enc is not None and X_val_enc is not None:
        X_bg_enc = X_train_enc.sample(min(bg_sample, len(X_train_enc)), random_state=0)
        get_attr_rf = make_get_attribs_tree(rf_model, X_bg_enc, model_output="probability")
        A_rf = get_attr_rf(X_val_enc)      # shape (n_val, n_features_encoded)
        results["rf"] = A_rf
        results["meta"]["rf_shape"] = A_rf.shape

    # -------- XGB sklearn wrapper (encoded) --------
    if xgb_clf is not None and X_train_enc is not None and X_val_enc is not None:
        X_bg_enc = X_train_enc.sample(min(bg_sample, len(X_train_enc)), random_state=0)
        get_attr_xgb = make_get_attribs_tree(xgb_clf, X_bg_enc, model_output="probability")
        A_xgb = get_attr_xgb(X_val_enc)    # shape (n_val, n_features_encoded)
        results["xgb"] = A_xgb
        results["meta"]["xgb_shape"] = A_xgb.shape

    # -------- XGB native Booster (encoded) --------
    if xgb_booster is not None and X_val_enc is not None:
        get_attr_boost = make_get_attribs_xgb_booster(xgb_booster)
        # pred_contribs expects numeric float32
        A_boost = get_attr_boost(X_val_enc.astype(np.float32))
        results["xgb_booster"] = A_boost   # Δ log-odds space
        results["meta"]["xgb_booster_shape"] = A_boost.shape

    return results

def _proba_fn(model):
    # sklearn-like
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]
    
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    return lambda X: booster.predict(xgb.DMatrix(X), output_margin=False)

# ========= XAI test utilities =========
def local_single_feature_impacts(
    model,
    x_row: pd.Series,                 # one row (features index-aligned)
    background: pd.DataFrame,         # small sample (e.g., 200 rows) from train/val
    n_draws: int = 64,                # MC samples for stability (keep small for speed)
) -> pd.Series:
    """
    Returns a Series: per-feature 'impact' = baseline_pred - mean_pred_with_feature_neutralized.
    Positive = prediction drops when we remove that feature's specific value (i.e., it was supporting the prediction).
    """
    proba = _proba_fn(model)

    # Baseline prediction for this row
    p0 = float(proba(x_row.to_frame().T))

    # Pre-sample indices to avoid Python loop overhead
    rng = np.random.RandomState(42)
    idx = rng.choice(len(background), size=n_draws, replace=True)

    impacts = {}
    for f in x_row.index:
        Xmc = pd.DataFrame([x_row.values] * n_draws, columns=x_row.index)
        # replace feature f with draws from empirical background distribution
        Xmc.loc[:, f] = background.iloc[idx][f].values
        p_mc = proba(Xmc.values) if not isinstance(Xmc, pd.DataFrame) else proba(Xmc)
        impacts[f] = p0 - float(np.mean(p_mc))
    return pd.Series(impacts).sort_values(ascending=False)

def local_spearman_rho(a_row: pd.Series, local_impacts: pd.Series) -> float:
    # align and use magnitudes (direction doesn’t matter for rank agreement)
    s_attr = a_row.abs().reindex(local_impacts.index)
    s_imp  = local_impacts.abs()
    rho, _ = spearmanr(s_attr.values, s_imp.values)
    return float(rho)

def local_deletion_curve(
    model,
    x_row: pd.Series,
    a_row: pd.Series,                 # local attribution values for x_row (same index)
    background: pd.DataFrame,
    n_steps: int = 10,
    replace_with: str = "mean",       # "mean" or "draw"
) -> dict:
    proba = _proba_fn(model)
    ranked = a_row.abs().sort_values(ascending=False).index.tolist()

    # baseline prob
    p0 = float(proba(x_row.to_frame().T))

    # precompute replacement values
    bg_mean = background.mean(numeric_only=True)
    rng = np.random.RandomState(0)

    ks = np.linspace(0, min(len(ranked), len(x_row)), n_steps, dtype=int)
    ks[0] = 0  # ensure starts at 0
    preds = []
    for k in ks:
        xk = x_row.copy()
        cols = ranked[:k]
        if replace_with == "mean":
            xk.loc[cols] = bg_mean.loc[cols]
        else:  # random draw per feature
            ridx = rng.randint(0, len(background))
            xk.loc[cols] = background.iloc[ridx][cols]
        preds.append(float(proba(xk.to_frame().T)))

    # AUC over normalized steps (higher drop -> smaller AUC -> better faithfulness)
    x_norm = ks / ks.max() if ks.max() > 0 else ks
    curve_auc = auc(x_norm, preds)
    return {"k": ks, "preds": preds, "auc": float(curve_auc), "p0": p0}

def _rank_by_mean_abs(A: np.ndarray, feature_names: list[str]) -> pd.Series:
    mean_abs = np.abs(A).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

def _permute_columns(X: pd.DataFrame, cols: list[str], rng: np.random.RandomState) -> pd.DataFrame:
    Xp = X.copy()
    for c in cols:
        Xp[c] = rng.permutation(Xp[c].values)
    return Xp

def local_faithfulness_report(
    model,
    x_row: pd.Series,
    a_row: pd.Series,            # local attributions for x_row
    background: pd.DataFrame,
    n_draws: int = 64,
    n_steps: int = 10,
    replace_with: str = "mean"
) -> dict:
    impacts = local_single_feature_impacts(model, x_row, background, n_draws=n_draws)
    rho = local_spearman_rho(a_row, impacts)
    delc = local_deletion_curve(model, x_row, a_row, background, n_steps=n_steps, replace_with=replace_with)
    # Optional flip with k features removed
    flipped_k = None
    for k, p in zip(delc["k"], delc["preds"]):
        if (x_row.name is None) or (k == 0):
            continue
        # Define your operating threshold as needed (example: 0.5)
        if (delc["p0"] >= 0.5) and (p < 0.5):
            flipped_k = int(k)
            break
        if (delc["p0"] < 0.5) and (p >= 0.5):
            flipped_k = int(k)
            break

    return {
        "rho_local": rho,
        "single_feature_impacts": impacts,
        "deletion_curve": delc,        # dict with k, preds, auc, p0
        "flip_k": flipped_k
    }

def faithfulness_report(
    model, X_val: pd.DataFrame, y_val: pd.Series | None,
    A: np.ndarray, feature_names: list[str],
    n_steps: int = 10, seed: int = 0, use_proba: bool = True
) -> dict:
    """
    1) Rank features by |SHAP|.
    2) Permute top-k features progressively and measure performance drop.
    3) Compute Spearman ρ between feature ranks and impact.
    Returns a dict with:
      - curve: DataFrame(k, metric_before, metric_after, delta)
      - rho: Spearman correlation (rank importance vs impact)
    """
    rng = np.random.RandomState(seed)
    # rank by average |SHAP|
    rank_s = np.abs(A).mean(axis=0)
    rank_s = pd.Series(rank_s, index=feature_names).sort_values(ascending=False)

    ranked_feats = rank_s.index.tolist()

    # baseline outputs
    if use_proba and hasattr(model, "predict_proba"):
        p0 = model.predict_proba(X_val)[:, 1]
        metric0 = p0.mean() if y_val is None else roc_auc_score(y_val, p0)
    else:
        yhat0 = model.predict(X_val)
        metric0 = (yhat0.mean() if y_val is None else (yhat0 == y_val).mean())

    # progressive permutation curve
    ks = np.linspace(1, min(len(ranked_feats), X_val.shape[1]), n_steps, dtype=int)
    rows = []
    impacts = []
    for k in ks:
        cols = ranked_feats[:k]
        Xp = _permute_columns(X_val, cols, rng)
        if use_proba and hasattr(model, "predict_proba"):
            p = model.predict_proba(Xp)[:, 1]
            metric = p.mean() if y_val is None else roc_auc_score(y_val, p)
        else:
            yhat = model.predict(Xp)
            metric = (yhat.mean() if y_val is None else (yhat == y_val).mean())
        rows.append({"k": k, "metric": metric, "delta": metric0 - metric})
        impacts.append(metric0 - metric)

    curve = pd.DataFrame(rows)

    # correlation between rank (1..n) and per-feature single-permutation impact
    # (rough, but useful): permute one at a time
    single_rows = []
    for i, f in enumerate(ranked_feats):
        Xp = _permute_columns(X_val, [f], rng)
        if use_proba and hasattr(model, "predict_proba"):
            p = model.predict_proba(Xp)[:, 1]
            m = p.mean() if y_val is None else roc_auc_score(y_val, p)
        else:
            yhat = model.predict(Xp)
            m = (yhat.mean() if y_val is None else (yhat == y_val).mean())
        single_rows.append({"feature": f, "impact": metric0 - m})
    single_df = pd.DataFrame(single_rows).set_index("feature").reindex(ranked_feats)
    rank_positions = np.arange(1, len(ranked_feats) + 1)
    rho, _ = spearmanr(rank_positions, np.abs(single_df["impact"].values))

    return {"curve": curve, "rho": rho, "rank": rank_s, "single_impacts": single_df}

def stability_report(
    get_attribs_callable, X_val: pd.DataFrame,
    n_boot: int = 10, sample_frac: float = 0.8, seed: int = 0
) -> dict:
    """
    Resample X_val multiple times, recompute attributions, and report
    Kendall’s τ between per-run feature rankings.
    """
    rng = np.random.RandomState(seed)
    ranks = []
    for b in range(n_boot):
        idx = rng.choice(len(X_val), size=max(10, int(sample_frac*len(X_val))), replace=False)
        Ab = get_attribs_callable(X_val.iloc[idx])
        r = pd.Series(np.abs(Ab).mean(axis=0), index=X_val.columns).rank(ascending=False, method="average")
        ranks.append(r)
    # pairwise Kendall τ
    taus = []
    for i in range(n_boot):
        for j in range(i+1, n_boot):
            tau, _ = kendalltau(ranks[i], ranks[j])
            taus.append(tau)
    return {"kendall_tau_mean": float(np.mean(taus)), "kendall_tau_std": float(np.std(taus)), "n_pairs": len(taus)}

def sanity_report(
    model, X_val: pd.DataFrame, y_val: pd.Series | None,
    get_attribs_callable,
    randomize: str = "model", seed: int = 0
) -> dict:
    """
    Randomization sanity checks (Adebayo et al.):
      - randomize='model': shuffle model weights by refitting on shuffled labels.
      - randomize='labels': shuffle y and refit.
    Reports correlation between original mean|SHAP| ranking and randomized one.
    """
    rng = np.random.RandomState(seed)

    # original ranking
    A0 = get_attribs_callable(X_val)
    r0 = pd.Series(np.abs(A0).mean(axis=0), index=X_val.columns)

    # randomized
    if randomize == "labels" and y_val is not None and hasattr(model, "fit"):
        y_shuf = y_val.sample(frac=1.0, random_state=seed).values
        m2 = type(model)()  # assumes sklearn-like
        m2.fit(X_val, y_shuf)
        def get_attr2(X):  # rebuild an explainer against the randomized model
            from shap import TreeExplainer, LinearExplainer
            if hasattr(m2, "predict_proba") and not hasattr(m2, "estimators_"):
                ex = LinearExplainer(m2, X_val.sample(min(200, len(X_val)), random_state=seed), model_output="probability")
                sv = ex(X); v = getattr(sv, "values", sv); 
                if v.ndim == 3 and v.shape[-1] == 2: v = v[:, :, 1]
                return np.asarray(v)
            else:
                ex = TreeExplainer(m2, data=X_val.sample(min(200, len(X_val)), random_state=seed), model_output="probability")
                sv = ex(X, check_additivity=False); v = getattr(sv, "values", sv)
                if v.ndim == 3 and v.shape[-1] == 2: v = v[:, :, 1]
                return np.asarray(v)
        A1 = get_attr2(X_val)
    else:
        # model-randomization fallback: permute columns inside explainer call
        X_rand = X_val.copy()
        for c in X_rand.columns:
            X_rand[c] = rng.permutation(X_rand[c].values)
        A1 = get_attribs_callable(X_rand)

    r1 = pd.Series(np.abs(A1).mean(axis=0), index=X_val.columns)
    rho, _ = spearmanr(r0.rank(ascending=False), r1.rank(ascending=False))
    return {"spearman_rho_vs_original": float(rho)}


def lr_local_shap_by_truth(
    lr_model,
    background_data,                 # scaled, same preprocessing as model training
    X_test_scaled: pd.DataFrame,     # scaled, same columns/order as model input
    X_test: pd.DataFrame,            # ORIGINAL/HUMAN-READABLE values, same index as X_test_scaled
    y_test: pd.Series,               # 0/1, index aligned with X_test_scaled
    shap_tab,
    threshold: float = 0.5,
    top_display: int = 12,
    title_prefix: str = "Local SHAP – Logistic Regression Explanation",
    value_format: dict | None = None # optional: {"MonthlyCharges":"{:.2f}", "tenure":"{:.0f}"}
):
    """Explain the first non-churner (0) and first churner (1) in y_test, showing real feature values (table without SHAP column)."""

    # --- 1) pick rows by TRUE label (by index/label)
    sel_labels = get_customers_for_explanation(y_test)

    # --- 2) predict & SHAP for those exact rows (keep order of sel_labels)
    X_sel_scaled = X_test_scaled.loc[sel_labels]
    proba_sel = lr_model.predict_proba(X_sel_scaled)[:, 1]

    expl = shap.LinearExplainer(lr_model, background_data, model_output="probability")
    sv = expl(X_sel_scaled)  # returns in same order as X_sel_scaled

    with shap_tab:

        st.markdown(
                "### Local Explanation"
            )
        
        for k, lbl in enumerate(sel_labels):
            pred = float(proba_sel[k])
            true = int(y_test.loc[lbl])

            # contributions
            vals = sv.values[k] if hasattr(sv, "values") else sv[k]
            base = sv.base_values[k] if np.ndim(sv.base_values) else float(sv.base_values)

            # sort by |impact|
            s = (pd.Series(vals, index=X_sel_scaled.columns)
                    .sort_values(key=lambda x: x.abs(), ascending=False))
            top = s.head(top_display)
            other = s.iloc[top_display:].sum() if len(s) > top_display else 0.0

            # --- Build the table with REAL values from X_test
            feat_list = list(top.index)
            # fill None/NaN with 0 in displayed feature values
            values_row = (X_test.loc[lbl].reindex(feat_list)
                                   .fillna(0)
                                   .replace({None: 0}))

            # TABLE: drop SHAP column (only show Feature + Value)
            df_top = pd.DataFrame({
                "Feature": feat_list,
                "Value":   values_row.values
            })
            if other != 0:
                # keep "other" row for completeness; value shown as em-dash
                df_top.loc[len(df_top)] = ["other", "—"]

            # Display header + table
            st.markdown(
                f"#### **Customer id `{lbl}`** — Churn={true} | prediction={pred:.3f} "
                f"→ **{'CHURN' if pred>=threshold else 'STAY'}**"
            )

            local_col1, local_col2, local_col3, local_col4= st.columns([1, 0.5, 1.5, 0.5])

            fmt = {}
            if value_format:
                # only apply provided formats to columns that exist in the table
                fmt.update({k: v for k, v in value_format.items() if k in df_top.columns})
            local_col1.dataframe(df_top.style.format(fmt), use_container_width=True)

            # --- Plotly Express contribution bar (built directly from SHAP 'top')
            df_bar = pd.DataFrame({
                "feature": feat_list + (["other"] if other != 0 else []),
                "delta":   list(top.values) + ([other] if other != 0 else [])
            })
            fig = px.bar(
                df_bar,
                x="delta", y="feature", orientation="h",
                color="delta",
                color_continuous_scale=px.colors.diverging.RdBu_r,  # blue=neg, red=pos
                range_color=(df_bar["delta"].min(), df_bar["delta"].max()),
                title=f"{title_prefix} — Row {lbl} (Churn={true}, prediction={pred:.3f})",
            )
            fig.update_layout(
                xaxis_title="SHAP weight",
                yaxis_title="",
                coloraxis_colorbar_title="Weight",
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False
            )
            fig.add_vline(x=0, line_width=1, line_dash="dash", opacity=0.6)

            # annotate base & SHAP reconstruction
            pred_from_shap = float(base + s.sum())
            fig.add_annotation(x=base, y=1.02*len(df_bar), text=f"base={base:.3f}",
                               showarrow=False, yref="y domain")
            fig.add_annotation(x=pred_from_shap, y=1.08*len(df_bar),
                               text=f"pred≈{pred_from_shap:.3f}",
                               showarrow=False, yref="y domain")

            local_col3.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def _make_tree_explainer(_model, X_bg):   # <- underscore = don't hash
    return shap.TreeExplainer(_model, data=X_bg, model_output="probability")

@st.cache_data
def _explain(_explainer, X):   # <- underscore!
    sv = _explainer(X, check_additivity=False)
    vals = getattr(sv, "values", sv)
    if vals.ndim == 3 and vals.shape[-1] == 2:
        vals = vals[:, :, 1]
    return vals, sv.base_values

def get_rf_explanation(
    rf_model,
    X_background,
    X_test,
    shap_tab,
    top_n=15,
    bg_max=200,           # cap background size
    test_max=1000,        # cap test size for plotting
    per_feature_cap=500,  # cap points per feature in the beeswarm
    seed=42
):
    """
    Fast SHAP beeswarm for RandomForest.
    Pass *unscaled* numeric features in X_background and X_test.
    """

    # ---- 1) Subsample background & test to bounded sizes ----
    if len(X_background) > bg_max:
        X_bg = X_background.sample(n=bg_max, random_state=seed)
    else:
        X_bg = X_background

    if len(X_test) > test_max:
        X_te = X_test.sample(n=test_max, random_state=seed)
    else:
        X_te = X_test

    # Ensure index is simple range (helps later joins/reshapes)
    X_te = X_te.reset_index(drop=True)

    # ---- 2) Build / cache explainer and compute SHAP (raw, fast path) ----
    expl = _make_tree_explainer(rf_model, X_bg)
    shap_vals, base_vals = _explain(expl, X_te)   # shap_vals: [n_rows, n_features]

    # ---- 3) Pick top features without stacking everything ----
    mean_abs = np.abs(shap_vals).mean(axis=0)  # [n_features]
    feat_names = np.asarray(X_te.columns)
    top_idx = np.argsort(-mean_abs)[:top_n]
    top_feats = feat_names[top_idx]

    # Restrict arrays to top features
    shap_top = shap_vals[:, top_idx]                      # [n_rows, top_n]
    vals_top = X_te[top_feats].to_numpy(copy=False)       # [n_rows, top_n]
    n_rows, n_feats = shap_top.shape

    # ---- 4) (Optional) Cap points per feature for plotting speed ----
    if per_feature_cap and n_rows > per_feature_cap:
        # same subset for all features to keep row ids consistent
        sel = np.random.RandomState(seed).choice(n_rows, size=per_feature_cap, replace=False)
        shap_top = shap_top[sel, :]
        vals_top = vals_top[sel, :]
        n_rows = shap_top.shape[0]

    # ---- 5) Build compact long-form in O(n_rows * top_n) via reshape ----
    long = pd.DataFrame({
        "row": np.repeat(np.arange(n_rows), n_feats),
        "feature": np.tile(top_feats, n_rows),
        "shap": shap_top.ravel(),
        "value": pd.to_numeric(vals_top.ravel(), errors="coerce")
    })

    # ---- 6) Plot (scatter/strip). Scattergl helps on very large point counts. ----
    fig_beeswarm = px.strip(
        long, x="shap", y="feature", color="value", orientation="h",
        hover_data={"row": True, "value": ":.3f", "shap": ":.4f"},
        title="Global Feature impact on churn"
    )
    fig_beeswarm.update_traces(jitter=0.35, marker={"opacity": 0.55, "size": 4})
    fig_beeswarm.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=list(top_feats)),
        yaxis_title="Feature",
        xaxis_title="Impact on churn probability",
        coloraxis_showscale=False
    )
    fig_beeswarm.update_layout(coloraxis=dict(cmin=0, cmax=1))

    with shap_tab:
        st.markdown(
                "### Global Explanation"
            )
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(fig_beeswarm, use_container_width=True)

    return fig_beeswarm

def rf_local_shap_by_truth(
    rf_model,
    X_test_enc: pd.DataFrame,          # unscaled/encoded features for RF (same cols/order as training)
    X_values: pd.DataFrame,            # human-readable values for the same rows (same index)
    y_test: pd.Series,
    threshold,                 # 0/1, index aligned with X_test_enc
    shap_tab,     # your helper that returns [one_0, one_1] labels
    background: pd.DataFrame | None = None,  # optional small background (100–200 rows, same features)
    top_display: int = 12,
    title_prefix: str = "Local SHAP – Random Forest Explanation",
    value_format: dict | None = None    # e.g., {"MonthlyCharges":"{:.2f}","tenure":"{:.0f}"}
):
    """
    Explain the first non-churner (0) and first churner (1) in y_test (by true label) for RF,
    showing a Feature|Value table (no SHAP column) and a bar chart of Δ log-odds (SHAP).
    """

    # --- 1) which customers to explain (one 0, one 1 by TRUE label)
    sel_labels = get_customers_for_explanation(y_test)

    # --- 2) background for TreeExplainer (keep small, from TRAIN ideally)
    if background is None:
        # fall back to a small sample from the test set (still OK for local plots)
        background = X_test_enc.sample(min(200, len(X_test_enc)), random_state=42)

    # --- 3) explainer (RAW = log-odds, fast path) and predictions
    expl = shap.TreeExplainer(rf_model, data=background, model_output="probability")
    X_sel = X_test_enc.loc[sel_labels]
    sv = expl(X_sel, check_additivity=False)        # returns in same order as X_sel
    proba_sel = rf_model.predict_proba(X_sel)[:, 1] # predicted churn probabilities

    with shap_tab:
        st.markdown("### Local Explanation")

        for k, lbl in enumerate(sel_labels):
            pred = float(proba_sel[k])
            true = int(y_test.loc[lbl])

            # per-sample SHAP arrays (may be [n_feat, 2] for a classifier)
            vals = sv.values[k] if hasattr(sv, "values") else sv[k]
            base = sv.base_values[k] if np.ndim(sv.base_values) else float(sv.base_values)

            # If it's multi-class (binary), take the positive class (index 1)
            if vals.ndim == 2 and vals.shape[1] == 2:
                vals = vals[:, 1]
            if np.ndim(base) == 1 and len(base) == 2:
                base = float(base[1])

            # Now this is 1-D: length = n_features
            s = pd.Series(vals, index=X_sel.columns).sort_values(key=lambda x: x.abs(), ascending=False)
            top = s.head(top_display)
            other = s.iloc[top_display:].sum() if len(s) > top_display else 0.0

            # --- Table with REAL values (no SHAP column), fill None/NaN -> 0
            feat_list = list(top.index)
            values_row = (X_values.loc[lbl].reindex(feat_list).fillna(0).replace({None: 0}))
            df_top = pd.DataFrame({"Feature": feat_list, "Value": values_row.values})
            if other != 0:
                df_top.loc[len(df_top)] = ["other", "—"]

            # Header
            st.markdown(
                f"#### **Customer id `{lbl}`** — Churn={true} | prediction={pred:.3f} "
                f"→ **{'CHURN' if pred>=threshold else 'STAY'}**"
            )

            # Layout: table + bar chart
            local_col1, spacer, local_col2, spacer2 = st.columns([1, 0.5, 1.5, 0.5])

            # Table formatting
            fmt = {}
            if value_format:
                fmt.update({k: v for k, v in value_format.items() if k in df_top.columns})
            local_col1.dataframe(df_top.style.format(fmt), use_container_width=True)

            # --- Contribution bar chart in RAW units (Δ log-odds)
            df_bar = pd.DataFrame({
                "feature": feat_list + (["other"] if other != 0 else []),
                "delta":   list(top.values) + ([other] if other != 0 else [])
            })
            fig = px.bar(
                df_bar,
                x="delta", y="feature", orientation="h",
                color="delta",
                color_continuous_scale=px.colors.diverging.RdBu_r,   # blue=neg, red=pos
                range_color=(df_bar["delta"].min(), df_bar["delta"].max()),
                title=f"{title_prefix} — Row {lbl} (Churn={true}, prediction={pred:.3f})",
            )
            fig.update_layout(
                xaxis_title="SHAP weight",    # raw SHAP for trees
                yaxis_title="",
                coloraxis_colorbar_title="Weight",
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False
            )
            fig.add_vline(x=0, line_width=1, line_dash="dash", opacity=0.6)

            # (Optional) show base and reconstruction in raw units
            pred_raw_from_shap = float(base + s.sum())
            fig.add_annotation(x=base, y=1.02*len(df_bar), text=f"base(raw)={base:.3f}",
                               showarrow=False, yref="y domain")
            fig.add_annotation(x=pred_raw_from_shap, y=1.08*len(df_bar),
                               text=f"sum(raw)≈{pred_raw_from_shap:.3f}",
                               showarrow=False, yref="y domain")

            local_col2.plotly_chart(fig, use_container_width=True)

def _normalize_base_score_value(val):
    # Accept strings like "[5E-1]" / "[0.5]" / "0.5" / "5e-1"
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        return str(float(s))  # raises if not numeric
    if isinstance(val, (int, float)):
        return str(float(val))
    # e.g., a list like ["5E-1"]
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], str):
        return str(float(val[0]))
    raise ValueError(f"Unsupported base_score value type: {type(val)} -> {val!r}")

def _fix_base_score_json_inplace(booster) -> None:
    """
    Ensure all occurrences of base_score in booster config are plain numeric strings (e.g. "0.5").
    Works across different xgboost config layouts.
    """
    cfg_txt = booster.save_config()
    try:
        cfg = json.loads(cfg_txt)
    except Exception as e:
        # In case some versions output nonstandard JSON (rare)
        raise RuntimeError("Could not parse booster config as JSON") from e

    before_examples = []

    def _walk_and_fix(node):
        changed_local = False
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k == "base_score":
                    before_examples.append(v)
                    try:
                        node[k] = _normalize_base_score_value(v)
                    except Exception:
                        node[k] = "0.5"
                    changed_local = True
                else:
                    if _walk_and_fix(v):
                        changed_local = True
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if _walk_and_fix(v):
                    changed_local = True
        return changed_local

    changed = _walk_and_fix(cfg)

    # instrumentation (visible in Streamlit UI if available)
    try:
        st.write("base_score values found (before):", before_examples)
        st.write("Was config changed?:", changed)
    except Exception:
        pass  # ignore if not running inside Streamlit

    if changed:
        booster.load_config(json.dumps(cfg))
        # verify after
        cfg_after = json.loads(booster.save_config())

        def _collect_after(node, acc):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "base_score":
                        acc.append(v)
                    else:
                        _collect_after(v, acc)
            elif isinstance(node, list):
                for v in node:
                    _collect_after(v, acc)

        after_examples = []
        _collect_after(cfg_after, after_examples)

        try:
            st.write("base_score values after:", after_examples)
        except Exception:
            pass

        # Ensure none remain bracketed
        if any(isinstance(v, str) and v.strip().startswith("[") for v in after_examples):
            raise RuntimeError("base_score still bracketed after normalization.")
    else:
        # Nothing was changed — likely the key path differs in your model version.
        # Raise to make the failure obvious *before* SHAP crashes.
        raise RuntimeError("No base_score entries found to normalize in booster config.")


def _predict_proba_from(booster_or_sklearn):
    """Return a callable f(X) -> prob of positive class."""
    try:
        # sklearn wrapper path (XGBClassifier etc.)
        if hasattr(booster_or_sklearn, "predict_proba"):
            return lambda X: booster_or_sklearn.predict_proba(X)[:, 1]
    except Exception:
        pass

    # Native Booster path
    import xgboost as xgb
    booster = booster_or_sklearn.get_booster() if hasattr(booster_or_sklearn, "get_booster") else booster_or_sklearn
    # booster.predict with output_margin=False returns probabilities for binary:logistic
    return lambda X: booster.predict(xgb.DMatrix(X), output_margin=False)


def get_xgb_explanation(xgb_model, X_background, X_test, shap_tab, top_n=15):
    """
    Global SHAP-style view using XGBoost native contributions (pred_contribs=True).
    Works in raw margin (Δ log-odds). No TreeExplainer, no base_score JSON parsing.
    """
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import plotly.express as px
    import streamlit as st

    # 0) Booster
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model

    # 1) Ensure numeric float32 (for DMatrix); keep original X_test for coloring if you prefer
    def _to_f32(df_like):
        if isinstance(df_like, pd.DataFrame):
            return df_like.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        return np.asarray(df_like, dtype=np.float32)

    X_test_f32 = _to_f32(X_test)

    # 2) Compute native SHAP contributions for ALL test rows
    dtest = xgb.DMatrix(X_test_f32)
    contribs = booster.predict(dtest, pred_contribs=True, approx_contribs=False)
    # contribs shape: (n_rows, n_features + 1); last column is the bias/base term
    vals_all = contribs[:, :-1]                  # (n_rows, n_features)
    feat_names = list(X_test.columns)
    shap_df = pd.DataFrame(vals_all, columns=feat_names, index=X_test.index)

    # 3) Tidy (long) format with feature values for coloring
    #    Note: these SHAP values are Δ log-odds (raw margin), not probabilities.
    long = (
        shap_df.stack().rename("shap").to_frame()
        .join(X_test.stack().rename("value"))
        .reset_index()
        .rename(columns={"level_0": "row", "level_1": "feature"})
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # 4) Rank features by mean(|SHAP|) and keep top_n
    influence = long.groupby("feature")["shap"].apply(lambda s: s.abs().mean()).sort_values(ascending=False)
    top_feats = influence.head(top_n).index.tolist()
    long_top = long[long["feature"].isin(top_feats)]

    # 5) Beeswarm-like strip (horizontal)
    fig_beeswarm = px.strip(
        long_top,
        x="shap", y="feature", color="value",
        orientation="h",
        hover_data={"row": True, "value": ":.3f", "shap": ":.4f"},
        title="Global Feature impact on churn",
    )
    # Order features by importance
    fig_beeswarm.update_layout(yaxis=dict(categoryorder="array", categoryarray=top_feats))
    fig_beeswarm.update_traces(jitter=0.35, marker={"opacity": 0.55, "size": 4})
    fig_beeswarm.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=list(top_feats)),
        yaxis_title="Feature",
        xaxis_title="Impact on churn probability",
        coloraxis_showscale=False
    )
    fig_beeswarm.update_layout(coloraxis=dict(cmin=0, cmax=1))

    # 6) Streamlit output
    with shap_tab:
        st.markdown("### Global Explanation")
        col1, col2 = st.columns([2, 1])
        col1.plotly_chart(fig_beeswarm, use_container_width=True)


def _patch_base_score_in_modelfile(booster: xgb.Booster) -> xgb.Booster:
    """
    Rewrites the model JSON so all 'base_score' entries are plain numeric strings (e.g., "0.5"),
    then loads and returns a *new* Booster built from the patched model.
    """
    # 1) Dump to a temp JSON file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()
    booster.save_model(tmp.name)

    # 2) Load JSON, normalize any 'base_score'
    with open(tmp.name, "r", encoding="utf-8") as f:
        model_json = json.load(f)

    def _normalize(val):
        if isinstance(val, str):
            s = val.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
            return str(float(s))  # may raise; that's fine
        if isinstance(val, (int, float)):
            return str(float(val))
        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], str):
            return str(float(val[0]))
        # fallback
        return "0.5"

    def _walk(node):
        changed = False
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k == "base_score":
                    try:
                        node[k] = _normalize(v)
                    except Exception:
                        node[k] = "0.5"
                    changed = True
                else:
                    if _walk(v): changed = True
        elif isinstance(node, list):
            for v in node:
                if _walk(v): changed = True
        return changed

    _walk(model_json)

    # 3) Write back and reload into a *new* Booster
    with open(tmp.name, "w", encoding="utf-8") as f:
        json.dump(model_json, f)

    patched = xgb.Booster()
    patched.load_model(tmp.name)

    # 4) Clean up
    try: os.remove(tmp.name)
    except Exception: pass

    return patched

def xgb_local_shap_by_truth(
    xgb_model,
    X_test_enc: pd.DataFrame,          # encoded/unscaled features used by XGB
    X_values: pd.DataFrame,            # human-readable values (same index)
    y_test: pd.Series, 
    threshold,                # 0/1, index aligned with X_test_enc
    shap_tab,
    background: pd.DataFrame | None = None,  # ~100–200 rows (TRAIN preferred)
    top_display: int = 12,
    title_prefix: str = "Local SHAP – XGBoost Explanation",
    value_format: dict | None = None,
):
    """
    Local explanation for XGBoost using native SHAP contributions (pred_contribs=True).
    - Works in raw margin (log-odds). We annotate base & sum(raw) and show bars per feature.
    - Avoids SHAP's model JSON parsing, so no '[5E-1]' base_score crash.

    Notes:
      * 'background' is unused here (TreeExplainer would need it; pred_contribs does not).
      * get_customers_for_explanation(y_test) should return labels present in X_test_enc.index
        (e.g., one with true=0 and one with true=1).
    """

    # 0) Get Booster
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model

    # 1) Choose which customers to explain
    sel_labels = get_customers_for_explanation(y_test)
    if isinstance(sel_labels, (np.ndarray, pd.Index)):
        sel_labels = sel_labels.tolist()
    if not sel_labels:
        with shap_tab:
            st.warning("No rows selected for explanation.")
        return

    # 2) Prepare numeric float32 matrices for the model; keep human-readable values separate
    def _to_f32(df_like):
        if isinstance(df_like, pd.DataFrame):
            return df_like.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        return np.asarray(df_like, dtype=np.float32)

    X_sel_enc = X_test_enc.loc[sel_labels]
    X_sel_enc_f32 = _to_f32(X_sel_enc)
    dmatrix_sel = xgb.DMatrix(X_sel_enc_f32)

    # 3) Native XGBoost SHAP contributions (Δ log-odds)
    #    Shape: (n_rows, n_features + 1). Last column is bias/base value.
    contribs = booster.predict(dmatrix_sel, pred_contribs=True, approx_contribs=False)
    vals_all = contribs[:, :-1]            # per-feature contributions
    base_all = contribs[:, -1]             # bias/base term per row
    feat_names = list(X_sel_enc.columns)

    # 4) Predicted probabilities for display
    try:
        # If using the sklearn wrapper, use predict_proba for convenience
        proba_sel = (xgb_model.predict_proba(X_test_enc.loc[sel_labels])[:, 1]
                     if hasattr(xgb_model, "predict_proba") else None)
    except Exception:
        proba_sel = None

    # Fallback: compute probs from raw margin (base + sum of contributions)
    if proba_sel is None:
        raw_sum = base_all + vals_all.sum(axis=1)
        proba_sel = expit(raw_sum)

    # 5) Render in Streamlit
    with shap_tab:
        st.markdown(f"### Local Explanation")
        for k, lbl in enumerate(sel_labels):
            # Per-row values
            vals = vals_all[k, :]                 # Δ log-odds per feature
            base = float(base_all[k])             # base (bias) in raw margin
            pred_p = float(proba_sel[k])
            true = int(y_test.loc[lbl])

            # Sort by |impact|
            s = pd.Series(vals, index=feat_names).sort_values(key=lambda x: x.abs(), ascending=False)
            top = s.head(top_display)
            other = s.iloc[top_display:].sum() if len(s) > top_display else 0.0

            # Table with REAL values for the top features (from X_values)
            feat_list = list(top.index)
            row_values = (X_values.loc[lbl].reindex(feat_list).fillna(0).replace({None: 0}))
            df_top = pd.DataFrame({"Feature": feat_list, "Value": row_values.values})
            if other != 0:
                df_top.loc[len(df_top)] = ["other", "—"]

            st.markdown(
                f"#### **Customer id `{lbl}`** — True={true} | Pred={pred_p:.3f} "
                f"→ **{'CHURN' if pred_p >= threshold else 'STAY'}**"
            )

            # Layout: table + bar chart
            local_col1, spacer, local_col2, spacer2 = st.columns([1, 0.5, 1.5, 0.5])

            fmt = {}
            if value_format:
                fmt.update({k: v for k, v in value_format.items() if k in df_top.columns})
            local_col1.dataframe(df_top.style.format(fmt), use_container_width=True)

            # Bar chart in Δ log-odds
            df_bar = pd.DataFrame({
                "feature": feat_list + (["other"] if other != 0 else []),
                "delta":   list(top.values) + ([other] if other != 0 else [])
            })

            fig = px.bar(
                df_bar,
                x="delta", y="feature", orientation="h",
                color="delta",
                color_continuous_scale=px.colors.diverging.RdBu_r,
                range_color=(df_bar["delta"].min(), df_bar["delta"].max()),
                title=f"{title_prefix} — Row {lbl} (Churn={true}, prediction={pred_p:.3f})"
            )
            fig.update_layout(
                xaxis_title="SHAP weight",
                yaxis_title="",
                coloraxis_colorbar_title="Weight",
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False
            )
            fig.add_vline(x=0, line_width=1, line_dash="dash", opacity=0.6)

            raw_sum = float(base + s.sum())
            fig.add_annotation(x=base, y=1.02 * max(1, len(df_bar)), text=f"base(raw)={base:.3f}",
                               showarrow=False, yref="y domain")
            fig.add_annotation(x=raw_sum, y=1.08 * max(1, len(df_bar)),
                               text=f"sum(raw)≈{raw_sum:.3f} (p≈{expit(raw_sum):.3f})",
                               showarrow=False, yref="y domain")

            local_col2.plotly_chart(fig, use_container_width=True)


def _normalize_to_en_dash(name: str) -> str:
    s = str(name)
    # turn literal escape text into the real en dash (works whether it’s "\u2013" or "\\u2013")
    s = s.replace("\\u2013", "\u2013")      # backslash-u-2013 -> en dash
    # normalize other common dash/minus variants to en dash too
    return (s
            .replace("\u2212", "\u2013")   # minus sign → en dash
            .replace("-", "\u2013")        # ASCII hyphen → en dash
            .replace("–", "\u2013"))       # already en dash, just ensure same codepoint


def align_to_training_columns(X, feat_train):
    import pandas as pd
    # ensure DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feat_train)

    # normalize dashes in BOTH sides, then map back to canonical feat_train
    feat_train_norm = [_normalize_to_en_dash(c) for c in feat_train]
    canon = { _normalize_to_en_dash(c): c for c in feat_train }

    # rename incoming columns to their canonical names when they match after normalization
    ren = {}
    for c in X.columns:
        c_norm = _normalize_to_en_dash(c)
        if c_norm in canon and canon[c_norm] != c:
            ren[c] = canon[c_norm]
    if ren:
        X = X.rename(columns=ren)

    # add missing training columns as zeros
    missing = [c for c in feat_train if c not in X.columns]
    for m in missing:
        X[m] = 0.0

    # drop extras not seen at training
    extras = [c for c in X.columns if c not in feat_train]
    if extras:
        X = X.drop(columns=extras)

    # reorder & cast
    return X.reindex(columns=feat_train).astype("float32")

def make_get_attribs_xgb_booster(booster, feature_names: list[str] | None = None):
    """
    Returns a callable get_attribs(X) -> np.ndarray of SHAP-like contributions
    (drops the bias column). If feature_names is provided, we hard-align every
    input to that exact training schema (handles missing one-hots and dash chars).
    """
    train_feat_names = feature_names

    # If caller didn't pass a list, try booster.feature_names (may be None)
    if train_feat_names is None:
        train_feat_names = getattr(booster, "feature_names", None)

    def get_attribs(X):
        import numpy as np, pandas as pd, xgboost as xgb
        if train_feat_names is not None:
            X_df = align_to_training_columns(X, train_feat_names)
            d = xgb.DMatrix(X_df, feature_names=train_feat_names)
        else:
            # Fallback: trust incoming columns (no alignment possible)
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            d = xgb.DMatrix(X_df.astype("float32"),
                            feature_names=list(X_df.columns))
        contribs = booster.predict(d, pred_contribs=True, approx_contribs=False)
        return np.asarray(contribs[:, :-1])  # drop bias term
    return get_attribs

def make_get_attribs_xgb_booster_with_schema(booster, feature_names: list[str]):
    import numpy as np, pandas as pd, xgboost as xgb

    def _normalize_to_en_dash(name: str) -> str:
        s = str(name)
        # turn literal escape text into the real en dash (works whether it’s "\u2013" or "\\u2013")
        s = s.replace("\\u2013", "\u2013")      # backslash-u-2013 -> en dash
        # normalize other common dash/minus variants to en dash too
        return (s
                .replace("\u2212", "\u2013")   # minus sign → en dash
                .replace("-", "\u2013")        # ASCII hyphen → en dash
                .replace("–", "\u2013"))       # already en dash, just ensure same codepoint


    def align_to_training_columns(X, feat_train):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feat_train)

        feat_train_norm = [_normalize_to_en_dash(c) for c in feat_train]
        canon = { _normalize_to_en_dash(c): c for c in feat_train }

        ren = {}
        for c in X.columns:
            c_norm = _normalize_to_en_dash(c)
            if c_norm in canon and canon[c_norm] != c:
                ren[c] = canon[c_norm]
        if ren:
            X = X.rename(columns=ren)

        missing = [c for c in feat_train if c not in X.columns]
        for m in missing:
            X[m] = 0.0

        extras = [c for c in X.columns if c not in feat_train]
        if extras:
            X = X.drop(columns=extras)

        return X.reindex(columns=feat_train).astype("float32")

    def get_attribs(X):
        X_df = align_to_training_columns(X, feature_names)
        d = xgb.DMatrix(X_df, feature_names=feature_names)
        contribs = booster.predict(d, pred_contribs=True, approx_contribs=False)
        return np.asarray(contribs[:, :-1])  # drop bias
    return get_attribs

