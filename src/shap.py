import shap
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
import json
import re


def get_lr_explanation(lr_model, background_data, X_test_scaled, shap_tab, top_n=15):

    expl_lr = shap.LinearExplainer(lr_model, background_data, model_output="probability")
    sv  = expl_lr(X_test_scaled)

    shap_vals = sv.values if hasattr(sv, "values") else sv
    base_vals = sv.base_values if hasattr(sv, "base_values") else np.full(X_test_scaled.shape[0], shap_vals.mean())

    # --- Tidy long format for Plotly ---
    shap_df = pd.DataFrame(shap_vals, columns=X_test_scaled.columns, index=X_test_scaled.index)
    feat_df = X_test_scaled.copy()

    long = (
        shap_df.stack().rename("shap")
        .to_frame()
        .join(feat_df.stack().rename("value"))
        .reset_index()
        .rename(columns={"level_0": "row", "level_1": "feature"})
    )

    # --- rank features by mean(|SHAP|) ---
    feat_order = (
        long.groupby("feature")["shap"]
            .apply(lambda s: s.abs().mean())
            .sort_values(ascending=True)
            .index.tolist()
    )

    # --- 1) Beeswarm-like strip (direction + value color) ---
    # Right = raises churn prob; Left = lowers churn prob
    fig_beeswarm = px.strip(
        long, x="shap", y="feature", color="value",
        orientation="h",
        hover_data={"row": True, "value": ":.3f", "shap": ":.4f"},
        title="Feature Impact on Churn"
    )
    fig_beeswarm.update_layout(yaxis=dict(categoryorder="array", categoryarray=feat_order))
    fig_beeswarm.update_traces(jitter=0.35, marker={"opacity": 0.55, "size": 4})
    fig_beeswarm.update_layout(
        yaxis_title="Feature", 
        xaxis_title="Impact on churn probability",
        coloraxis_showscale=False,
        showlegend=False)

    with shap_tab:
        column1, column2 = st.columns([2, 1])

        column1.plotly_chart(fig_beeswarm)




    # # --- 2) Global strength: mean |SHAP| ---
    # importance = (
    #     long.groupby("feature")["shap"]
    #     .apply(lambda s: s.abs().mean())
    #     .sort_values(ascending=False)
    #     .head(top_n)
    #     .reset_index(name="mean_abs_shap")
    # )
    # fig_importance = px.bar(
    #     importance.sort_values("mean_abs_shap"),
    #     x="mean_abs_shap", y="feature", orientation="h",
    #     title=f"Average magnitude of impact (Top {top_n})"
    # )
    # fig_importance.update_layout(xaxis_title="Mean |SHAP|", yaxis_title="Feature")

    # # --- 3) Direction signal per feature (corr between value and SHAP) ---
    # def safe_corr(g):
    #     if g["value"].std(ddof=0) == 0 or g["shap"].std(ddof=0) == 0:
    #         return np.nan
    #     return np.corrcoef(g["value"], g["shap"])[0, 1]

    # direction = (
    #     long.groupby("feature").apply(safe_corr)
    #     .reset_index(name="direction_corr")
    # )
    # dir_top = direction.merge(importance[["feature"]], on="feature", how="right")
    # fig_direction = px.bar(
    #     dir_top.sort_values("direction_corr"),
    #     x="direction_corr", y="feature", orientation="h",
    #     title="Direction signal (corr(value, SHAP))"
    # )
    # fig_direction.update_layout(
    #     xaxis_title="Correlation ( >0 ⇒ higher values increase churn )",
    #     yaxis_title="Feature"
    # )

    # # Optional: a local WATERFALL for a single row (uses graph_objects Waterfall)
    # # Pick a row—e.g., the first predicted churner
    # preds = (base_vals + shap_vals.sum(axis=1) > 0.5).astype(int)
    # idx = int(np.argmax(preds))  # choose any row you like

    # contrib = pd.Series(shap_vals[idx, :], index=X_test_scaled.columns)
    # # Show top contributors and collapse the tail for readability
    # top = contrib.abs().sort_values(ascending=False).head(12).index
    # tail_sum = contrib.drop(top).sum()
    # wf_parts = contrib[top].append(pd.Series({"(others)": tail_sum}))

    # measure = ["relative"] * len(wf_parts)
    # x = wf_parts.index.tolist()
    # y = wf_parts.values.tolist()

    # pred_prob = float(base_vals[idx] + shap_vals[idx].sum())
    # fig_waterfall = go.Figure(go.Waterfall(
    #     name="Contributions",
    #     orientation="v",
    #     measure=measure,
    #     x=x,
    #     text=[f"{v:+.3f}" for v in y],
    #     y=y
    # ))
    # fig_waterfall.update_layout(
    #     title=f"Local explanation for row {idx} (prob≈{pred_prob:.3f})",
    #     yaxis_title="Δ probability from base",
    #     xaxis_title="Feature"
    # )


    # return {
    #     "fig_beeswarm": fig_beeswarm,
    #     "fig_importance": fig_importance,
    #     "fig_direction": fig_direction,
    #     "fig_waterfall": fig_waterfall,
    # }

def get_rf_explanation(rf_model, X_background, X_test, shap_tab, top_n=15):
    """
    RF SHAP beeswarm-like plot.
    Note: pass *unscaled* features for tree models (X_background, X_test).
    """
    # --- SHAP for Random Forest (probability for class=1) ---
    expl_rf = shap.TreeExplainer(rf_model, data=X_background, model_output="probability")
    sv = expl_rf(X_test)  # returns shap.Explanation on recent SHAP

    shap_vals = sv.values if hasattr(sv, "values") else sv
    base_vals = sv.base_values if hasattr(sv, "base_values") else None

    # If multi-output (e.g., shape = [n, n_feat, 2]), take positive class (index 1)
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 2:
        shap_vals = shap_vals[:, :, 1]
        if base_vals is not None and base_vals.ndim == 2 and base_vals.shape[1] == 2:
            base_vals = base_vals[:, 1]

    # --- Tidy long format for Plotly ---
    shap_df = pd.DataFrame(shap_vals, columns=X_test.columns, index=X_test.index)
    feat_df = X_test.copy()

    long = (
        shap_df.stack().rename("shap").to_frame()
        .join(feat_df.stack().rename("value"))
        .reset_index()
        .rename(columns={"level_0": "row", "level_1": "feature"})
    )

    # ensure numeric so we get a colorbar (not a legend)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # --- rank features by mean(|SHAP|) and keep top_n ---
    influence = (long.groupby("feature")["shap"].apply(lambda s: s.abs().mean())
                 .sort_values(ascending=False))
    top_feats = influence.head(top_n).index.tolist()
    long_top = long[long["feature"].isin(top_feats)]

    # order features: most influential at top
    feat_order = top_feats  # already sorted descending

    # --- Beeswarm-like strip ---
    fig_beeswarm = px.strip(
        long_top, x="shap", y="feature", color="value",
        orientation="h",
        hover_data={"row": True, "value": ":.3f", "shap": ":.4f"},
        title="Feature impact on churn"
    )
    
    fig_beeswarm.update_layout(yaxis=dict(categoryorder="array", categoryarray=feat_order))
    fig_beeswarm.update_traces(jitter=0.35, marker={"opacity": 0.55, "size": 4})
    fig_beeswarm.update_layout(
        yaxis_title="Feature",
        xaxis_title="Impact on churn probability",
        coloraxis_showscale=False,
        showlegend=False
    )
    with shap_tab:
        column1, column2 = st.columns([2, 1])

        column1.plotly_chart(fig_beeswarm)

def _fix_base_score_json_inplace(booster) -> None:
    """
    Rewrites the booster config so `"base_score"` is a plain numeric string (e.g., "0.5").
    Handles weird forms like "[5E-1]" or "[0.5]".
    """
    try:
        cfg = booster.save_config()
        # Replace "base_score": "[...]" (any single token inside brackets) -> numeric string without brackets
        def _repl(m):
            num = m.group(1)
            try:
                return f'"base_score": "{float(num)}"'
            except Exception:
                return '"base_score": "0.5"'

        fixed = re.sub(
            r'"base_score"\s*:\s*"\[\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*\]"',
            _repl,
            cfg,
            flags=re.IGNORECASE,
        )

        if fixed != cfg:
            booster.load_config(fixed)
            return

        # If still bracketed (other odd formats), force to "0.5"
        if re.search(r'"base_score"\s*:\s*"\[', booster.save_config(), flags=re.IGNORECASE):
            forced = re.sub(
                r'"base_score"\s*:\s*".*?"',
                '"base_score": "0.5"',
                cfg,
                flags=re.IGNORECASE,
            )
            booster.load_config(forced)
    except Exception:
        pass  # non-fatal; we’ll try a fallback if TreeExplainer fails


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
    # Get booster (works for sklearn wrappers or booster directly)
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model

    # Try to sanitize base_score inside the JSON config
    try:
        booster.set_param({"base_score": 0.5})  # harmless hint
    except Exception:
        pass
    _fix_base_score_json_inplace(booster)

    # --- Try fast TreeExplainer first ---
    sv = None
    try:
        expl_xgb = shap.TreeExplainer(booster, data=X_background, model_output="probability")
        sv = expl_xgb(X_test)
    except Exception as e_tree:
        # --- Robust fallback: model-agnostic explainer (works with any model) ---
        f = _predict_proba_from(xgb_model)
        # Keep background small for speed (e.g., 100 rows)
        bg = X_background
        if len(bg) > 100:
            bg = bg.sample(100, random_state=42)

        try:
            # Prefer Permutation or Kernel depending on SHAP version
            try:
                expl_generic = shap.Explainer(f, bg, algorithm="permutation")
            except Exception:
                expl_generic = shap.KernelExplainer(f, bg)
            sv = expl_generic(X_test)
        except Exception as e_fallback:
            raise RuntimeError(
                f"TreeExplainer failed with: {e_tree}\n"
                f"Generic explainer also failed with: {e_fallback}"
            )

    # --- Extract SHAP values (handle different SHAP return shapes) ---
    shap_vals = sv.values if hasattr(sv, "values") else sv
    base_vals = getattr(sv, "base_values", None)

    # If multi-output (n, n_feat, 2), take positive class
    if hasattr(shap_vals, "ndim") and shap_vals.ndim == 3 and shap_vals.shape[-1] == 2:
        shap_vals = shap_vals[:, :, 1]
        if base_vals is not None and getattr(base_vals, "ndim", 1) == 2 and base_vals.shape[1] == 2:
            base_vals = base_vals[:, 1]

    # --- Long/tidy DF for Plotly ---
    shap_df = pd.DataFrame(shap_vals, columns=X_test.columns, index=X_test.index)
    long = (
        shap_df.stack().rename("shap").to_frame()
        .join(X_test.stack().rename("value"))
        .reset_index()
        .rename(columns={"level_0": "row", "level_1": "feature"})
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # Rank features by mean(|SHAP|)
    influence = long.groupby("feature")["shap"].apply(lambda s: s.abs().mean()).sort_values(ascending=False)
    top_feats = influence.head(top_n).index.tolist()
    long_top = long[long["feature"].isin(top_feats)]

    # --- Beeswarm-like strip plot ---
    fig_beeswarm = px.strip(
        long_top, x="shap", y="feature", color="value",
        orientation="h",
        hover_data={"row": True, "value": ":.3f", "shap": ":.4f"},
        title="Feature impact on churn"
    )
    fig_beeswarm.update_layout(yaxis=dict(categoryorder="array", categoryarray=top_feats))
    fig_beeswarm.update_traces(jitter=0.35, marker={"opacity": 0.55, "size": 4})
    fig_beeswarm.update_layout(
        yaxis_title="Feature",
        xaxis_title="Impact on churn probability",
        coloraxis_showscale=False,
        showlegend=False
    )

    # --- Streamlit output ---
    with shap_tab:
        col1, col2 = st.columns([2, 1])
        col1.plotly_chart(fig_beeswarm, use_container_width=True)