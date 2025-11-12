from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import streamlit as st
from src.customer import get_customers_for_explanation
import plotly.express as px
import xgboost as xgb
from scipy.special import expit
from numpy.testing import assert_equal
from scipy.stats import spearmanr

def lime_local_rank_vs_impact(
    model,
    explainer,
    X_row: pd.Series,
    X_background: pd.DataFrame,
    num_features: int = 10,
    mask_strategy: str = "mean"
) -> dict:
    # LIME explanation
    exp = explainer.explain_instance(
        data_row=X_row.values,
        predict_fn=model.predict_proba,
        num_features=num_features
    )
    class_idx = 1
    pairs = exp.local_exp[class_idx]       # (feature_idx, weight)
    pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
    ranked_idx = [i for (i, w) in pairs_sorted]
    ranked_feats = [explainer.feature_names[i] for i in ranked_idx]
    weights = np.array([w for (i, w) in pairs_sorted])

    # baseline prob
    p0 = float(model.predict_proba(X_row.to_frame().T)[:, 1])

    # per-feature single deletion impact
    col_means = X_background.mean(numeric_only=True)
    impacts = []
    for feat in ranked_feats:
        x_ = X_row.copy()
        if mask_strategy == "mean":
            x_[feat] = col_means.get(feat, 0.0)
        else:
            x_[feat] = X_background[feat].sample(1, random_state=0).iloc[0]
        p = float(model.predict_proba(x_.to_frame().T)[:, 1])
        impacts.append(p0 - p)  # positive = removal lowers prob

    impacts = np.array(impacts)
    # Spearman between |weights| rank (1..k) and |impact|
    ranks = np.arange(1, len(ranked_feats) + 1)
    rho, pval = spearmanr(ranks, np.abs(impacts))

    return {
        "features": ranked_feats,
        "weights": weights,
        "impacts": impacts,
        "rho": float(rho),
        "pval": float(pval),
        "p0": p0
    }


def lime_local_deletion_test(
    model,
    explainer,                 # LimeTabularExplainer already built on TRAIN data
    X_row: pd.Series,          # one row (features exactly as model expects)
    X_background: pd.DataFrame,# same preprocessing as model input (for means)
    num_features: int = 10,    # how many LIME features to use
    k_max: int | None = None,  # how far along the curve to delete
    mask_strategy: str = "mean"  # "mean" or "permute"
) -> dict:
    """
    Returns:
      {
        'order': list[str],               # features ranked by |LIME weight|
        'p0': float,                      # original prob
        'curve': pd.DataFrame(k, prob, drop, frac_deleted),
        'auc': float,                     # AUC of prob vs frac_deleted (lower is better)
        'flip_k': int | None              # first k where predicted label flips (if any)
      }
    """
    # 1) LIME explanation for this row
    # NOTE: model must have predict_proba
    exp = explainer.explain_instance(
        data_row=X_row.values,
        predict_fn=model.predict_proba,
        num_features=num_features
    )
    # exp.as_list(): list[(feature_rule_str, weight)]
    # We need raw feature names; LIME gives humanized rules. Use explainer.feature_names
    # and exp.local_exp[1] (class=1) to get (feature_index, weight).
    # Fallback to class 1; if your positive class index differs, adjust accordingly.
    class_idx = 1
    pairs = exp.local_exp[class_idx]  # list of (feature_idx, weight)
    pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
    ranked_idx = [i for (i, w) in pairs_sorted]
    ranked_feats = [explainer.feature_names[i] for i in ranked_idx]

    # 2) baseline prob
    p0 = float(model.predict_proba(X_row.to_frame().T)[:, 1])

    # 3) iterative deletion
    if k_max is None:
        k_max = len(ranked_feats)

    probs = []
    drops = []
    fracs = []
    flip_k = None
    y0 = int(p0 >= 0.5)

    # helpers
    X_work = X_row.copy()
    col_means = X_background.mean(numeric_only=True)

    def mask_column(series: pd.Series, col: str):
        if mask_strategy == "mean":
            val = col_means.get(col, 0.0)
            series[col] = val
        elif mask_strategy == "permute":
            # single-row permutation makes no sense; draw a random background value
            series[col] = X_background[col].sample(1, random_state=0).iloc[0]
        else:
            raise ValueError("mask_strategy must be 'mean' or 'permute'.")

    current = X_work.copy()
    for k in range(0, k_max + 1):
        # evaluate current row
        pk = float(model.predict_proba(current.to_frame().T)[:, 1])
        probs.append(pk)
        drops.append(p0 - pk)
        fracs.append(k / max(1, k_max))

        # flip check (first k where predicted label changes)
        if flip_k is None and int(pk >= 0.5) != y0:
            flip_k = k

        # prepare next step by masking the next feature
        if k < k_max:
            mask_column(current, ranked_feats[k])

    # 4) AUC of prob curve (trapezoid over frac_deleted)
    auc = float(np.trapz(y=probs, x=fracs))

    curve = pd.DataFrame({"k": range(0, k_max + 1),
                          "frac_deleted": fracs,
                          "prob": probs,
                          "drop": drops})

    return {
        "order": ranked_feats,
        "p0": p0,
        "curve": curve,
        "auc": auc,
        "flip_k": flip_k
    }



def get_lime_explanations_binary(
    model,                      # sklearn estimator with predict_proba OR xgboost.Booster
    X_values: pd.DataFrame,     # human-readable values to display (same index as X_test_enc)
    X_train_enc: pd.DataFrame,  # encoded features used to train the model (UNscaled for trees is fine)
    X_test_enc: pd.DataFrame,   # encoded features for test (same columns as train)
    y_test: pd.Series,
    threshold: float,
    lime_tab,
    *,
    top_display: int = 12,
    title_prefix: str = "Local LIME – Binary Classifier",
    value_format: dict | None = None,
    num_features: int = 50,
    random_state: int = 42,
):
    # --- Align indices and columns ---
    if not y_test.index.equals(X_test_enc.index):
        y_test = y_test.reindex(X_test_enc.index)
    if not X_values.index.equals(X_test_enc.index):
        X_values = X_values.reindex(X_test_enc.index)
    # ensure test has same columns/order as train
    if not X_test_enc.columns.equals(X_train_enc.columns):
        X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    feat_names = X_train_enc.columns.tolist()

    # --- Build predict_fn that returns (n, 2) probabilities ---
    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
        # class names / positive class index from sklearn
        class_names = [str(c) for c in getattr(model, "classes_", [0, 1])]
        try:
            pos_class_idx = int(np.where(np.asarray(model.classes_) == 1)[0][0])
        except Exception:
            pos_class_idx = 1  # fallback
    elif isinstance(model, xgb.Booster):
        feat_names = X_train_enc.columns.tolist()
        # Booster outputs P(positive) with objective=binary:logistic
        def predict_fn(X_batch: np.ndarray) -> np.ndarray:
            dm = xgb.DMatrix(X_batch, feature_names=feat_names)   
            p1 = model.predict(dm).reshape(-1)
            return np.column_stack([1.0 - p1, p1])
        class_names = ["0", "1"]
        pos_class_idx = 1
    else:
        raise TypeError("model must be a sklearn classifier with predict_proba or an xgboost.Booster")

    # --- LIME explainer on encoded TRAIN data (this is LIME's 'background') ---
    explainer = LimeTabularExplainer(
        training_data=X_train_enc.to_numpy(),
        feature_names=feat_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        sample_around_instance=True,
        random_state=random_state,
    )

    # --- Choose one true 0 and one true 1 ---
    idx0 = y_test.index[y_test == 0]
    idx1 = y_test.index[y_test == 1]
    if len(idx0) == 0 or len(idx1) == 0:
        with lime_tab:
            st.error("Need at least one 0 and one 1 in y_test to show two LIME examples.")
        return
    row_labels = [idx0[0], idx1[0]]

    with lime_tab:
        st.markdown("### Local Explanation")
        for lbl in row_labels:
            x_i = X_test_enc.loc[lbl].to_numpy()

            # Explicitly ask LIME to explain the positive class
            exp = explainer.explain_instance(
                data_row=x_i,
                predict_fn=predict_fn,
                num_features=max(top_display, num_features),
                labels=[pos_class_idx],
            )

            # (feature_idx, weight) → Series keyed by feature name
            pairs = dict(exp.as_map()[pos_class_idx])
            w = pd.Series({feat_names[j]: pairs[j] for j in pairs})
            s = w.sort_values(key=lambda x: x.abs(), ascending=False)
            top = s.head(top_display)
            other = s.iloc[top_display:].sum() if len(s) > top_display else 0.0

            # Feature | Value table pulled from X_values (human-readable)
            feat_list = list(top.index)
            values_row = X_values.loc[lbl].reindex(feat_list).fillna(0).replace({None: 0})
            df_top = pd.DataFrame({"Feature": feat_list, "Value": values_row.values})
            if other != 0:
                df_top.loc[len(df_top)] = ["other", "—"]

            # Header with model probability and your threshold decision
            p = float(predict_fn(np.asarray([x_i]))[0, pos_class_idx])
            true = int(y_test.loc[lbl])
            st.markdown(
                f"#### **Customer id `{lbl}`** — Churn={true} | prediction={p:.3f} "
                f"→ **{'CHURN' if p >= threshold else 'STAY'}**"
            )

            local_col1, spacer, local_col2, spacer2 = st.columns([1, 0.5, 1.5, 0.5])

            fmt = {}
            if value_format:
                fmt.update({k: v for k, v in value_format.items() if k in df_top.columns})
            local_col1.dataframe(df_top.style.format(fmt), use_container_width=True)
          
            df_bar = pd.DataFrame({
                "feature": feat_list + (["other"] if other != 0 else []),
                "weight":  list(top.values) + ([other] if other != 0 else [])
            })
            fig = px.bar(
                df_bar, x="weight", y="feature", orientation="h",
                color="weight", color_continuous_scale=px.colors.diverging.RdBu_r,
                range_color=(df_bar["weight"].min(), df_bar["weight"].max()),
                title=f"{title_prefix} — Row {lbl} (Churn={true}, prediction={p:.3f})",
            )
            fig.update_layout(
                xaxis_title="LIME weight", yaxis_title="",
                coloraxis_colorbar_title="Weight", margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False
            )
            fig.add_vline(x=0, line_width=1, line_dash="dash", opacity=0.6)
            local_col2.plotly_chart(fig, use_container_width=True)

def lime_xgb_local_rank_vs_impact(
    explainer,
    X_row: pd.Series,
    X_background: pd.DataFrame,
    predict_fn,                  # <- callable(X[n,d]) -> (n,2) proba
    num_features: int = 10,
    mask_strategy: str = "mean"
) -> dict:
    exp = explainer.explain_instance(
        data_row=X_row.values,
        predict_fn=predict_fn,
        num_features=num_features
    )
    class_idx = 1
    pairs = exp.local_exp[class_idx]   # list of (feature_idx, weight)
    pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
    ranked_idx = [i for (i, w) in pairs_sorted]
    ranked_feats = [explainer.feature_names[i] for i in ranked_idx]
    weights = np.array([w for (i, w) in pairs_sorted])

    p0 = float(predict_fn(X_row.to_frame().T.values)[0, class_idx])

    col_means = X_background.mean(numeric_only=True)
    impacts = []
    for feat in ranked_feats:
        x_ = X_row.copy()
        if mask_strategy == "mean":
            x_[feat] = col_means.get(feat, 0.0)
        else:  # "permute" -> sample a background value
            x_[feat] = X_background[feat].sample(1, random_state=0).iloc[0]
        p = float(predict_fn(x_.to_frame().T.values)[0, class_idx])
        impacts.append(p0 - p)

    ranks = np.arange(1, len(ranked_feats) + 1)
    rho, pval = spearmanr(ranks, np.abs(impacts))

    return {
        "features": ranked_feats,
        "weights": weights,
        "impacts": np.array(impacts),
        "rho": float(rho),
        "pval": float(pval),
        "p0": p0
    }


def lime_xgb_local_deletion_test(
    explainer,
    X_row: pd.Series,
    X_background: pd.DataFrame,
    predict_fn,                 # <- callable(X[n,d]) -> (n,2) proba
    num_features: int = 10,
    k_max: int | None = None,
    mask_strategy: str = "mean"
) -> dict:
    exp = explainer.explain_instance(
        data_row=X_row.values,
        predict_fn=predict_fn,
        num_features=num_features
    )
    class_idx = 1
    pairs = exp.local_exp[class_idx]
    pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
    ranked_idx = [i for (i, w) in pairs_sorted]
    ranked_feats = [explainer.feature_names[i] for i in ranked_idx]

    p0 = float(predict_fn(X_row.to_frame().T.values)[0, class_idx])

    if k_max is None:
        k_max = len(ranked_feats)

    probs, drops, fracs = [], [], []
    flip_k = None
    y0 = int(p0 >= 0.5)

    current = X_row.copy()
    col_means = X_background.mean(numeric_only=True)

    def mask_column(series: pd.Series, col: str):
        if mask_strategy == "mean":
            series[col] = col_means.get(col, 0.0)
        elif mask_strategy == "permute":
            series[col] = X_background[col].sample(1, random_state=0).iloc[0]
        else:
            raise ValueError("mask_strategy must be 'mean' or 'permute'.")

    for k in range(0, k_max + 1):
        pk = float(predict_fn(current.to_frame().T.values)[0, class_idx])
        probs.append(pk)
        drops.append(p0 - pk)
        fracs.append(k / max(1, k_max))

        if flip_k is None and int(pk >= 0.5) != y0:
            flip_k = k

        if k < k_max:
            mask_column(current, ranked_feats[k])

    auc = float(np.trapz(y=probs, x=fracs))
    curve = pd.DataFrame({"k": range(0, k_max + 1),
                          "frac_deleted": fracs,
                          "prob": probs,
                          "drop": drops})

    return {"order": ranked_feats, "p0": p0, "curve": curve, "auc": auc, "flip_k": flip_k}
