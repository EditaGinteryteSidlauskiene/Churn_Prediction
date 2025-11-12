import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, recall_score
from sklearn.ensemble import RandomForestClassifier
import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from src.config import data_path
from src.features import convert_str_to_numeric, fill_na_values_with_mean, create_churn_flag
import seaborn as sns
import matplotlib.pyplot as plt
from src.telco_data_analysis import get_correlation_heatmap_with_engineered_features, show_telco_correlations_and_interactions,show_telco_demographic_drivers, show_telco_engagement_and_churn, show_telco_categorical_drivers, get_telco_data_analysis, get_text_about_engineered_features, get_telco_pie_caption, show_telco_numeric_drivers
from src.internet_data_analysis import show_internet_correlations_and_interactions, engineer_internet_features, show_internet_engagement_and_churn, show_internet_categorical_drivers, show_internet_numeric_drivers, get_internet_pie_caption, get_internet_data_analysis, get_internet_data_null_heatmap, get_internet_dataset_null_percentage, get_internet_data_missing_values_caption, get_internet_data_null_hist
from src.exploratory_data_analysis import show_balance_between_yes_and_no
from src.telco_data_preprocessing import split_telco_data, get_scaled_telco_features, map_telco_data_features, engineer_telco_data_features, encode_telco_data
from src.internet_data_preprocessing import split_internet_data, get_scaled_internet_features, encode_internet_data
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from src.logistic_regression import get_internet_lr_metrics_caption, get_telco_lr_metrics_caption, display_lr_metrics, perform_primary_lg_training, cross_validate_lg_model, hyperparameter_tune_lg, retrain_lg_model
from src.random_forest import get_internet_rf_metrics_caption, get_telco_rf_metrics_caption, display_rf_metrics, perform_primary_rf_training, cross_validate_rf_model, hyperparameter_tune_rf, retrain_rf_model
from src.xgboost import get_internet_xgb_metrics_caption, get_telco_xgb_metrics_caption, display_xgb_metrics, perform_primary_xgb_training, cross_validate_xgb_model, xgb_random_search, xgb_grid_refine, retrain_xgb_model
from sklearn.model_selection import StratifiedKFold
from src.shap import (
    get_lr_explanation, get_rf_explanation, get_xgb_explanation,
    lr_local_shap_by_truth, rf_local_shap_by_truth, xgb_local_shap_by_truth,
    collect_attributions_for_tests, faithfulness_report, stability_report, sanity_report, local_faithfulness_report, _patch_base_score_in_modelfile, 
    make_get_attribs_xgb_booster, make_get_attribs_xgb_booster_with_schema
)
from src.lime import get_lime_explanations_binary, lime_local_deletion_test, lime_local_rank_vs_impact, lime_xgb_local_deletion_test, lime_xgb_local_rank_vs_impact
from src.dice import get_counterfactual_analysis
from src.helper import wrap_for_threshold
from src.threshold_model import ThresholdedModel
from src.XGBooster_adapter import XGBoosterAdapter
from sklearn.metrics import roc_auc_score
from src.customer import get_customers_for_explanation
from io import StringIO
from src.stability_tests import local_stability_report_generic
from typing import Dict, Iterable, Tuple, Callable
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
import hashlib

def _spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ between |a| and |b| (robust to sign)."""
    a_abs, b_abs = np.abs(a).ravel(), np.abs(b).ravel()
    if np.all(a_abs == a_abs[0]) or np.all(b_abs == b_abs[0]):
        return 0.0
    return float(spearmanr(a_abs, b_abs).correlation)

def _get_row_attr(get_attr_batch: callable, X: pd.DataFrame, row_id) -> np.ndarray:
    """Return attribution vector for a single row_id using a batch getter."""
    A = get_attr_batch(X.loc[[row_id]])
    return np.asarray(A).ravel()

def _feature_shuffle_once(get_attr_batch: callable, X: pd.DataFrame, row_id, feat: str, rng: np.random.RandomState) -> float:
    """Shuffle one feature column in X (incl. the picked row), recompute attributions for that row."""
    X_shuf = X.copy()
    X_shuf[feat] = X_shuf[feat].sample(frac=1.0, random_state=int(rng.randint(0, 1e9))).values
    return _get_row_attr(get_attr_batch, X_shuf, row_id)


def _ensure_series(y, index_like):
    """Return y as a pandas Series aligned to index_like."""
    import pandas as pd
    if isinstance(y, pd.Series):
        return y.reindex(index_like)
    if hasattr(y, "values"):  # DataFrame with a single col "Churn" etc.
        if y.shape[1] == 1:
            return y.iloc[:, 0].reindex(index_like)
        if "Churn" in y.columns:
            return y["Churn"].reindex(index_like)
        raise ValueError("y_val DataFrame must have a single column or 'Churn'.")
    # numpy array
    return pd.Series(y, index=index_like)

def make_stratified_batch(X, y, include_index=None, n_per_class=80, seed=42):
    """
    Return (X_batch, y_batch) with both classes present.
    Ensures include_index is in the batch. Works even if classes imbalanced.
    """
    import numpy as np, pandas as pd
    rng = np.random.RandomState(seed)
    y = _ensure_series(y, X.index)

    # Map Yes/No -> 1/0 if needed
    if y.dtype == object:
        y_num = y.map({"Yes": 1, "No": 0})
    else:
        y_num = y.astype(int)

    classes = np.unique(y_num.dropna().values)
    if len(classes) < 2:
        # fallback: use the full test set; the caller can decide to skip label-rand
        return X, y

    idx = []
    for c in classes:
        cls_idx = y_num[y_num == c].index
        take = min(n_per_class, len(cls_idx))
        if take == 0:
            continue
        picked = rng.choice(cls_idx, size=take, replace=False)
        idx.extend(picked)

    if include_index is not None:
        # make sure the chosen instance is present
        if include_index not in idx:
            idx.append(include_index)

    idx = pd.Index(idx).unique()
    return X.loc[idx], y.loc[idx]


st.set_page_config(page_title='Churn Prediction Dashboard', layout="wide")
st.title('Churn Prediction Dashboard')

st.sidebar.subheader('Visualization Settings')

# Read datasets
telco_data = pd.read_csv(data_path("telco_data.csv"))
internet_data = pd.read_csv(data_path("internet_service_churn.csv"))
internet_data = internet_data.rename(columns={'reamining_contract': 'remaining_contract'})

# Map dataset names to DataFrames
datasets = {
    "Telco Dataset": telco_data,
    "Internet Dataset": internet_data
}

selected_dataset = st.sidebar.selectbox("Choose a dataset:", list(datasets.keys()))

# Display info of the selected dataset
if selected_dataset in datasets:
    # Add subtitles and expander
    st.markdown('#### Data Exploratory Analysis')
    st.markdown(f'##### {selected_dataset}')
    expander = st.expander('About this dataset')
    
    # Display data explanation
    if selected_dataset == 'Telco Dataset':
        expander.write(get_telco_data_analysis())
    elif selected_dataset == 'Internet Dataset':
        expander.write(get_internet_data_analysis())

    # Create four columns and get metrics
    data_info_col1, data_info_col2, data_info_col3, data_info_col4 = st.columns([1, 1, 1, 1])
    n_rows, n_cols = datasets[selected_dataset].shape
    total_missing = int(datasets[selected_dataset].isna().sum().sum())
    memory_mb = datasets[selected_dataset].memory_usage(deep=True).sum() / 1024**2

    # Display metrics
    data_info_col1.metric('Rows', f'{n_rows:,}')
    data_info_col2.metric('Columns', f'{n_cols:,}')
    data_info_col3.metric("Total Missing", f"{total_missing:,}")
    data_info_col4.metric("Memory", f"{memory_mb:.1f} MB")

    # Display first five rows of the dataset
    st.markdown('###### First five rows of the dataset')
    st.write(datasets[selected_dataset].head())
        
    if selected_dataset == 'Telco Dataset':
        selected_anaysis_type = st.sidebar.selectbox(
        options=[
            'Choose Analysis Type', 
            'Numeric Drivers of Churn', 
            'Categorical Drivers of Churn',
            'Service Engagement and Churn',
            'Demographic Drivers of Churn',
            'Correlations and Interactions'],
        label='Choose Analysis Type',
        label_visibility='collapsed',
        index=0
        )

        selected_prediction_model = st.sidebar.selectbox(
        options=[
            'Choose Prediction Model', 
            'Logistic Regression', 
            'Random Forest',
            'XGBoost'],
        label='Choose Prediction Model',
        label_visibility='collapsed',
        index=0
        )
        
        #---------------   Cleaning telco data ------------------------
        telco_data['TotalCharges'] = convert_str_to_numeric(telco_data['TotalCharges'])
        telco_data['TotalCharges'] = fill_na_values_with_mean(telco_data['TotalCharges'])
        telco_data['ChurnFlag'] = create_churn_flag(telco_data['Churn'])
        
        pie_col1, pie_col2 = st.columns([1, 1])
        pie_col1.plotly_chart(show_balance_between_yes_and_no(telco_data))
        pie_col2.caption(body=get_telco_pie_caption(), unsafe_allow_html=True)

        if selected_anaysis_type == 'Numeric Drivers of Churn':
            st.subheader(selected_anaysis_type)
            show_telco_numeric_drivers(telco_data)

        elif selected_anaysis_type == 'Categorical Drivers of Churn':
            st.subheader(selected_anaysis_type)
            show_telco_categorical_drivers(telco_data)

        elif selected_anaysis_type == 'Service Engagement and Churn':
            st.subheader(selected_anaysis_type)
            show_telco_engagement_and_churn(telco_data)

        elif selected_anaysis_type == 'Demographic Drivers of Churn':
            st.subheader(selected_anaysis_type)
            show_telco_demographic_drivers(telco_data)

        elif selected_anaysis_type == 'Correlations and Interactions':
            st.subheader(selected_anaysis_type)
            show_telco_correlations_and_interactions(telco_data)

        #----------------- Preprocessing and engineering telco data ------------------------
        map_telco_data_features(telco_data)
        telco_data_engineered = engineer_telco_data_features(telco_data)
        X_train, X_test, y_train, y_test = split_telco_data(telco_data_engineered)

        X_train_encoded = encode_telco_data(telco_data, X_train)
        X_test_encoded = encode_telco_data(telco_data, X_test)
        
        if selected_anaysis_type == 'Correlations and Interactions':
            get_text_about_engineered_features()
            telco_data_encoded = encode_telco_data(telco_data, telco_data_engineered)
            get_correlation_heatmap_with_engineered_features(telco_data_encoded)

        scaler, scaled_X_train_features, scaled_X_test_features, num_cols_in_scaler_order = get_scaled_telco_features(X_train_encoded, X_test_encoded)
        background_data_scaled = (
            scaled_X_train_features
                .assign(Churn=y_train.values)
                .groupby('Churn', group_keys=False)
                .apply(lambda x: x.sample(frac=500/len(scaled_X_train_features), random_state=42))
                .drop(columns='Churn')
            )
        background_data_encoded = (
            X_train_encoded
                .assign(Churn=y_train.values)
                .groupby('Churn', group_keys=False)
                .apply(lambda x: x.sample(frac=500/len(X_train_encoded), random_state=42))
                .drop(columns='Churn')
            )
        #------------------- Train models -----------------------------

        cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        immutable = {"Is_female","SeniorCitizen","Partner","Dependents","tenure","TotalCharges","IsLongTermContract",
             "AvgMonthlyCharge","AvgPricePerService","OnlineServiceCount"}
        onehot_groups = {
            "Contract": ["Contract_One year","Contract_Two year"],
            "PaymentMethod": ["PaymentMethod_Credit card (automatic)","PaymentMethod_Electronic check","PaymentMethod_Mailed check"],
            "InternetService": ["InternetService_DSL","InternetService_Fiber optic","InternetService_No"],
        }
        if selected_prediction_model == 'Logistic Regression':
            
            # perform_primary_lg_training(scaled_X_train_features, y_train, scaled_X_test_features, y_test, "balanced")
            # cross_validate_lg_model(X_train_encoded, y_train, cross_validation, "balanced")
            # hyperparameter_tune_lg(X_train_encoded, y_train, cross_validation)
            metrics, tuned_model = retrain_lg_model(scaled_X_train_features, y_train, scaled_X_test_features, y_test, 0.5832, 0.00316)
            display_fairness_table_lr(X_test, tuned_model, scaled_X_test_features, threshold=0.5832)
            

            st.subheader("Logistic Regression Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_lr_metrics(metrics_tab, metrics)
            get_telco_lr_metrics_caption(column2)
            
            # Explainability models
            attribs = collect_attributions_for_tests(
                lr_model=tuned_model,
                X_train_scaled_bg=scaled_X_train_features.sample(200, random_state=0),
                X_val_scaled=scaled_X_test_features,

                rf_model=None,
                xgb_clf=None,
                X_train_enc=None,
                X_val_enc=None,

                xgb_booster=None,
                use_streamlit=True
            )
            A_lr = attribs["lr"]

            # val_tab, shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["Validation","SHAP", "LIME", "Counterfactuals"])
            shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["SHAP", "LIME", "Counterfactuals"])

            # Faithfulness
            
            # rep = faithfulness_report(
            #     model=tuned_model,
            #     X_val=scaled_X_test_features,
            #     y_val=y_test,
            #     A=A_lr,
            #     feature_names=scaled_X_test_features.columns.tolist(),
            #     n_steps=10,
            #     seed=42,
            #     use_proba=True
            # )
            # val_tab.markdown("### Faithfulness")
            # val_tab.write(f"Spearman ρ (rank vs. single-feature impact): **{rep['rho']:.3f}**")
            # val_tab.dataframe(rep["curve"])

            # x_row = scaled_X_test_features.iloc[0]
            # a_row = pd.Series(A_lr[0], index=scaled_X_test_features.columns)

            # rep = local_faithfulness_report(
            #     model=tuned_model,               # RF / LR / XGB model object
            #     x_row=x_row,
            #     a_row=a_row,
            #     background=background_data_scaled,
            #     n_draws=64,
            #     n_steps=10,
            #     replace_with="mean"              # or "draw"
            # )
            # val_tab.markdown("### Local Faithfulness (this customer)")
            # val_tab.write(f"Spearman ρ (local): **{rep['rho_local']:.3f}**")
            # val_tab.line_chart(pd.DataFrame({"p(k)": rep["deletion_curve"]["preds"]}, index=rep["deletion_curve"]["k"]))
            # val_tab.caption(
            #     f"Deletion curve AUC: {rep['deletion_curve']['auc']:.4f} (lower is better). "
            #     f"Baseline p0={rep['deletion_curve']['p0']:.3f}. "
            #     f"{'Flip at k=' + str(rep['flip_k']) if rep['flip_k'] is not None else ''}"
            # )
            # val_tab.dataframe(rep["single_feature_impacts"].head(12).to_frame("local_impact"))

            # X_train_for_lime = scaled_X_train_features  # use X_train_encoded for RF/XGB branches
            # X_test_for_lime = scaled_X_test_features
            # random_state = 42
            # # choose a row to test (must be same feature space as the explainer/model)
            # row_id = X_test_for_lime.index[0]           # e.g., scaled_X_test_features or X_test_encoded
            # x_row = X_test_for_lime.loc[row_id]

            # lime_explainer = LimeTabularExplainer(
            #     training_data=X_train_for_lime.to_numpy(),
            #     feature_names=X_train_for_lime.columns.tolist(),
            #     class_names=[str(c) for c in getattr(tuned_model, "classes_", [0, 1])],  # <-- FIXED
            #     mode="classification",
            #     discretize_continuous=True,
            #     sample_around_instance=True,
            #     random_state=random_state,
            # )

            # lt = lime_local_deletion_test(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     k_max=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Deletion curve AUC: **{lt['auc']:.4f}** (lower is better). "
            #     f"Baseline p0={lt['p0']:.3f}. "
            #     f"{'Flip at k=' + str(lt['flip_k']) if lt['flip_k'] is not None else 'No flip'}."
            # )

            # corr = lime_local_rank_vs_impact(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Spearman ρ (LIME rank vs single-feature impact): **{corr['rho']:.3f}** "
            #     f"(p={corr['pval']:.3g})."
            # )


            # Stability
            # get_attr_lr = lambda X: collect_attributions_for_tests(
            #     lr_model=tuned_model,
            #     X_train_scaled_bg=scaled_X_train_features.sample(200, random_state=0),
            #     X_val_scaled=X,
            #     use_streamlit=False
            # )["lr"]
            # stab = stability_report(get_attr_lr, scaled_X_test_features, n_boot=20, sample_frac=0.8, seed=42)
            # val_tab.markdown("### Stability")
            # val_tab.write(f"Kendall’s τ (mean ± sd): **{stab['kendall_tau_mean']:.3f} ± {stab['kendall_tau_std']:.3f}**")

            # ---------- LR: cache full-test attributions once (fast reuse) ----------
            # --- 1) Build a hashable fingerprint for LR ---
            # --- Hashable fingerprint for LR (for cache invalidation) ---
            # def _lr_model_key(m: LogisticRegression) -> str:
            #     coef = getattr(m, "coef_", None)
            #     intercept = getattr(m, "intercept_", None)
            #     n_iter = getattr(m, "n_iter_", None)
            #     parts = []
            #     if coef is not None:      parts.append(coef.ravel().tobytes())
            #     if intercept is not None: parts.append(np.atleast_1d(intercept).ravel().tobytes())
            #     if n_iter is not None:    parts.append(np.atleast_1d(n_iter).ravel().tobytes())
            #     parts.append(str(getattr(m, "C", None)).encode())
            #     parts.append(str(getattr(m, "penalty", None)).encode())
            #     parts.append(str(getattr(m, "solver", None)).encode())
            #     return str(abs(hash(b"||".join(parts))))

            # @st.cache_data(show_spinner=False)
            # def _lr_full_attr(
            #     _model: LogisticRegression,  # ignored by Streamlit hasher (leading underscore)
            #     model_key: str,
            #     X_train_bg: pd.DataFrame,
            #     X_val_scaled: pd.DataFrame,
            # ) -> pd.DataFrame:
            #     A = collect_attributions_for_tests(
            #         lr_model=_model,
            #         X_train_scaled_bg=X_train_bg,
            #         X_val_scaled=X_val_scaled,
            #         rf_model=None, xgb_clf=None,
            #         X_train_enc=None, X_val_enc=None,
            #         xgb_booster=None,
            #         use_streamlit=False
            #     )["lr"]
            #     return pd.DataFrame(A, index=X_val_scaled.index, columns=X_val_scaled.columns)

            # # --- use it (always pass model_key!) ---
            # lr_key   = _lr_model_key(tuned_model)
            # bg_sample = scaled_X_train_features.sample(min(200, len(scaled_X_train_features)), random_state=0)

            # A_lr_full = _lr_full_attr(
            #     _model=tuned_model,
            #     model_key=lr_key,
            #     X_train_bg=bg_sample,
            #     X_val_scaled=scaled_X_test_features,
            # )

            # def get_attr_lr_cached(X_batch: pd.DataFrame) -> np.ndarray:
            #     # try reuse from full cache
            #     try:
            #         return A_lr_full.loc[X_batch.index].to_numpy()
            #     except KeyError:
            #         # small recompute for perturbed rows not in A_lr_full
            #         A_tmp = collect_attributions_for_tests(
            #             lr_model=tuned_model,
            #             X_train_scaled_bg=bg_sample,
            #             X_val_scaled=X_batch,
            #             use_streamlit=False
            #         )["lr"]
            #         return A_tmp

            # with val_tab:
            #     st.markdown("### Local Stability (Logistic Regression)")
            #     lr_row_id = st.selectbox("Pick a row (LR):", scaled_X_test_features.index.tolist(), index=0, key="lr_row_pick")
            #     n_pert = st.slider("Perturbations", 10, 100, 30, key="lr_npert")
            #     noise = st.number_input("Numeric noise (σ)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, key="lr_noise")
            #     flip = st.number_input("Flip prob (binary/one-hot)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="lr_flip")

            #     if st.button("Run local stability (LR)"):
            #         # In the scaled matrix, nearly everything is numeric; treat one-hots as binary [0/1].
            #         numeric_cols = scaled_X_test_features.columns.tolist()
            #         # If you can enumerate true one-hots, pass them here; otherwise an empty dict is safe.
            #         report = local_stability_report_generic(
            #             get_attr_for_batch=get_attr_lr_cached,
            #             X_matrix=scaled_X_test_features,
            #             row_id=lr_row_id,
            #             numeric_cols=numeric_cols,
            #             binary_cols=[],                    # (optional) add pure binary cols here if you keep them separate
            #             onehot_groups=onehot_groups,       # if columns exist in scaled matrix with same names
            #             n_perturb=n_pert,
            #             noise_scale=noise,
            #             flip_prob=flip,
            #             k_list=(5,10)
            #         )
            #         st.write(f"Kendall’s τ (mean): **{report['kendall_tau_mean']:.3f}**")
            #         st.write(f"Spearman ρ (mean): **{report['spearman_rho_mean']:.3f}**")
            #         st.write(f"Avg std(|attrib|): **{report['mean_attr_std']:.4f}**")
            #         st.write({f"Top-{k} overlap": f"{report['topk_overlap_mean'][k]:.2f}" for k in (5,10)})
            #         st.dataframe(report["attr_std_by_feature"].head(12).to_frame("std").style.format({"std": "{:.4f}"}))


            # Sanity (label randomization)
        #     san = sanity_report(
        #     model=tuned_model,
        #     X_val=scaled_X_test_features,
        #     y_val=y_test,
        #     get_attribs_callable=get_attr_lr_cached,  # <-- new name
        #     randomize="labels",
        #     seed=42
        # )
        #     val_tab.markdown("### Sanity")
        #     val_tab.write(f"Spearman ρ vs original after label randomization: **{san['spearman_rho_vs_original']:.3f}** "
        #                   "(should drop toward 0 if explanations depend on learned signal)")

            

            get_lr_explanation(tuned_model, background_data_scaled, scaled_X_test_features, shap_tab)
            lr_local_shap_by_truth(
                lr_model=tuned_model,
                background_data=background_data_scaled,
                X_test_scaled=scaled_X_test_features,
                X_test=X_test,
                y_test=y_test,             # must be aligned with X_test_scaled index
                shap_tab=shap_tab,
                threshold=0.5832,            # your operating threshold
                top_display=12
                )   
            get_lime_explanations_binary(tuned_model, X_test, scaled_X_train_features, scaled_X_test_features, y_test, 0.5832, lime_tab, title_prefix="Local LIME – Logistic Regression")        

            lr_for_dice = ThresholdedModel(tuned_model, threshold=0.5832)

            continuous_features = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",          
                "AvgMonthlyCharge",
                "OnlineServiceCount",
                "AvgPricePerService",
            ]

            features_to_vary = [
                "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                "PaperlessBilling","MonthlyCharges",
                "Contract_One year","Contract_Two year",
                "PaymentMethod_Credit card (automatic)","PaymentMethod_Electronic check","PaymentMethod_Mailed check",
                "InternetService_Fiber optic","InternetService_No",
            ]

            results = get_counterfactual_analysis(
                y_test=y_test,
                X_test=scaled_X_test_features,
                X_train=scaled_X_train_features,
                y_train=y_train,
                model=lr_for_dice,                         # must expose predict_proba
                continuous_features=continuous_features,
                counterfactual_tab=counterfactual_tab,
                outcome_name="Churn",
                total_CFs=6,
                features_to_vary=features_to_vary,                    # or "all"
                permitted_range={"MonthlyCharges":[15,150]},
                scaler=scaler,
                numeric_feature_names_in_scaler_order=num_cols_in_scaler_order,
                immutable_features=immutable,
                onehot_groups=onehot_groups,
            )

        elif selected_prediction_model == 'Random Forest':
            
            # perform_primary_rf_training(X_train_encoded, y_train, X_test_encoded, y_test, "balanced")
            # cross_validate_rf_model(X_train_encoded, y_train, cross_validation, "balanced")
            # hyperparameter_tune_rf(X_train_encoded, y_train, cross_validation)
            metrics, tuned_model = retrain_rf_model(X_train_encoded, y_train, X_test_encoded, y_test, 0.5418, 477, 17, 12, 0.3, None)

            # def group_fairness_table(df, group_col, y_true='y_true', y_proba='y_proba', threshold=0.5418, reference=None):
            #     """Return a per-group fairness table and the chosen reference group."""
            #     # pick reference = largest group if not provided
            #     ref = reference or df[group_col].value_counts().idxmax()

            #     def recall_pos(g):
            #         mask = g[y_true] == 1
            #         if mask.sum() == 0:
            #             return np.nan
            #         y_hat = (g[y_proba] >= threshold).astype(int)
            #         return recall_score(g.loc[mask, y_true], y_hat[mask])

            #     agg = (
            #         df.groupby(group_col)
            #           .apply(lambda g: pd.Series({
            #               'n': len(g),
            #               'observed_churn_rate': g[y_true].mean(),
            #               'mean_pred_proba': g[y_proba].mean(),
            #               f'prediction_rate@{threshold:.3f}': (g[y_proba] >= threshold).mean(),
            #               'recall_pos': recall_pos(g),
            #           }))
            #           .reset_index()
            #     )

            #     ref_mean = agg.loc[agg[group_col] == ref, 'mean_pred_proba'].values[0]
            #     agg[f'Δ_vs_{ref}'] = (agg['mean_pred_proba'] - ref_mean).abs()
            #     agg['stat_parity_ok(≤0.05)'] = agg[f'Δ_vs_{ref}'] <= 0.05
            #     return agg.sort_values('n', ascending=False), ref

            # group_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]  
            # threshold = 0.5418
            # X_test_original_groups = X_test[group_cols].copy()
            # if hasattr(y_test, "dtype") and y_test.dtype == object:
            #     y_true = y_test.map({"No": 0, "Yes": 1}).astype(int)
            # elif isinstance(y_test, pd.DataFrame) and "Churn" in y_test.columns:
            #     y_true = y_test["Churn"].map({"No": 0, "Yes": 1}).astype(int)
            # else:
            #     y_true = y_test.astype(int)  # already 0/1

            # if hasattr(tuned_model, "predict_proba"):           # single estimator
            #     models = {"logreg_final": tuned_model}
            # elif isinstance(tuned_model, (list, tuple)):        # list/tuple of estimators
            #     models = {f"model_{i}": m for i, m in enumerate(tuned_model)}
            # elif isinstance(tuned_model, dict):                  # already a dict
            #     models = tuned_model
            # else:
            #     raise TypeError("tuned_model must be an estimator, list/tuple of estimators, or dict.")

            # X_for_pred = X_test_encoded

            # results = {}
            # for name, model in models.items():
            #     # get probabilities for the positive class
            #     y_proba = model.predict_proba(X_for_pred)[:, 1]

            #     # build evaluation frame aligned by index
            #     df_eval = X_test_original_groups.copy()
            #     df_eval["y_true"] = pd.Series(y_true.values, index=df_eval.index)
            #     df_eval["y_proba"] = pd.Series(y_proba, index=df_eval.index)

            # # per-group tables for this model
            #     model_tables = {}
            #     for gcol in group_cols:
            #         sub = df_eval[[gcol, "y_true", "y_proba"]].dropna()
            #         table, ref = group_fairness_table(
            #             sub, group_col=gcol, y_true="y_true", y_proba="y_proba", threshold=threshold
            #         )
            #         model_tables[gcol] = (table, ref)
            #     results[name] = model_tables

            # st.title("Fairness Evaluation — Group-Based Metrics")
            # st.write("Global threshold for recall comparisons (equal opportunity): ", threshold)
            # for model_name, tables in results.items():
            #     with st.expander(f"Model: {model_name}", expanded=False):
            #         for gcol, (table, ref) in tables.items():
            #             st.subheader(f"Group column: {gcol} (ref: {ref})")
            #             st.dataframe(table, use_container_width=True)

            #             # CSV download
            #             csv_buf = StringIO()
            #             table.to_csv(csv_buf, index=False)
            #             st.download_button(
            #                 label=f"Download CSV — {model_name} · {gcol}",
            #                 data=csv_buf.getvalue(),
            #                 file_name=f"fairness_{model_name}_{gcol}.csv",
            #                 mime="text/csv"
            #             )

            st.subheader("Random Forest Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_rf_metrics(metrics_tab, metrics)
            get_telco_rf_metrics_caption(column2)

            # Explainability models
            # ---- SHAP attribution matrices for tests (Telco RF) ----
            attribs = collect_attributions_for_tests(
                lr_model=None,
                X_train_scaled_bg=None,
                X_val_scaled=None,

                rf_model=tuned_model,
                xgb_clf=None,
                X_train_enc=X_train_encoded,
                X_val_enc=X_test_encoded,

                xgb_booster=None,
                use_streamlit=True
            )
            A_rf = attribs["rf"]        # (n_test, n_features_encoded)

            # val_tab, shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["Validation","SHAP", "LIME", "Counterfactuals"])
            shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["SHAP", "LIME", "Counterfactuals"])


            # Faithfulness
            
            # rep = faithfulness_report(
            #     model=tuned_model,
            #     X_val=X_test_encoded,
            #     y_val=y_test,
            #     A=A_rf,
            #     feature_names=X_test_encoded.columns.tolist(),
            #     n_steps=10,
            #     seed=42,
            #     use_proba=True
            # )
            # val_tab.markdown("### Faithfulness")
            # val_tab.write(f"Spearman ρ (rank vs. single-feature impact): **{rep['rho']:.3f}**")
            # val_tab.dataframe(rep["curve"])

            # x_row = X_test_encoded.iloc[0]
            # a_row = pd.Series(A_rf[0], index=X_test_encoded.columns)

            # rep = local_faithfulness_report(
            #     model=tuned_model,               # RF / LR / XGB model object
            #     x_row=x_row,
            #     a_row=a_row,
            #     background=background_data_encoded,
            #     n_draws=64,
            #     n_steps=10,
            #     replace_with="mean"              # or "draw"
            # )
            # val_tab.markdown("### Local Faithfulness (this customer)")
            # val_tab.write(f"Spearman ρ (local): **{rep['rho_local']:.3f}**")
            # val_tab.line_chart(pd.DataFrame({"p(k)": rep["deletion_curve"]["preds"]}, index=rep["deletion_curve"]["k"]))
            # val_tab.caption(
            #     f"Deletion curve AUC: {rep['deletion_curve']['auc']:.4f} (lower is better). "
            #     f"Baseline p0={rep['deletion_curve']['p0']:.3f}. "
            #     f"{'Flip at k=' + str(rep['flip_k']) if rep['flip_k'] is not None else ''}"
            # )
            # val_tab.dataframe(rep["single_feature_impacts"].head(12).to_frame("local_impact"))

            # X_train_for_lime = X_train_encoded  # use X_train_encoded for RF/XGB branches
            # X_test_for_lime = X_test_encoded
            # random_state = 42
            # choose a row to test (must be same feature space as the explainer/model)
            # row_id = X_test_for_lime.index[0]           # e.g., scaled_X_test_features or X_test_encoded
            # x_row = X_test_for_lime.loc[row_id]

            # lime_explainer = LimeTabularExplainer(
            #     training_data=X_train_for_lime.to_numpy(),
            #     feature_names=X_train_for_lime.columns.tolist(),
            #     class_names=[str(c) for c in getattr(tuned_model, "classes_", [0, 1])],  # <-- FIXED
            #     mode="classification",
            #     discretize_continuous=True,
            #     sample_around_instance=True,
            #     random_state=random_state,
            # )

            # lt = lime_local_deletion_test(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     k_max=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Deletion curve AUC: **{lt['auc']:.4f}** (lower is better). "
            #     f"Baseline p0={lt['p0']:.3f}. "
            #     f"{'Flip at k=' + str(lt['flip_k']) if lt['flip_k'] is not None else 'No flip'}."
            # )

            # corr = lime_local_rank_vs_impact(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Spearman ρ (LIME rank vs single-feature impact): **{corr['rho']:.3f}** "
            #     f"(p={corr['pval']:.3g})."
            # )

            # # Stability
            # # Ensure indices are simple and unique
            # X_test_encoded_copy = X_test_encoded.copy()
            # X_test_encoded_copy.index = pd.RangeIndex(len(X_test_encoded))

            # # # 1) Precompute RF attributions once on the FULL test set
            # A_rf_full = pd.DataFrame(
            #     A_rf,  # from collect_attributions_for_tests on FULL X_test_encoded
            #     index=X_test_encoded_copy.index,
            #     columns=X_test_encoded_copy.columns
            # )

            # # Build a position map for robust slicing even if a resample resets index
            # pos_map = pd.Series(np.arange(len(X_test_encoded_copy)), index=X_test_encoded_copy.index)

            # def get_attr_rf_cached(X_subset: pd.DataFrame):
            #     # Primary: index-aligned slice (fast path)
            #     try:
            #         return A_rf_full.loc[X_subset.index].to_numpy()
            #     except KeyError:
            #         # Fallback: map indices to positions (handles reset_index in resample)
            #         idx = pos_map.reindex(X_subset.index)
            #         if idx.isna().any():
            #             # As a last resort, assume positional slice (only works if X_subset
            #             # is taken by .iloc from the same matrix)
            #             return A_rf_full.to_numpy()[np.arange(len(X_subset)), :]
            #         return A_rf_full.to_numpy()[idx.to_numpy().astype(int), :]

            # Now run stability (cheap, no SHAP recompute)
            # stab = stability_report(get_attr_rf_cached, X_test_encoded_copy, n_boot=20, sample_frac=0.8, seed=42)
            # val_tab.markdown("### Stability")
            # val_tab.write(f"Kendall’s τ (mean ± sd): **{stab['kendall_tau_mean']:.3f} ± {stab['kendall_tau_std']:.3f}**")

            # --- 1) Build a hashable fingerprint for LR ---
            # --- Hashable fingerprint for LR (for cache invalidation) ---
            # def _rf_model_key(m: RandomForestClassifier) -> str:
            #     """
            #     Build a hashable fingerprint for a fitted RF.
            #     Includes: key hyperparams, n_estimators, feature_importances_,
            #     oob_score_ (if present). This is sufficient to invalidate cache
            #     when the trained forest changes.
            #     """
            #     parts = []
            #     # core hyperparams that change the fit
            #     hp = m.get_params(deep=False)
            #     key_params = (
            #         hp.get("n_estimators"), hp.get("criterion"),
            #         hp.get("max_depth"), hp.get("max_features"),
            #         hp.get("min_samples_split"), hp.get("min_samples_leaf"),
            #         hp.get("min_weight_fraction_leaf"), hp.get("max_leaf_nodes"),
            #         hp.get("bootstrap"), hp.get("class_weight"),
            #         hp.get("ccp_alpha"), hp.get("random_state")
            #     )
            #     parts.append(str(key_params).encode())

            #     # learned quantities
            #     if hasattr(m, "n_estimators"):
            #         parts.append(str(m.n_estimators).encode())
            #     if hasattr(m, "feature_importances_"):
            #         parts.append(np.asarray(m.feature_importances_, dtype=np.float64).tobytes())
            #     if hasattr(m, "oob_score_"):
            #         parts.append(str(m.oob_score_).encode())

            #     return str(abs(hash(b"||".join(parts))))

            # # -------- Cache full-test RF attributions (ignore unhashable model) --------
            # @st.cache_data(show_spinner=False)
            # def _rf_full_attr(
            #     _model: RandomForestClassifier,  # leading underscore => Streamlit won't hash it
            #     model_key: str,                  # <- hashable fingerprint to invalidate when RF changes
            #     X_train_enc: pd.DataFrame,       # needed by your collector
            #     X_val_enc: pd.DataFrame
            # ) -> pd.DataFrame:
            #     A = collect_attributions_for_tests(
            #         lr_model=None,
            #         X_train_scaled_bg=None, X_val_scaled=None,
            #         rf_model=_model,
            #         xgb_clf=None,
            #         X_train_enc=X_train_enc,
            #         X_val_enc=X_val_enc,
            #         xgb_booster=None,
            #         use_streamlit=False
            #     )["rf"]  # -> (n_val, n_features)
            #     return pd.DataFrame(A, index=X_val_enc.index, columns=X_val_enc.columns)

            # # -------- Build cache + getter in your RF branch --------
            # rf_key = _rf_model_key(tuned_model)
            # A_rf_full_df = _rf_full_attr(
            #     _model=tuned_model,
            #     model_key=rf_key,
            #     X_train_enc=X_train_encoded,
            #     X_val_enc=X_test_encoded,
            # )

            # def get_attr_rf_cached(X_batch: pd.DataFrame) -> np.ndarray:
            #     # fast path: reuse precomputed rows
            #     try:
            #         return A_rf_full_df.loc[X_batch.index].to_numpy()
            #     except KeyError:
            #         # small-batch recompute for perturbed rows not in the cache
            #         A_tmp = collect_attributions_for_tests(
            #             rf_model=tuned_model,
            #             X_train_enc=X_train_encoded,
            #             X_val_enc=X_batch,
            #             use_streamlit=False
            #         )["rf"]
            #         return A_tmp

            # -------- Local stability UI (RF) --------
            # with val_tab:
            #     st.markdown("### Local Stability (Random Forest)")
            #     rf_row_id = st.selectbox("Pick a row (RF):", X_test_encoded.index.tolist(), index=0, key="rf_row_pick")
            #     n_pert = st.slider("Perturbations", 10, 100, 30, key="rf_npert")
            #     noise = st.number_input("Numeric noise (σ)", 0.0, 1.0, 0.02, 0.01, key="rf_noise")
            #     flip = st.number_input("Flip prob (binary/one-hot)", 0.0, 1.0, 0.05, 0.01, key="rf_flip")

            #     # heuristics: treat uint8 one-hots as binary; the rest as numeric
            #     binary_cols_rf  = [c for c in X_test_encoded.columns
            #                        if set(pd.unique(X_test_encoded[c])).issubset({0, 1})]
            #     numeric_cols_rf = [c for c in X_test_encoded.columns if c not in binary_cols_rf]

            #     if st.button("Run local stability (RF)"):
            #         report = local_stability_report_generic(
            #             get_attr_for_batch=get_attr_rf_cached,
            #             X_matrix=X_test_encoded,
            #             row_id=rf_row_id,
            #             numeric_cols=numeric_cols_rf,
            #             binary_cols=binary_cols_rf,
            #             onehot_groups=onehot_groups,  # your existing dict
            #             n_perturb=n_pert,
            #             noise_scale=noise,
            #             flip_prob=flip,
            #             k_list=(5, 10)
            #         )
            #         st.write(f"Kendall’s τ (mean): **{report['kendall_tau_mean']:.3f}**")
            #         st.write(f"Spearman ρ (mean): **{report['spearman_rho_mean']:.3f}**")
            #         st.write(f"Avg std(|attrib|): **{report['mean_attr_std']:.4f}**")
            #         st.write({f"Top-{k} overlap": f"{report['topk_overlap_mean'][k]:.2f}" for k in (5, 10)})
            #         st.dataframe(report["attr_std_by_feature"].head(12).to_frame("std").style.format({"std": "{:.4f}"}))



            # Sanity (label randomization)
            # def _to_hashable_tuple(idx: pd.Index) -> tuple:
            #     vals = idx.tolist()
            #     try:
            #         tuple(vals)
            #         return tuple(vals)
            #     except TypeError:
            #         return tuple(map(str, vals))
            
            # @st.cache_data(show_spinner=False)
            # def _rf_attr_cache(A_rf_array: np.ndarray,
            #                    index_values: tuple,
            #                    column_values: tuple) -> tuple[pd.DataFrame, pd.Series]:
            #     """
            #     Materialize RF attributions into a DataFrame with original index/columns,
            #     and build a position map for robust slicing.
            #     """
            #     x_index = pd.Index(list(index_values))
            #     x_columns = pd.Index(list(column_values))
            #     A_rf_full_df = pd.DataFrame(A_rf_array, index=x_index, columns=x_columns).astype("float64")
            
            #     if A_rf_full_df.shape[0] != len(x_index):
            #         raise ValueError("A_rf_full rows != X_test_encoded rows.")
            #     if not A_rf_full_df.columns.equals(x_columns):
            #         raise ValueError("A_rf_full columns differ from X_test_encoded columns.")
            #     if not x_index.is_unique:
            #         st.warning("X_test_encoded has non-unique index; positional fallback may be used.")
            
            #     pos_map = pd.Series(np.arange(len(x_index)), index=x_index)
            #     return A_rf_full_df, pos_map
            
            # def make_get_attr_rf_cached(A_rf_full_df: pd.DataFrame, pos_map: pd.Series):
            #     def _getter(X_subset: pd.DataFrame) -> np.ndarray:
            #         if len(X_subset.index) and X_subset.index.equals(A_rf_full_df.index):
            #             return A_rf_full_df.to_numpy()
            
            #         if X_subset.index.isin(A_rf_full_df.index).all():
            #             return A_rf_full_df.reindex(X_subset.index).to_numpy()
            
            #         idx = pos_map.reindex(X_subset.index)
            #         if not idx.isna().any():
            #             return A_rf_full_df.to_numpy()[idx.to_numpy().astype(int), :]
            
            #         st.warning("Attribution slice fell back to pure positional matching; verify index handling.", icon="⚠️")
            #         n = len(X_subset)
            #         if n > len(A_rf_full_df):
            #             raise ValueError("Subset longer than reference — cannot slice positionally.")
            #         return A_rf_full_df.to_numpy()[:n, :]
            #     return _getter
            
            # # ---- RF sanity: build in strict order, keep everything inside the RF branch ----
            # # 1) Full-test RF attributions as array
            # attribs = collect_attributions_for_tests(
            #     rf_model=tuned_model,
            #     X_train_enc=X_train_encoded,
            #     X_val_enc=X_test_encoded,
            #     lr_model=None, xgb_clf=None, X_train_scaled_bg=None, X_val_scaled=None,
            #     xgb_booster=None,
            #     use_streamlit=False
            # )
            # A_rf_array = attribs["rf"]  # (n_test, n_features)
            
            # # 2) Cache DF + position map (hashable surrogates)
            # A_rf_full_df, pos_map = _rf_attr_cache(
            #     A_rf_array,
            #     _to_hashable_tuple(X_test_encoded.index),
            #     _to_hashable_tuple(X_test_encoded.columns),
            # )
            
            # # 3) Getter (reuse cache; recompute only for unseen perturbed rows)
            # get_attr_rf_cached = make_get_attr_rf_cached(A_rf_full_df, pos_map)
            
            # 4) Sanity test UI
            # with val_tab:
            #     st.markdown("### Sanity (Random Forest)")
            #     san = sanity_report(
            #         model=tuned_model,
            #         X_val=X_test_encoded,
            #         y_val=y_test,
            #         get_attribs_callable=get_attr_rf_cached,
            #         randomize="labels",
            #         seed=42
            #     )
            #     st.write(
            #         f"Spearman ρ vs original after label randomization: "
            #         f"**{san['spearman_rho_vs_original']:.3f}** "
            #         "(should drop toward 0 if explanations depend on learned signal)"
            #     )


            # get_rf_explanation(tuned_model, background_data_encoded, X_test_encoded, shap_tab)
            rf_local_shap_by_truth(tuned_model, X_test_encoded, X_test, y_test, 0.5418, shap_tab, background_data_encoded)
            get_lime_explanations_binary(tuned_model, X_test, X_train_encoded, X_test_encoded, y_test, 0.5418, lime_tab, title_prefix="Local LIME – Random Forest")

            rf_for_dice = ThresholdedModel(tuned_model, threshold=0.5418)

            continuous_features = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",          
                "AvgMonthlyCharge",
                "OnlineServiceCount",
                "AvgPricePerService",
            ]

            features_to_vary = [
                "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                "PaperlessBilling","MonthlyCharges",
                "Contract_One year","Contract_Two year",
                "PaymentMethod_Credit card (automatic)","PaymentMethod_Electronic check","PaymentMethod_Mailed check",
                "InternetService_Fiber optic","InternetService_No",
            ]

            permitted_range = {"MonthlyCharges": [15, 150]}

            results = get_counterfactual_analysis(
            y_test=y_test,
            X_test=X_test_encoded,          
            X_train=X_train_encoded,
            y_train=y_train,
            model=rf_for_dice,              
            continuous_features=continuous_features,
            counterfactual_tab=counterfactual_tab,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,       # raw bounds
            scaler=None,                           
            numeric_feature_names_in_scaler_order=None,
            immutable_features=immutable,
            onehot_groups=onehot_groups
        )

        elif selected_prediction_model == "XGBoost":
            # Train
            # perform_primary_xgb_training(X_train_encoded, y_train, X_test_encoded, y_test, True)
            # cross_validate_xgb_model(X_train_encoded, y_train, cross_validation, True)
            # best_stage1, rs_results, _ = xgb_random_search(
            # X_train_encoded, y_train, cross_validation,
            # n_iter=60,                # 60–100 is a good start
            # pos_label=1,
            # optimize_max_f1=True      # keep consistent with your LR threshold tuning
            # )
            # print("\n[Stage 1] Best (random):", best_stage1)

            # # --- Stage 2: Grid refinement around the Stage-1 best
            # best_stage2, gs_results = xgb_grid_refine(
            #     X_train_encoded, y_train, cross_validation,
            #     best_params_stage1=best_stage1,
            #     pos_label=1,
            #     optimize_max_f1=True
            # )
            # print("\n[Stage 2] Best (refined):", best_stage2)
            row, bst, params = retrain_xgb_model(X_train_encoded, y_train, X_test_encoded, y_test, 
                                                 0.58, 0.05, 5, 0.872, 0.853, 0.715, 17.402, 0.006)

            # def group_fairness_table(df, group_col, y_true='y_true', y_proba='y_proba', threshold=0.58, reference=None):
            #     ref = reference or df[group_col].value_counts().idxmax()

            #     def recall_pos(g):
            #         mask = g[y_true] == 1
            #         if mask.sum() == 0:
            #             return np.nan
            #         y_hat = (g[y_proba] >= threshold).astype(int)
            #         return recall_score(g.loc[mask, y_true], y_hat[mask])

            #     agg = (
            #         df.groupby(group_col)
            #           .apply(lambda g: pd.Series({
            #               'n': len(g),
            #               'observed_churn_rate': g[y_true].mean(),
            #               'mean_pred_proba': g[y_proba].mean(),
            #               f'prediction_rate@{threshold:.3f}': (g[y_proba] >= threshold).mean(),
            #               'recall_pos': recall_pos(g),
            #           }))
            #           .reset_index()
            #     )
            #     ref_mean = agg.loc[agg[group_col] == ref, 'mean_pred_proba'].values[0]
            #     agg[f'Δ_vs_{ref}'] = (agg['mean_pred_proba'] - ref_mean).abs()
            #     agg['stat_parity_ok(≤0.05)'] = agg[f'Δ_vs_{ref}'] <= 0.05
            #     return agg.sort_values('n', ascending=False), ref

            # # --- config / inputs ---
            # threshold = 0.58
            # group_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]

            # # IMPORTANT: use the RAW test DF for grouping, not the encoded/scaled matrix.
            # X_test_original_groups = X_test[group_cols].copy()

            # # y_true must be 0/1
            # if isinstance(y_test, pd.DataFrame) and "Churn" in y_test.columns:
            #     y_true = y_test["Churn"].map({"No": 0, "Yes": 1}).astype(int)
            # elif getattr(y_test, "dtype", None) == object:
            #     y_true = y_test.map({"No": 0, "Yes": 1}).astype(int)
            # else:
            #     y_true = y_test.astype(int)

            # # Choose the feature matrix your Booster was trained on (typically encoded, not scaled unless you scaled during training)
            # X_for_pred = X_test_encoded  # if your Booster was trained on the encoded matrix

            # # --- normalize `tuned_model` to a dict of models ---
            # models = {}
            # # You can add more models here later; for now just the Booster:
            # models["xgb_final"] = bst  # bst returned by retrain_xgb_model(...)

            # def predict_proba_any(model, X):
            #     # scikit-like estimators
            #     if hasattr(model, "predict_proba"):
            #         return model.predict_proba(X)[:, 1]
            #     # XGBoost Booster
            #     if isinstance(model, xgb.Booster):
            #         # Ensure X is numpy/pandas -> DMatrix
            #         dtest = xgb.DMatrix(X)
            #         # For binary:logistic this returns probabilities
            #         return model.predict(dtest)
            #     # decision_function fallback (e.g., linear SVM)
            #     if hasattr(model, "decision_function"):
            #         from scipy.special import expit
            #         return expit(model.decision_function(X))
            #     # last resort: predict hard labels (not ideal)
            #     yhat = model.predict(X)
            #     return np.asarray(yhat).astype(float)

            # # --- compute results ---
            # results = {}
            # for name, model in models.items():
            #     y_proba = predict_proba_any(model, X_for_pred)

            #     df_eval = X_test_original_groups.copy()
            #     df_eval["y_true"]  = pd.Series(y_true.values, index=df_eval.index)
            #     df_eval["y_proba"] = pd.Series(y_proba,       index=df_eval.index)

            #     model_tables = {}
            #     for gcol in group_cols:
            #         sub = df_eval[[gcol, "y_true", "y_proba"]].dropna()
            #         table, ref = group_fairness_table(sub, group_col=gcol, threshold=threshold)
            #         model_tables[gcol] = (table, ref)
            #     results[name] = model_tables

            # # --- Streamlit UI ---
            # st.title("Fairness Evaluation — Group-Based Metrics")
            # st.write("Global threshold for recall comparisons (equal opportunity): ", threshold)
            # for model_name, tables in results.items():
            #     with st.expander(f"Model: {model_name}", expanded=False):
            #         for gcol, (table, ref) in tables.items():
            #             st.subheader(f"Group column: {gcol} (ref: {ref})")
            #             st.dataframe(table, use_container_width=True)
            #             csv_buf = StringIO()
            #             table.to_csv(csv_buf, index=False)
            #             st.download_button(
            #                 label=f"Download CSV — {model_name} · {gcol}",
            #                 data=csv_buf.getvalue(),
            #                 file_name=f"fairness_{model_name}_{gcol}.csv",
            #                 mime="text/csv"
            #             )
            st.subheader("XGBoost Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_xgb_metrics(metrics_tab, row)
            # get_telco_xgb_metrics_caption(column2)

            # Explainability models
            # ---- SHAP attribution matrices for tests (Telco XGB) ----
            attribs = collect_attributions_for_tests(
                lr_model=None,
                X_train_scaled_bg=None,
                X_val_scaled=None,

                rf_model=None,
                xgb_clf=None,                           # you have a native Booster in `bst`
                X_train_enc=X_train_encoded,
                X_val_enc=X_test_encoded,

                xgb_booster=bst,                        # native Booster path (pred_contribs)
                use_streamlit=True
            )
            A_xgb_booster = attribs["xgb_booster"]     # (n_test, n_features_encoded), Δ log-odds
            
            # val_tab, shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["Validation","SHAP", "LIME", "Counterfactuals"])
            shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["SHAP", "LIME", "Counterfactuals"])

            # Faithfulness
            
            # rep = faithfulness_report(
            #     model=XGBoosterAdapter(bst, objective="binary:logistic"),
            #     X_val=X_test_encoded,
            #     y_val=y_test,
            #     A=A_xgb_booster,
            #     feature_names=X_test_encoded.columns.tolist(),
            #     n_steps=10,
            #     seed=42,
            #     use_proba=True
            # )
            # val_tab.markdown("### Faithfulness")
            # val_tab.write(f"Spearman ρ (rank vs. single-feature impact): **{rep['rho']:.3f}**")
            # val_tab.dataframe(rep["curve"])

            # x_row = X_test_encoded.iloc[0]
            # a_row = pd.Series(A_xgb_booster[0], index=X_test_encoded.columns)

            # rep = local_faithfulness_report(
            #     model=bst,               # RF / LR / XGB model object
            #     x_row=x_row,
            #     a_row=a_row,
            #     background=background_data_encoded,
            #     n_draws=64,
            #     n_steps=10,
            #     replace_with="mean"              # or "draw"
            # )
            # val_tab.markdown("### Local Faithfulness (this customer)")
            # val_tab.write(f"Spearman ρ (local): **{rep['rho_local']:.3f}**")
            # val_tab.line_chart(pd.DataFrame({"p(k)": rep["deletion_curve"]["preds"]}, index=rep["deletion_curve"]["k"]))
            # val_tab.caption(
            #     f"Deletion curve AUC: {rep['deletion_curve']['auc']:.4f} (lower is better). "
            #     f"Baseline p0={rep['deletion_curve']['p0']:.3f}. "
            #     f"{'Flip at k=' + str(rep['flip_k']) if rep['flip_k'] is not None else ''}"
            # )
            # val_tab.dataframe(rep["single_feature_impacts"].head(12).to_frame("local_impact"))

            # X_train_for_lime = X_train_encoded  # use X_train_encoded for RF/XGB branches
            # X_test_for_lime = X_test_encoded
            # random_state = 42
            # # choose a row to test (must be same feature space as the explainer/model)
            # row_id = X_test_for_lime.index[0]           # e.g., scaled_X_test_features or X_test_encoded
            # x_row = X_test_for_lime.loc[row_id]
            # tuned_model = bst

            # if hasattr(tuned_model, "predict_proba"):
            #     # sklearn model (LR, RF, XGBClassifier)
            #     predict_fn = tuned_model.predict_proba
            #     class_names = [str(c) for c in getattr(tuned_model, "classes_", [0, 1])]
            # else:
            #     # native Booster (bst)
            #     booster = bst if isinstance(bst, xgb.Booster) else bst.get_booster()
            #     feat_names = X_train_for_lime.columns.tolist()
            #     def predict_fn(X_batch: np.ndarray) -> np.ndarray:
            #         dm = xgb.DMatrix(X_batch, feature_names=feat_names)
            #         p1 = booster.predict(dm).reshape(-1)
            #         return np.column_stack([1.0 - p1, p1])
            #     class_names = ["0", "1"]

            # # ---- Create LIME explainer on TRAIN data (same feature space as model) ----
            # lime_explainer = LimeTabularExplainer(
            #     training_data=X_train_for_lime.to_numpy(),
            #     feature_names=X_train_for_lime.columns.tolist(),
            #     class_names=class_names,
            #     mode="classification",
            #     discretize_continuous=True,
            #     sample_around_instance=True,
            #     random_state=42,
            # )

            # # ---- Run the tests (note: we pass predict_fn, not model) ----
            # lt = lime_xgb_local_deletion_test(
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     predict_fn=predict_fn,
            #     num_features=10,
            #     k_max=10,
            #     mask_strategy="mean",
            # )
            # val_tab.write(
            #     f"Deletion curve AUC: **{lt['auc']:.4f}** (lower is better). "
            #     f"Baseline p0={lt['p0']:.3f}. "
            #     f"{'Flip at k='+str(lt['flip_k']) if lt['flip_k'] is not None else 'No flip'}."
            # )

            # corr = lime_xgb_local_rank_vs_impact(
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     predict_fn=predict_fn,
            #     num_features=10,
            #     mask_strategy="mean",
            # )
            # val_tab.write(
            #     f"Spearman ρ (LIME rank vs single-feature impact): **{corr['rho']:.3f}** "
            #     f"(p={corr['pval']:.3g})."
            # )

            # # Stability
            # bst_patched = _patch_base_score_in_modelfile(bst)

            # # # Then use this patched model instead of bst
            # get_attr_xgb = lambda X: collect_attributions_for_tests(
            #     xgb_booster=bst_patched,           # ✅ Use patched Booster, not rf_model
            #     X_val_enc=X,
            #     use_streamlit=False
            # )["xgb_booster"]

            # stab = stability_report(get_attr_xgb, X_test_encoded, n_boot=8, sample_frac=0.8, seed=42)

            # val_tab.markdown("### Stability")
            # val_tab.write(f"Kendall’s τ (mean ± sd): **{stab['kendall_tau_mean']:.3f} ± {stab['kendall_tau_std']:.3f}**")

            # def _xgb_model_key(model) -> str:
            #     """
            #     Deterministic fingerprint for a trained XGBoost model.
            #     Works for xgb.Booster and xgb.XGBClassifier.
            #     """
            #     # get raw bytes of the trained trees
            #     if isinstance(model, xgb.Booster):
            #         raw = model.save_raw()              # bytearray
            #         raw_b = bytes(raw)                  # -> bytes
            #         base = raw_b
            #         extra = b""
            #     elif hasattr(model, "get_booster"):     # XGBClassifier
            #         booster = model.get_booster()
            #         raw = booster.save_raw()            # bytearray
            #         raw_b = bytes(raw)
            #         base = raw_b
            #         # include key hyperparams to be extra-safe
            #         hp_names = ["n_estimators", "max_depth", "learning_rate", "subsample",
            #                     "colsample_bytree", "colsample_bylevel", "colsample_bynode",
            #                     "reg_lambda", "reg_alpha", "gamma", "min_child_weight", "random_state"]
            #         hp_vals = tuple(getattr(model, n, None) for n in hp_names)
            #         extra = repr(hp_vals).encode()
            #     else:
            #         # fallback
            #         return hashlib.sha1(repr(model).encode()).hexdigest()

            #     # stable digest string (streamlit-friendly)
            #     return hashlib.sha1(base + extra).hexdigest()

            # # 2) Cache full-test contribs (pred_contribs). Ignore unhashable model via leading underscore.
            # @st.cache_data(show_spinner=False)
            # def _xgb_full_attr(
            #     _model,                 # not hashed by Streamlit
            #     model_key: str,         # hashable fingerprint to invalidate cache when the model changes
            #     feature_names: tuple,   # tuple(X_test_encoded.columns)
            #     X_val_index: tuple,     # tuple(X_test_encoded.index)
            #     X_val_values: np.ndarray
            # ) -> pd.DataFrame:
            #     # get Booster
            #     if isinstance(_model, xgb.Booster):
            #         booster = _model
            #     elif hasattr(_model, "get_booster"):
            #         check_is_fitted(_model)
            #         booster = _model.get_booster()
            #     else:
            #         raise TypeError("Unsupported XGBoost model type for _xgb_full_attr")

            #     feats = list(feature_names)
            #     dm = xgb.DMatrix(X_val_values, feature_names=feats)
            #     contribs = booster.predict(dm, pred_contribs=True)   # (n, p+1) last col = bias
            #     A = contribs[:, :-1]                                 # drop bias to align with features
            #     return pd.DataFrame(A, index=list(X_val_index), columns=feats)

            # # 3) Pick which model to use for explanations
            # try:
            #     xgb_used = bst_patched   # if you patched base_score earlier
            # except NameError:
            #     xgb_used = bst           # Booster returned by retrain_xgb_model(...)

            # # Build cache
            # xgb_key = _xgb_model_key(xgb_used)
            # A_xgb_full = _xgb_full_attr(
            #     _model=xgb_used,
            #     model_key=xgb_key,
            #     feature_names=tuple(X_test_encoded.columns.tolist()),
            #     X_val_index=tuple(X_test_encoded.index.tolist()),
            #     X_val_values=X_test_encoded.to_numpy(),
            # )

            # # Getter: reuse cache for existing rows; recompute contribs only for perturbed rows
            # def get_attr_xgb_cached(X_batch: pd.DataFrame) -> np.ndarray:
            #     try:
            #         return A_xgb_full.loc[X_batch.index].to_numpy()
            #     except KeyError:
            #         feats = X_batch.columns.tolist()
            #         booster = xgb_used if isinstance(xgb_used, xgb.Booster) else xgb_used.get_booster()
            #         dm = xgb.DMatrix(X_batch[feats].to_numpy(), feature_names=feats)
            #         contribs = booster.predict(dm, pred_contribs=True)
            #         return contribs[:, :-1]

            # # 4) Local stability UI
            # with val_tab:
            #     st.markdown("### Local Stability (XGBoost)")
            #     xgb_row_id = st.selectbox("Pick a row (XGB):", X_test_encoded.index.tolist(), index=0, key="xgb_row_pick")
            #     n_pert = st.slider("Perturbations", 10, 100, 30, key="xgb_npert")
            #     noise = st.number_input("Numeric noise (σ)", 0.0, 1.0, 0.02, 0.01, key="xgb_noise")
            #     flip  = st.number_input("Flip prob (binary/one-hot)", 0.0, 1.0, 0.05, 0.01, key="xgb_flip")

            #     # One-hots are strictly {0,1}; the rest numeric
            #     binary_cols_xgb  = [c for c in X_test_encoded.columns
            #                         if set(pd.unique(X_test_encoded[c])).issubset({0, 1})]
            #     numeric_cols_xgb = [c for c in X_test_encoded.columns if c not in binary_cols_xgb]

                # if st.button("Run local stability (XGB)"):
                #     report = local_stability_report_generic(
                #         get_attr_for_batch=get_attr_xgb_cached,
                #         X_matrix=X_test_encoded,
                #         row_id=xgb_row_id,
                #         numeric_cols=numeric_cols_xgb,
                #         binary_cols=binary_cols_xgb,
                #         onehot_groups=onehot_groups,   # your existing dict
                #         n_perturb=n_pert,
                #         noise_scale=noise,
                #         flip_prob=flip,
                #         k_list=(5, 10)
                #     )
                #     st.write(f"Kendall’s τ (mean): **{report['kendall_tau_mean']:.3f}**")
                #     st.write(f"Spearman ρ (mean): **{report['spearman_rho_mean']:.3f}**")
                #     st.write(f"Avg std(|attrib|): **{report['mean_attr_std']:.4f}**")
                #     st.write({f"Top-{k} overlap": f"{report['topk_overlap_mean'][k]:.2f}" for k in (5, 10)})
                #     st.dataframe(report["attr_std_by_feature"].head(12).to_frame("std").style.format({"std": "{:.4f}"}))

            # Sanity (label randomization)
            # san = sanity_report(
            #     model=bst,
            #     X_val=X_test_encoded,
            #     y_val=y_test,
            #     get_attribs_callable=get_attr_xgb,
            #     randomize="labels",
            #     seed=42
            # )
            # val_tab.markdown("### Sanity")
            # val_tab.write(f"Spearman ρ vs original after label randomization: **{san['spearman_rho_vs_original']:.3f}** "
            #               "(should drop toward 0 if explanations depend on learned signal)")


            # get_xgb_explanation(bst, background_data_encoded, X_test_encoded, shap_tab)
            xgb_local_shap_by_truth(bst, X_test_encoded, X_test, y_test, 0.58, shap_tab, background_data_encoded)
            get_lime_explanations_binary(bst, X_test, X_train_encoded, X_test_encoded, y_test, 0.58, lime_tab, title_prefix="Local LIME – XGBoost")
        
            adapted = XGBoosterAdapter(bst, objective="binary:logistic")
            xgb_for_dice = ThresholdedModel(adapted, threshold=0.58)

            continuous_features = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",          
                "AvgMonthlyCharge",
                "OnlineServiceCount",
                "AvgPricePerService",
            ]

            features_to_vary = [
                "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                "PaperlessBilling","MonthlyCharges",
                "Contract_One year","Contract_Two year",
                "PaymentMethod_Credit card (automatic)","PaymentMethod_Electronic check","PaymentMethod_Mailed check",
                "InternetService_Fiber optic","InternetService_No",
            ]

            permitted_range = {"MonthlyCharges": [15, 150]}

            results = get_counterfactual_analysis(
            y_test=y_test,
            X_test=X_test_encoded,          
            X_train=X_train_encoded,
            y_train=y_train,
            model=xgb_for_dice,              
            continuous_features=continuous_features,
            counterfactual_tab=counterfactual_tab,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,       # raw bounds
            scaler=None,                           
            numeric_feature_names_in_scaler_order=None,
            immutable_features=immutable,
            onehot_groups=onehot_groups
        )


    # Display internet dataset's null values in a heatmap
    else:
        selected_anaysis_type = st.sidebar.selectbox(
        options=[
            'Choose Analysis Type', 
            'Numeric Drivers of Churn', 
            'Categorical Drivers of Churn',
            'Service Engagement and Churn',
            'Correlations and Interactions'],
        label='Choose Analysis Type',
        label_visibility='collapsed',
        index=0
        )   

        selected_prediction_model = st.sidebar.selectbox(
        options=[
            'Choose Prediction Model', 
            'Logistic Regression', 
            'Random Forest',
            'XGBoost'],
        label='Choose Prediction Model',
        label_visibility='collapsed',
        index=0
        )

        # Rename churn column into Churn, so it would be the same as in telco data
        internet_data = internet_data.rename(columns={"churn": "Churn"})
        
        # per column
        missing_download_avg_percentage = get_internet_dataset_null_percentage(datasets[selected_dataset]['download_avg'])
        missing_upload_avg_percentage = get_internet_dataset_null_percentage(datasets[selected_dataset]['upload_avg'])
        missing_remaining_contract_percentage = get_internet_dataset_null_percentage(datasets[selected_dataset]['remaining_contract'])
        
        # Add pie
        pie_col1, pie_col2 = st.columns([1, 1])
        pie_col1.plotly_chart(show_balance_between_yes_and_no(internet_data))
        pie_col2.caption(body=get_internet_pie_caption(), unsafe_allow_html=True)

        # Copy the dataset to map Churn values as Yes and No
        dataset_copy = internet_data.copy()
        dataset_copy['Churn'] = dataset_copy['Churn'].map({1: 'Yes', 0: 'No'})

        if selected_anaysis_type == 'Choose Analysis Type' and selected_prediction_model == 'Choose Prediction Model':
            # Add diagrams for missing values
            st.markdown('##### Exploring Missing Data')
            heatmap_col1, heatmap_col2 = st.columns([1, 1])
            figure, axes = get_internet_data_null_heatmap(datasets[selected_dataset])
            heatmap_col1.pyplot(figure)

            fig = get_internet_data_null_hist(datasets[selected_dataset])
            heatmap_col2.plotly_chart(fig)

            st.caption(body=get_internet_data_missing_values_caption(missing_download_avg_percentage, missing_remaining_contract_percentage),
                   unsafe_allow_html=True)
        elif selected_anaysis_type == 'Numeric Drivers of Churn':
            st.subheader(selected_anaysis_type)
            show_internet_numeric_drivers(dataset_copy)

        # -------------- Cleaning internet data -------------------------
        
        # Delete rows with null values in upload_avg and download_avg columns
        internet_data = internet_data.dropna(subset=["upload_avg", "download_avg"])

        # Create a new feature has_contract
        engineered_internet_data = engineer_internet_features(internet_data)
        engineered_internet_data["remaining_contract"] = engineered_internet_data["remaining_contract"].fillna(0)

        if selected_anaysis_type == 'Categorical Drivers of Churn':
            st.subheader(selected_anaysis_type)
            show_internet_categorical_drivers(engineered_internet_data)

        elif selected_anaysis_type == 'Service Engagement and Churn':
            st.subheader(selected_anaysis_type)
            show_internet_engagement_and_churn(engineered_internet_data)

        elif selected_anaysis_type == 'Correlations and Interactions':
            st.subheader(selected_anaysis_type)
            show_internet_correlations_and_interactions(engineered_internet_data)

        X_train, X_test, y_train, y_test = split_internet_data(engineered_internet_data)

        X_train_encoded = encode_internet_data(X_train)
        X_test_encoded = encode_internet_data(X_test)

        scaler, scaled_X_train_features, scaled_X_test_features, num_cols_in_scaler_order = get_scaled_internet_features(X_train_encoded, X_test_encoded)
        background_data_scaled = (
            scaled_X_train_features
                .assign(Churn=y_train.values)
                .groupby('Churn', group_keys=False)
                .apply(lambda x: x.sample(frac=1000/len(scaled_X_train_features), random_state=42))
                .drop(columns='Churn')
            )      

        background_data_encoded = (
            X_train_encoded
                .assign(Churn=y_train.values)
                .groupby('Churn', group_keys=False)
                .apply(lambda x: x.sample(frac=1000/len(scaled_X_train_features), random_state=42))
                .drop(columns='Churn')
            )
        #-------------------------------
        #------------------- Train models -----------------------------
        
        cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        immutable = {"subscription_age", "download_avg", "upload_avg", "total_usage", "usage_tier_Medium", "usage_tier_Heavy"}
        onehot_groups = {
            "total_usage": ["usage_tier_Medium","usage_tier_Heavy"],
            "subscription_age": ["contract_stage_0.5-1y", "contract_stage_1-2y", "contract_stage_>2y", "contract_stage_no_contract"]
        }
        if selected_prediction_model == 'Logistic Regression':
            
            # perform_primary_lg_training(scaled_X_train_features, y_train, scaled_X_test_features, y_test, None)
            # cross_validate_lg_model(X_train_encoded, y_train, cross_validation, None)
            # hyperparameter_tune_lg(X_train_encoded, y_train, cross_validation)
            metrics, tuned_model = retrain_lg_model(scaled_X_train_features, y_train, scaled_X_test_features, y_test, 0.408, 0.01)

            # def group_fairness_table(df, group_col, y_true='y_true', y_proba='y_proba', threshold=0.408, reference=None):
            #     """Return a per-group fairness table and the chosen reference group."""
            #     # pick reference = largest group if not provided
            #     ref = reference or df[group_col].value_counts().idxmax()

            #     def recall_pos(g):
            #         mask = g[y_true] == 1
            #         if mask.sum() == 0:
            #             return np.nan
            #         y_hat = (g[y_proba] >= threshold).astype(int)
            #         return recall_score(g.loc[mask, y_true], y_hat[mask])

            #     agg = (
            #         df.groupby(group_col)
            #           .apply(lambda g: pd.Series({
            #               'n': len(g),
            #               'observed_churn_rate': g[y_true].mean(),
            #               'mean_pred_proba': g[y_proba].mean(),
            #               f'prediction_rate@{threshold:.3f}': (g[y_proba] >= threshold).mean(),
            #               'recall_pos': recall_pos(g),
            #           }))
            #           .reset_index()
            #     )

            #     ref_mean = agg.loc[agg[group_col] == ref, 'mean_pred_proba'].values[0]
            #     agg[f'Δ_vs_{ref}'] = (agg['mean_pred_proba'] - ref_mean).abs()
            #     agg['stat_parity_ok(≤0.05)'] = agg[f'Δ_vs_{ref}'] <= 0.05
            #     return agg.sort_values('n', ascending=False), ref

            # group_cols = ["contract_stage", "bill_bucket", "usage_tier", "fail_count_bucket",]  
            # threshold = 0.408
            # X_test_original_groups = X_test[group_cols].copy()
            # if hasattr(y_test, "dtype") and y_test.dtype == object:
            #     y_true = y_test.map({"No": 0, "Yes": 1}).astype(int)
            # elif isinstance(y_test, pd.DataFrame) and "Churn" in y_test.columns:
            #     y_true = y_test["Churn"].map({"No": 0, "Yes": 1}).astype(int)
            # else:
            #     y_true = y_test.astype(int)  # already 0/1

            # if hasattr(tuned_model, "predict_proba"):           # single estimator
            #     models = {"logreg_final": tuned_model}
            # elif isinstance(tuned_model, (list, tuple)):        # list/tuple of estimators
            #     models = {f"model_{i}": m for i, m in enumerate(tuned_model)}
            # elif isinstance(tuned_model, dict):                  # already a dict
            #     models = tuned_model
            # else:
            #     raise TypeError("tuned_model must be an estimator, list/tuple of estimators, or dict.")

            # X_for_pred = scaled_X_test_features

            # results = {}
            # for name, model in models.items():
            #     # get probabilities for the positive class
            #     y_proba = model.predict_proba(X_for_pred)[:, 1]

            #     # build evaluation frame aligned by index
            #     df_eval = X_test_original_groups.copy()
            #     df_eval["y_true"] = pd.Series(y_true.values, index=df_eval.index)
            #     df_eval["y_proba"] = pd.Series(y_proba, index=df_eval.index)

            # # per-group tables for this model
            #     model_tables = {}
            #     for gcol in group_cols:
            #         sub = df_eval[[gcol, "y_true", "y_proba"]].dropna()
            #         table, ref = group_fairness_table(
            #             sub, group_col=gcol, y_true="y_true", y_proba="y_proba", threshold=threshold
            #         )
            #         model_tables[gcol] = (table, ref)
            #     results[name] = model_tables

            # st.title("Fairness Evaluation — Group-Based Metrics")
            # st.write("Global threshold for recall comparisons (equal opportunity): ", threshold)

            # for model_name, tables in results.items():
            #     with st.expander(f"Model: {model_name}", expanded=False):
            #         for gcol, (table, ref) in tables.items():
            #             st.subheader(f"Group column: {gcol} (ref: {ref})")
            #             st.dataframe(table, use_container_width=True)

            #             # CSV download
            #             csv_buf = StringIO()
            #             table.to_csv(csv_buf, index=False)
            #             st.download_button(
            #                 label=f"Download CSV — {model_name} · {gcol}",
            #                 data=csv_buf.getvalue(),
            #                 file_name=f"fairness_{model_name}_{gcol}.csv",
            #                 mime="text/csv"
            #             )

            st.subheader("Logistic Regression Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_lr_metrics(metrics_tab, metrics)
            get_internet_lr_metrics_caption(column2)

            # Explainability models
            attribs = collect_attributions_for_tests(
                lr_model=tuned_model,
                X_train_scaled_bg=scaled_X_train_features.sample(200, random_state=0),
                X_val_scaled=scaled_X_test_features,

                rf_model=None,
                xgb_clf=None,
                X_train_enc=None,
                X_val_enc=None,

                xgb_booster=None,
                use_streamlit=True
            )
            # Explainability models
            attribs = collect_attributions_for_tests(
                lr_model=tuned_model,
                X_train_scaled_bg=scaled_X_train_features.sample(200, random_state=0),
                X_val_scaled=scaled_X_test_features,

                rf_model=None,
                xgb_clf=None,
                X_train_enc=None,
                X_val_enc=None,

                xgb_booster=None,
                use_streamlit=True
            )
            A_lr = attribs["lr"]

            # val_tab, shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["Validation","SHAP", "LIME", "Counterfactuals"])
            shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["SHAP", "LIME", "Counterfactuals"])


            # Faithfulness
            
            # rep = faithfulness_report(
            #     model=tuned_model,
            #     X_val=scaled_X_test_features,
            #     y_val=y_test,
            #     A=A_lr,
            #     feature_names=scaled_X_test_features.columns.tolist(),
            #     n_steps=10,
            #     seed=42,
            #     use_proba=True
            # )
            # val_tab.markdown("### Faithfulness")
            # val_tab.write(f"Spearman ρ (rank vs. single-feature impact): **{rep['rho']:.3f}**")
            # val_tab.dataframe(rep["curve"])

            # x_row = scaled_X_test_features.iloc[0]
            # a_row = pd.Series(A_lr[0], index=scaled_X_test_features.columns)

            # rep = local_faithfulness_report(
            #     model=tuned_model,               # RF / LR / XGB model object
            #     x_row=x_row,
            #     a_row=a_row,
            #     background=background_data_scaled,
            #     n_draws=64,
            #     n_steps=10,
            #     replace_with="mean"              # or "draw"
            # )
            # val_tab.markdown("### Local Faithfulness (this customer)")
            # val_tab.write(f"Spearman ρ (local): **{rep['rho_local']:.3f}**")
            # val_tab.line_chart(pd.DataFrame({"p(k)": rep["deletion_curve"]["preds"]}, index=rep["deletion_curve"]["k"]))
            # val_tab.caption(
            #     f"Deletion curve AUC: {rep['deletion_curve']['auc']:.4f} (lower is better). "
            #     f"Baseline p0={rep['deletion_curve']['p0']:.3f}. "
            #     f"{'Flip at k=' + str(rep['flip_k']) if rep['flip_k'] is not None else ''}"
            # )
            # val_tab.dataframe(rep["single_feature_impacts"].head(12).to_frame("local_impact"))

            # X_train_for_lime = scaled_X_train_features  # use X_train_encoded for RF/XGB branches
            # X_test_for_lime = scaled_X_test_features
            # random_state = 42
            # # choose a row to test (must be same feature space as the explainer/model)
            # row_id = X_test_for_lime.index[0]           # e.g., scaled_X_test_features or X_test_encoded
            # x_row = X_test_for_lime.loc[row_id]

            # lime_explainer = LimeTabularExplainer(
            #     training_data=X_train_for_lime.to_numpy(),
            #     feature_names=X_train_for_lime.columns.tolist(),
            #     class_names=[str(c) for c in getattr(tuned_model, "classes_", [0, 1])],  # <-- FIXED
            #     mode="classification",
            #     discretize_continuous=True,
            #     sample_around_instance=True,
            #     random_state=random_state,
            # )

            # lt = lime_local_deletion_test(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     k_max=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Deletion curve AUC: **{lt['auc']:.4f}** (lower is better). "
            #     f"Baseline p0={lt['p0']:.3f}. "
            #     f"{'Flip at k=' + str(lt['flip_k']) if lt['flip_k'] is not None else 'No flip'}."
            # )

            # corr = lime_local_rank_vs_impact(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Spearman ρ (LIME rank vs single-feature impact): **{corr['rho']:.3f}** "
            #     f"(p={corr['pval']:.3g})."
            # )

            # # Stability
            # get_attr_lr = lambda X: collect_attributions_for_tests(
            #     lr_model=tuned_model,
            #     X_train_scaled_bg=scaled_X_train_features.sample(200, random_state=0),
            #     X_val_scaled=X,
            #     use_streamlit=False
            # )["lr"]
            # stab = stability_report(get_attr_lr, scaled_X_test_features, n_boot=8, sample_frac=0.8, seed=42)
            # val_tab.markdown("### Stability")
            # val_tab.write(f"Kendall’s τ (mean ± sd): **{stab['kendall_tau_mean']:.3f} ± {stab['kendall_tau_std']:.3f}**")

            # X_train_for_lime = scaled_X_train_features  # use X_train_encoded for RF/XGB branches
            # X_test_for_lime = scaled_X_test_features
            # random_state = 42
            # # choose a row to test (must be same feature space as the explainer/model)
            # row_id = X_test_for_lime.index[0]           # e.g., scaled_X_test_features or X_test_encoded
            # x_row = X_test_for_lime.loc[row_id]

            # lime_explainer = LimeTabularExplainer(
            #     training_data=X_train_for_lime.to_numpy(),
            #     feature_names=X_train_for_lime.columns.tolist(),
            #     class_names=[str(c) for c in getattr(tuned_model, "classes_", [0, 1])],  # <-- FIXED
            #     mode="classification",
            #     discretize_continuous=True,
            #     sample_around_instance=True,
            #     random_state=random_state,
            # )

            # lt = lime_local_deletion_test(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     k_max=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Deletion curve AUC: **{lt['auc']:.4f}** (lower is better). "
            #     f"Baseline p0={lt['p0']:.3f}. "
            #     f"{'Flip at k=' + str(lt['flip_k']) if lt['flip_k'] is not None else 'No flip'}."
            # )

            # corr = lime_local_rank_vs_impact(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Spearman ρ (LIME rank vs single-feature impact): **{corr['rho']:.3f}** "
            #     f"(p={corr['pval']:.3g})."
            # )

            # Stability
            # get_attr_lr = lambda X: collect_attributions_for_tests(
            #     lr_model=tuned_model,
            #     X_train_scaled_bg=scaled_X_train_features.sample(200, random_state=0),
            #     X_val_scaled=X,
            #     use_streamlit=False
            # )["lr"]
            # stab = stability_report(get_attr_lr, scaled_X_test_features, n_boot=20, sample_frac=0.8, seed=42)
            # val_tab.markdown("### Stability")
            # val_tab.write(f"Kendall’s τ (mean ± sd): **{stab['kendall_tau_mean']:.3f} ± {stab['kendall_tau_std']:.3f}**")


            # --- 1) Build a hashable fingerprint for LR ---
            # --- Hashable fingerprint for LR (for cache invalidation) ---
            # def _lr_model_key(m: LogisticRegression) -> str:
            #     coef = getattr(m, "coef_", None)
            #     intercept = getattr(m, "intercept_", None)
            #     n_iter = getattr(m, "n_iter_", None)
            #     parts = []
            #     if coef is not None:      parts.append(coef.ravel().tobytes())
            #     if intercept is not None: parts.append(np.atleast_1d(intercept).ravel().tobytes())
            #     if n_iter is not None:    parts.append(np.atleast_1d(n_iter).ravel().tobytes())
            #     parts.append(str(getattr(m, "C", None)).encode())
            #     parts.append(str(getattr(m, "penalty", None)).encode())
            #     parts.append(str(getattr(m, "solver", None)).encode())
            #     return str(abs(hash(b"||".join(parts))))

            # @st.cache_data(show_spinner=False)
            # def _lr_full_attr(
            #     _model: LogisticRegression,  # ignored by Streamlit hasher (leading underscore)
            #     model_key: str,
            #     X_train_bg: pd.DataFrame,
            #     X_val_scaled: pd.DataFrame,
            # ) -> pd.DataFrame:
            #     A = collect_attributions_for_tests(
            #         lr_model=_model,
            #         X_train_scaled_bg=X_train_bg,
            #         X_val_scaled=X_val_scaled,
            #         rf_model=None, xgb_clf=None,
            #         X_train_enc=None, X_val_enc=None,
            #         xgb_booster=None,
            #         use_streamlit=False
            #     )["lr"]
            #     return pd.DataFrame(A, index=X_val_scaled.index, columns=X_val_scaled.columns)

            # # --- use it (always pass model_key!) ---
            # lr_key   = _lr_model_key(tuned_model)
            # bg_sample = scaled_X_train_features.sample(min(200, len(scaled_X_train_features)), random_state=0)

            # A_lr_full = _lr_full_attr(
            #     _model=tuned_model,
            #     model_key=lr_key,
            #     X_train_bg=bg_sample,
            #     X_val_scaled=scaled_X_test_features,
            # )

            # def get_attr_lr_cached(X_batch: pd.DataFrame) -> np.ndarray:
            #     # try reuse from full cache
            #     try:
            #         return A_lr_full.loc[X_batch.index].to_numpy()
            #     except KeyError:
            #         # small recompute for perturbed rows not in A_lr_full
            #         A_tmp = collect_attributions_for_tests(
            #             lr_model=tuned_model,
            #             X_train_scaled_bg=bg_sample,
            #             X_val_scaled=X_batch,
            #             use_streamlit=False
            #         )["lr"]
            #         return A_tmp

            # with val_tab:
            #     st.markdown("### Local Stability (Logistic Regression)")
            #     lr_row_id = st.selectbox("Pick a row (LR):", scaled_X_test_features.index.tolist(), index=0, key="lr_row_pick")
            #     n_pert = st.slider("Perturbations", 10, 100, 30, key="lr_npert")
            #     noise = st.number_input("Numeric noise (σ)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, key="lr_noise")
            #     flip = st.number_input("Flip prob (binary/one-hot)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="lr_flip")

            #     if st.button("Run local stability (LR)"):
            #         # In the scaled matrix, nearly everything is numeric; treat one-hots as binary [0/1].
            #         numeric_cols = scaled_X_test_features.columns.tolist()
            #         # If you can enumerate true one-hots, pass them here; otherwise an empty dict is safe.
            #         report = local_stability_report_generic(
            #             get_attr_for_batch=get_attr_lr_cached,
            #             X_matrix=scaled_X_test_features,
            #             row_id=lr_row_id,
            #             numeric_cols=numeric_cols,
            #             binary_cols=[],                    # (optional) add pure binary cols here if you keep them separate
            #             onehot_groups=onehot_groups,       # if columns exist in scaled matrix with same names
            #             n_perturb=n_pert,
            #             noise_scale=noise,
            #             flip_prob=flip,
            #             k_list=(5,10)
            #         )
            #         st.write(f"Kendall’s τ (mean): **{report['kendall_tau_mean']:.3f}**")
            #         st.write(f"Spearman ρ (mean): **{report['spearman_rho_mean']:.3f}**")
            #         st.write(f"Avg std(|attrib|): **{report['mean_attr_std']:.4f}**")
            #         st.write({f"Top-{k} overlap": f"{report['topk_overlap_mean'][k]:.2f}" for k in (5,10)})
            #         st.dataframe(report["attr_std_by_feature"].head(12).to_frame("std").style.format({"std": "{:.4f}"}))


            # Sanity (label randomization)
            # san = sanity_report(
            # model=tuned_model,
            # X_val=scaled_X_test_features,
            # y_val=y_test,
            # get_attribs_callable=get_attr_lr_cached,  # <-- new name
            # randomize="labels",
            # seed=42
            # )
            # val_tab.markdown("### Sanity")
            # val_tab.write(f"Spearman ρ vs original after label randomization: **{san['spearman_rho_vs_original']:.3f}** "
            #               "(should drop toward 0 if explanations depend on learned signal)")


            # get_lr_explanation(tuned_model, background_data_scaled, scaled_X_test_features, shap_tab)
            lr_local_shap_by_truth(
                lr_model=tuned_model,
                background_data=background_data_scaled,
                X_test_scaled=scaled_X_test_features,
                X_test=X_test,
                y_test=y_test,             # must be aligned with X_test_scaled index
                shap_tab=shap_tab,
                threshold=0.408,            # your operating threshold
                top_display=12
                )
            get_lime_explanations_binary(tuned_model, X_test, scaled_X_train_features, scaled_X_test_features, y_test, 0.408, lime_tab, title_prefix="Local LIME – Logistic Regression")      


            lr_for_dice = ThresholdedModel(tuned_model, threshold=0.408)
            
            continuous_features = [
                "subscription_age",
                "bill_avg",
                "service_failure_count",          
                "download_avg",
                "upload_avg",
                "total_usage",
                "services_count",
                "download_over_limit"
            ]

            features_to_vary = [
                "is_tv_subscriber", "is_movie_package_subscriber", 
                "bill_avg",
                "download_over_limit",
                "contract_stage_0.5-1y", "contract_stage_1-2y", "contract_stage_>2y", "contract_stage_no_contract"
            ]

            permitted_range = {"bill_avg": [15, 150]}

            results = get_counterfactual_analysis(
                y_test=y_test,
                X_test=scaled_X_test_features,
                X_train=scaled_X_train_features,
                y_train=y_train,
                model=lr_for_dice,                         # must expose predict_proba
                continuous_features=continuous_features,
                counterfactual_tab=counterfactual_tab,
                outcome_name="Churn",
                total_CFs=6,
                features_to_vary=features_to_vary,                    # or "all"
                permitted_range={"MonthlyCharges":[15,150]},
                scaler=scaler,
                numeric_feature_names_in_scaler_order=num_cols_in_scaler_order,
                immutable_features=immutable,
                onehot_groups=onehot_groups,
            )

        elif selected_prediction_model == 'Random Forest':
            
            # perform_primary_rf_training(X_train_encoded, y_train, X_test_encoded, y_test, None)
            # cross_validate_rf_model(X_train_encoded, y_train, cross_validation, None)
            # hyperparameter_tune_rf(X_train_encoded, y_train, cross_validation)
            metrics, tuned_model = retrain_rf_model(X_train_encoded, y_train, X_test_encoded, y_test, 0.426, 609, 10, 3, 0.9, 28)

            # def group_fairness_table(df, group_col, y_true='y_true', y_proba='y_proba', threshold=0.426, reference=None):
            #     """Return a per-group fairness table and the chosen reference group."""
            #     # pick reference = largest group if not provided
            #     ref = reference or df[group_col].value_counts().idxmax()

            #     def recall_pos(g):
            #         mask = g[y_true] == 1
            #         if mask.sum() == 0:
            #             return np.nan
            #         y_hat = (g[y_proba] >= threshold).astype(int)
            #         return recall_score(g.loc[mask, y_true], y_hat[mask])

            #     agg = (
            #         df.groupby(group_col)
            #           .apply(lambda g: pd.Series({
            #               'n': len(g),
            #               'observed_churn_rate': g[y_true].mean(),
            #               'mean_pred_proba': g[y_proba].mean(),
            #               f'prediction_rate@{threshold:.3f}': (g[y_proba] >= threshold).mean(),
            #               'recall_pos': recall_pos(g),
            #           }))
            #           .reset_index()
            #     )

            #     ref_mean = agg.loc[agg[group_col] == ref, 'mean_pred_proba'].values[0]
            #     agg[f'Δ_vs_{ref}'] = (agg['mean_pred_proba'] - ref_mean).abs()
            #     agg['stat_parity_ok(≤0.05)'] = agg[f'Δ_vs_{ref}'] <= 0.05
            #     return agg.sort_values('n', ascending=False), ref

            # group_cols = ["contract_stage", "bill_bucket", "usage_tier", "fail_count_bucket",]  
            # threshold = 0.426
            # X_test_original_groups = X_test[group_cols].copy()
            # if hasattr(y_test, "dtype") and y_test.dtype == object:
            #     y_true = y_test.map({"No": 0, "Yes": 1}).astype(int)
            # elif isinstance(y_test, pd.DataFrame) and "Churn" in y_test.columns:
            #     y_true = y_test["Churn"].map({"No": 0, "Yes": 1}).astype(int)
            # else:
            #     y_true = y_test.astype(int)  # already 0/1

            # if hasattr(tuned_model, "predict_proba"):           # single estimator
            #     models = {"logreg_final": tuned_model}
            # elif isinstance(tuned_model, (list, tuple)):        # list/tuple of estimators
            #     models = {f"model_{i}": m for i, m in enumerate(tuned_model)}
            # elif isinstance(tuned_model, dict):                  # already a dict
            #     models = tuned_model
            # else:
            #     raise TypeError("tuned_model must be an estimator, list/tuple of estimators, or dict.")

            # X_for_pred = X_test_encoded

            # results = {}
            # for name, model in models.items():
            #     # get probabilities for the positive class
            #     y_proba = model.predict_proba(X_for_pred)[:, 1]

            #     # build evaluation frame aligned by index
            #     df_eval = X_test_original_groups.copy()
            #     df_eval["y_true"] = pd.Series(y_true.values, index=df_eval.index)
            #     df_eval["y_proba"] = pd.Series(y_proba, index=df_eval.index)

            # # per-group tables for this model
            #     model_tables = {}
            #     for gcol in group_cols:
            #         sub = df_eval[[gcol, "y_true", "y_proba"]].dropna()
            #         table, ref = group_fairness_table(
            #             sub, group_col=gcol, y_true="y_true", y_proba="y_proba", threshold=threshold
            #         )
            #         model_tables[gcol] = (table, ref)
            #     results[name] = model_tables

            # st.title("Fairness Evaluation — Group-Based Metrics")
            # st.write("Global threshold for recall comparisons (equal opportunity): ", threshold)
            # for model_name, tables in results.items():
            #     with st.expander(f"Model: {model_name}", expanded=False):
            #         for gcol, (table, ref) in tables.items():
            #             st.subheader(f"Group column: {gcol} (ref: {ref})")
            #             st.dataframe(table, use_container_width=True)

            #             # CSV download
            #             csv_buf = StringIO()
            #             table.to_csv(csv_buf, index=False)
            #             st.download_button(
            #                 label=f"Download CSV — {model_name} · {gcol}",
            #                 data=csv_buf.getvalue(),
            #                 file_name=f"fairness_{model_name}_{gcol}.csv",
            #                 mime="text/csv"
            #             )


            st.subheader("Random Forest Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_rf_metrics(metrics_tab, metrics)
            get_internet_rf_metrics_caption(column2)

            # Explainability models
            attribs = collect_attributions_for_tests(
                lr_model=None,
                X_train_scaled_bg=None,
                X_val_scaled=None,

                rf_model=tuned_model,
                xgb_clf=None,
                X_train_enc=X_train_encoded,
                X_val_enc=X_test_encoded,

                xgb_booster=None,
                use_streamlit=True
            )
            A_rf = attribs["rf"] 

            val_tab, shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["Validation","SHAP", "LIME", "Counterfactuals"])
            shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["SHAP", "LIME", "Counterfactuals"])

            # Faithfulness
            
            # rep = faithfulness_report(
            #     model=tuned_model,
            #     X_val=X_test_encoded,
            #     y_val=y_test,
            #     A=A_rf,
            #     feature_names=X_test_encoded.columns.tolist(),
            #     n_steps=10,
            #     seed=42,
            #     use_proba=True
            # )
            # val_tab.markdown("### Faithfulness")
            # val_tab.write(f"Spearman ρ (rank vs. single-feature impact): **{rep['rho']:.3f}**")
            # val_tab.dataframe(rep["curve"])

            # x_row = X_test_encoded.iloc[0]
            # a_row = pd.Series(A_rf[0], index=X_test_encoded.columns)

            # rep = local_faithfulness_report(
            #     model=tuned_model,               # RF / LR / XGB model object
            #     x_row=x_row,
            #     a_row=a_row,
            #     background=background_data_encoded,
            #     n_draws=64,
            #     n_steps=10,
            #     replace_with="mean"              # or "draw"
            # )
            # val_tab.markdown("### Local Faithfulness (this customer)")
            # val_tab.write(f"Spearman ρ (local): **{rep['rho_local']:.3f}**")
            # val_tab.line_chart(pd.DataFrame({"p(k)": rep["deletion_curve"]["preds"]}, index=rep["deletion_curve"]["k"]))
            # val_tab.caption(
            #     f"Deletion curve AUC: {rep['deletion_curve']['auc']:.4f} (lower is better). "
            #     f"Baseline p0={rep['deletion_curve']['p0']:.3f}. "
            #     f"{'Flip at k=' + str(rep['flip_k']) if rep['flip_k'] is not None else ''}"
            # )
            # val_tab.dataframe(rep["single_feature_impacts"].head(12).to_frame("local_impact"))

            # X_train_for_lime = X_train_encoded  # use X_train_encoded for RF/XGB branches
            # X_test_for_lime = X_test_encoded
            # random_state = 42
            # # choose a row to test (must be same feature space as the explainer/model)
            # row_id = X_test_for_lime.index[0]           # e.g., scaled_X_test_features or X_test_encoded
            # x_row = X_test_for_lime.loc[row_id]

            # lime_explainer = LimeTabularExplainer(
            #     training_data=X_train_for_lime.to_numpy(),
            #     feature_names=X_train_for_lime.columns.tolist(),
            #     class_names=[str(c) for c in getattr(tuned_model, "classes_", [0, 1])],  # <-- FIXED
            #     mode="classification",
            #     discretize_continuous=True,
            #     sample_around_instance=True,
            #     random_state=random_state,
            # )

            # lt = lime_local_deletion_test(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     k_max=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Deletion curve AUC: **{lt['auc']:.4f}** (lower is better). "
            #     f"Baseline p0={lt['p0']:.3f}. "
            #     f"{'Flip at k=' + str(lt['flip_k']) if lt['flip_k'] is not None else 'No flip'}."
            # )

            # corr = lime_local_rank_vs_impact(
            #     model=tuned_model,
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     num_features=10,
            #     mask_strategy="mean"
            # )
            # val_tab.write(
            #     f"Spearman ρ (LIME rank vs single-feature impact): **{corr['rho']:.3f}** "
            #     f"(p={corr['pval']:.3g})."
            # )

            # Stability
            # Ensure indices are simple and unique
            X_test_encoded_copy = X_test_encoded.copy()
            X_test_encoded_copy.index = pd.RangeIndex(len(X_test_encoded))

            # # 1) Precompute RF attributions once on the FULL test set
            # A_rf_full = pd.DataFrame(
            #     A_rf,  # from collect_attributions_for_tests on FULL X_test_encoded
            #     index=X_test_encoded_copy.index,
            #     columns=X_test_encoded_copy.columns
            # )

            # --- 1) Build a hashable fingerprint for LR ---
            # --- Hashable fingerprint for LR (for cache invalidation) ---
            # def _rf_model_key(m: RandomForestClassifier) -> str:
            #     """
            #     Build a hashable fingerprint for a fitted RF.
            #     Includes: key hyperparams, n_estimators, feature_importances_,
            #     oob_score_ (if present). This is sufficient to invalidate cache
            #     when the trained forest changes.
            #     """
            #     parts = []
            #     # core hyperparams that change the fit
            #     hp = m.get_params(deep=False)
            #     key_params = (
            #         hp.get("n_estimators"), hp.get("criterion"),
            #         hp.get("max_depth"), hp.get("max_features"),
            #         hp.get("min_samples_split"), hp.get("min_samples_leaf"),
            #         hp.get("min_weight_fraction_leaf"), hp.get("max_leaf_nodes"),
            #         hp.get("bootstrap"), hp.get("class_weight"),
            #         hp.get("ccp_alpha"), hp.get("random_state")
            #     )
            #     parts.append(str(key_params).encode())

            #     # learned quantities
            #     if hasattr(m, "n_estimators"):
            #         parts.append(str(m.n_estimators).encode())
            #     if hasattr(m, "feature_importances_"):
            #         parts.append(np.asarray(m.feature_importances_, dtype=np.float64).tobytes())
            #     if hasattr(m, "oob_score_"):
            #         parts.append(str(m.oob_score_).encode())

            #     return str(abs(hash(b"||".join(parts))))

            # # -------- Cache full-test RF attributions (ignore unhashable model) --------
            # @st.cache_data(show_spinner=False)
            # def _rf_full_attr(
            #     _model: RandomForestClassifier,  # leading underscore => Streamlit won't hash it
            #     model_key: str,                  # <- hashable fingerprint to invalidate when RF changes
            #     X_train_enc: pd.DataFrame,       # needed by your collector
            #     X_val_enc: pd.DataFrame
            # ) -> pd.DataFrame:
            #     A = collect_attributions_for_tests(
            #         lr_model=None,
            #         X_train_scaled_bg=None, X_val_scaled=None,
            #         rf_model=_model,
            #         xgb_clf=None,
            #         X_train_enc=X_train_enc,
            #         X_val_enc=X_val_enc,
            #         xgb_booster=None,
            #         use_streamlit=False
            #     )["rf"]  # -> (n_val, n_features)
            #     return pd.DataFrame(A, index=X_val_enc.index, columns=X_val_enc.columns)

            # # -------- Build cache + getter in your RF branch --------
            # rf_key = _rf_model_key(tuned_model)
            # A_rf_full = _rf_full_attr(
            #     _model=tuned_model,
            #     model_key=rf_key,
            #     X_train_enc=X_train_encoded,
            #     X_val_enc=X_test_encoded,
            # )

            # def get_attr_rf_cached(X_batch: pd.DataFrame) -> np.ndarray:
            #     # fast path: reuse precomputed rows
            #     try:
            #         return A_rf_full.loc[X_batch.index].to_numpy()
            #     except KeyError:
            #         # small-batch recompute for perturbed rows not in the cache
            #         A_tmp = collect_attributions_for_tests(
            #             rf_model=tuned_model,
            #             X_train_enc=X_train_encoded,
            #             X_val_enc=X_batch,
            #             use_streamlit=False
            #         )["rf"]
            #         return A_tmp

            # # -------- Local stability UI (RF) --------
            # with val_tab:
            #     st.markdown("### Local Stability (Random Forest)")
            #     rf_row_id = st.selectbox("Pick a row (RF):", X_test_encoded.index.tolist(), index=0, key="rf_row_pick")
            #     n_pert = st.slider("Perturbations", 10, 100, 30, key="rf_npert")
            #     noise = st.number_input("Numeric noise (σ)", 0.0, 1.0, 0.02, 0.01, key="rf_noise")
            #     flip = st.number_input("Flip prob (binary/one-hot)", 0.0, 1.0, 0.05, 0.01, key="rf_flip")

            #     # heuristics: treat uint8 one-hots as binary; the rest as numeric
            #     binary_cols_rf  = [c for c in X_test_encoded.columns
            #                        if set(pd.unique(X_test_encoded[c])).issubset({0, 1})]
            #     numeric_cols_rf = [c for c in X_test_encoded.columns if c not in binary_cols_rf]

            #     if st.button("Run local stability (RF)"):
            #         report = local_stability_report_generic(
            #             get_attr_for_batch=get_attr_rf_cached,
            #             X_matrix=X_test_encoded,
            #             row_id=rf_row_id,
            #             numeric_cols=numeric_cols_rf,
            #             binary_cols=binary_cols_rf,
            #             onehot_groups=onehot_groups,  # your existing dict
            #             n_perturb=n_pert,
            #             noise_scale=noise,
            #             flip_prob=flip,
            #             k_list=(5, 10)
            #         )
            #         st.write(f"Kendall’s τ (mean): **{report['kendall_tau_mean']:.3f}**")
            #         st.write(f"Spearman ρ (mean): **{report['spearman_rho_mean']:.3f}**")
            #         st.write(f"Avg std(|attrib|): **{report['mean_attr_std']:.4f}**")
            #         st.write({f"Top-{k} overlap": f"{report['topk_overlap_mean'][k]:.2f}" for k in (5, 10)})
            #         st.dataframe(report["attr_std_by_feature"].head(12).to_frame("std").style.format({"std": "{:.4f}"}))

            # Sanity checks (optional but helpful)
            # assert A_rf_full.shape[0] == len(X_test_encoded_copy), "Mismatch: A_rf_full rows vs X_test_encoded"
            # assert (A_rf_full.columns == X_test_encoded_copy.columns).all(), "Mismatch: columns differ"

            # Build a position map for robust slicing even if a resample resets index
            # pos_map = pd.Series(np.arange(len(X_test_encoded_copy)), index=X_test_encoded_copy.index)

            # def get_attr_rf_cached(X_subset: pd.DataFrame):
            #     # Primary: index-aligned slice (fast path)
            #     try:
            #         return A_rf_full.loc[X_subset.index].to_numpy()
            #     except KeyError:
            #         # Fallback: map indices to positions (handles reset_index in resample)
            #         idx = pos_map.reindex(X_subset.index)
            #         if idx.isna().any():
            #             # As a last resort, assume positional slice (only works if X_subset
            #             # is taken by .iloc from the same matrix)
            #             return A_rf_full.to_numpy()[np.arange(len(X_subset)), :]
            #         return A_rf_full.to_numpy()[idx.to_numpy().astype(int), :]

            # Now run stability (cheap, no SHAP recompute)
            # stab = stability_report(get_attr_rf_cached, X_test_encoded_copy, n_boot=20, sample_frac=0.8, seed=42)
            # val_tab.markdown("### Stability")
            # val_tab.write(f"Kendall’s τ (mean ± sd): **{stab['kendall_tau_mean']:.3f} ± {stab['kendall_tau_std']:.3f}**")


            # # Sanity (label randomization)
            # def _to_hashable_tuple(idx: pd.Index) -> tuple:
            #     vals = idx.tolist()
            #     try:
            #         tuple(vals)
            #         return tuple(vals)
            #     except TypeError:
            #         return tuple(map(str, vals))

            # @st.cache_data(show_spinner=False)
            # def _rf_attr_cache(A_rf_array: np.ndarray,
            #                    index_values: tuple,
            #                    column_values: tuple) -> tuple[pd.DataFrame, pd.Series]:
            #     """
            #     Materialize RF attributions into a DataFrame with original index/columns,
            #     and build a position map for robust slicing.
            #     """
            #     x_index = pd.Index(list(index_values))
            #     x_columns = pd.Index(list(column_values))
            #     A_rf_full_df = pd.DataFrame(A_rf_array, index=x_index, columns=x_columns).astype("float64")

            #     # safety checks
            #     if A_rf_full_df.shape[0] != len(x_index):
            #         raise ValueError("A_rf_full rows != X_test_encoded rows.")
            #     if not A_rf_full_df.columns.equals(x_columns):
            #         raise ValueError("A_rf_full columns differ from X_test_encoded columns.")
            #     if not x_index.is_unique:
            #         st.warning("X_test_encoded has non-unique index; positional fallback may be used.")

            #     pos_map = pd.Series(np.arange(len(x_index)), index=x_index)
            #     return A_rf_full_df, pos_map

            # def make_get_attr_rf_cached(A_rf_full_df: pd.DataFrame, pos_map: pd.Series):
                # def _getter(X_subset: pd.DataFrame) -> np.ndarray:
                #     # fast path: identical labels & order
                #     if len(X_subset.index) and X_subset.index.equals(A_rf_full_df.index):
                #         return A_rf_full_df.to_numpy()

                #     # align by label (preserves order of X_subset)
                #     if X_subset.index.isin(A_rf_full_df.index).all():
                #         return A_rf_full_df.reindex(X_subset.index).to_numpy()

                #     # fallback: position map (handles reset_index/resamples)
                #     idx = pos_map.reindex(X_subset.index)
                #     if not idx.isna().any():
                #         return A_rf_full_df.to_numpy()[idx.to_numpy().astype(int), :]

                #     # last resort: pure positional (warn)
                #     st.warning("Attribution slice fell back to pure positional matching; verify index handling.", icon="⚠️")
                #     n = len(X_subset)
                #     if n > len(A_rf_full_df):
                #         raise ValueError("Subset longer than reference — cannot slice positionally.")
                #     return A_rf_full_df.to_numpy()[:n, :]
                # return _getter

            # ---- RF sanity: build in strict order, keep everything inside the RF branch ----
            # 1) get per-row attributions (array) once for FULL X_test_encoded
            # attribs = collect_attributions_for_tests(
            #     rf_model=tuned_model,
            #     X_train_enc=X_train_encoded,
            #     X_val_enc=X_test_encoded,
            #     lr_model=None, xgb_clf=None, X_train_scaled_bg=None, X_val_scaled=None,
            #     xgb_booster=None,
            #     use_streamlit=False
            # )
            # A_rf_array = attribs["rf"]  # shape: (n_test, n_features)

            # # 2) cache the DF + position map using hashable surrogates
            # A_rf_full_df, pos_map = _rf_attr_cache(
            #     A_rf_array,
            #     _to_hashable_tuple(X_test_encoded.index),
            #     _to_hashable_tuple(X_test_encoded.columns),
            # )
            
            # assert A_rf_full_df.shape[0] == len(X_test_encoded_copy), "Mismatch: A_rf_full_df rows vs X_test_encoded"
            # assert A_rf_full_df.columns.equals(X_test_encoded.columns), "Mismatch: columns differ"

            # 3) getter that reuses the cached matrix
            # get_attr_rf_cached = make_get_attr_rf_cached(A_rf_full_df, pos_map)

            # 4) run sanity inside the UI tab (no references to A_rf_full outside)
            # with val_tab:
            #     n = min(1000, len(X_test_encoded))
            #     X_val_small = X_test_encoded.sample(n=n, random_state=42)
            #     y_val_small = y_test.loc[X_val_small.index]
        
            #     san = sanity_report(
            #         model=tuned_model,
            #         X_val=X_val_small,
            #         y_val=y_val_small,
            #         get_attribs_callable=get_attr_rf_cached,  # your cached slice
            #         randomize="labels",
            #         seed=42,
            #     )
            #     st.write(
            #         f"Spearman ρ vs original after label randomization: "
            #         f"**{san['spearman_rho_vs_original']:.3f}**"
            #     )

            # with val_tab:
            #     st.markdown("### Sanity (Random Forest)")

            #     # Just pick a row + seed; no button, no feature selector
            #     rf_row_id = st.selectbox(
            #         "Pick a row for sanity (RF):",
            #         X_test_encoded.index.tolist(), index=0, key="rf_row_sanity_pick"
            #     )
            #     seed = st.number_input("Seed", 0, 10_000_000, 42, key="rf_sanity_seed")
            #     rng = np.random.RandomState(seed)

            #     # ---------- (A) Label randomization sanity (local but trainable) ----------
            #     # use a small stratified batch that *includes* the selected row
            #     X_s, y_s = make_stratified_batch(
            #         X_test_encoded, y_test, include_index=rf_row_id, n_per_class=80, seed=seed
            #     )

            #     y_s_num = _ensure_series(y_s, X_s.index)
                # if y_s_num.dtype == object:
                #     y_s_num = y_s_num.map({"Yes": 1, "No": 0}).astype(int)
                # else:
                #     y_s_num = y_s_num.astype(int)

                # if y_s_num.nunique() < 2:
                #     st.warning(
                #         "Label-randomization skipped: the selected batch has only one class. "
                #         "Pick a different row or increase n_per_class."
                #     )
                # else:
                #     san = sanity_report(
                #         model=tuned_model,                     # RF model
                #         X_val=X_s,
                #         y_val=y_s_num,
                #         get_attribs_callable=lambda X_batch: get_attr_rf_cached(X_batch),
                #         randomize="labels",
                #         seed=seed
                #     )
                #     st.write(
                #         f"Label randomization — Spearman ρ vs. original (local batch): "
                #         f"**{san['spearman_rho_vs_original']:.3f}** (→ closer to 0 is better)"
                #     )



            # get_rf_explanation(tuned_model, background_data_encoded, X_test_encoded, shap_tab)
            rf_local_shap_by_truth(tuned_model, X_test_encoded, X_test, y_test, 0.426, shap_tab, background_data_encoded)
            get_lime_explanations_binary(tuned_model, X_test, X_train_encoded, X_test_encoded, y_test, 0.426, lime_tab, title_prefix="Local LIME – Random Forest")

            rf_for_dice = ThresholdedModel(tuned_model, threshold=0.426)

            continuous_features = [
                "subscription_age",
                "bill_avg",
                "service_failure_count",          
                "download_avg",
                "upload_avg",
                "total_usage",
                "services_count",
                "download_over_limit"
            ]

            features_to_vary = [
                "is_tv_subscriber", "is_movie_package_subscriber", 
                "bill_avg",
                "download_over_limit",
                "contract_stage_0.5-1y", "contract_stage_1-2y", "contract_stage_>2y", "contract_stage_no_contract"
            ]

            permitted_range = {"bill_avg": [15, 150]}

            results = get_counterfactual_analysis(
                y_test=y_test,
                X_test=X_test_encoded,          
                X_train=X_train_encoded,
                y_train=y_train,
                model=rf_for_dice,              
                continuous_features=continuous_features,
                counterfactual_tab=counterfactual_tab,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,       # raw bounds
                scaler=None,                           
                numeric_feature_names_in_scaler_order=None,
                immutable_features=immutable,
                onehot_groups=onehot_groups
            )


        elif selected_prediction_model == "XGBoost":
            # Train
            # perform_primary_xgb_training(X_train_encoded, y_train, X_test_encoded, y_test, False)
            # cross_validate_xgb_model(X_train_encoded, y_train, cross_validation, False)
            # best_stage1, rs_results, _ = xgb_random_search(
            # X_train_encoded, y_train, cross_validation,
            # n_iter=60,                # 60–100 is a good start
            # pos_label=1,
            # optimize_max_f1=True      # keep consistent with your LR threshold tuning
            # )
            # print("\n[Stage 1] Best (random):", best_stage1)

            # # --- Stage 2: Grid refinement around the Stage-1 best
            # best_stage2, gs_results = xgb_grid_refine(
            #     X_train_encoded, y_train, cross_validation,
            #     best_params_stage1=best_stage1,
            #     pos_label=1,
            #     optimize_max_f1=True
            # )
            # print("\n[Stage 2] Best (refined):", best_stage2)
            row, bst, params = retrain_xgb_model(X_train_encoded, y_train, X_test_encoded, y_test,
                                                 0.42, 0.01423, 9, 2.7843, 0.7848, 0.6645, 1.0097, 0.0041)

            # def group_fairness_table(df, group_col, y_true='y_true', y_proba='y_proba', threshold=0.42, reference=None):
            #     ref = reference or df[group_col].value_counts().idxmax()

            #     def recall_pos(g):
            #         mask = g[y_true] == 1
            #         if mask.sum() == 0:
            #             return np.nan
            #         y_hat = (g[y_proba] >= threshold).astype(int)
            #         return recall_score(g.loc[mask, y_true], y_hat[mask])

            #     agg = (
            #         df.groupby(group_col)
            #           .apply(lambda g: pd.Series({
            #               'n': len(g),
            #               'observed_churn_rate': g[y_true].mean(),
            #               'mean_pred_proba': g[y_proba].mean(),
            #               f'prediction_rate@{threshold:.3f}': (g[y_proba] >= threshold).mean(),
            #               'recall_pos': recall_pos(g),
            #           }))
            #           .reset_index()
            #     )
            #     ref_mean = agg.loc[agg[group_col] == ref, 'mean_pred_proba'].values[0]
            #     agg[f'Δ_vs_{ref}'] = (agg['mean_pred_proba'] - ref_mean).abs()
            #     agg['stat_parity_ok(≤0.05)'] = agg[f'Δ_vs_{ref}'] <= 0.05
            #     return agg.sort_values('n', ascending=False), ref

            # # --- config / inputs ---
            # threshold = 0.42
            # group_cols = ["contract_stage", "bill_bucket", "usage_tier", "fail_count_bucket",]

            # # IMPORTANT: use the RAW test DF for grouping, not the encoded/scaled matrix.
            # X_test_original_groups = X_test[group_cols].copy()

            # # y_true must be 0/1
            # if isinstance(y_test, pd.DataFrame) and "Churn" in y_test.columns:
            #     y_true = y_test["Churn"].map({"No": 0, "Yes": 1}).astype(int)
            # elif getattr(y_test, "dtype", None) == object:
            #     y_true = y_test.map({"No": 0, "Yes": 1}).astype(int)
            # else:
            #     y_true = y_test.astype(int)

            # # Choose the feature matrix your Booster was trained on (typically encoded, not scaled unless you scaled during training)
            # X_for_pred = X_test_encoded  # if your Booster was trained on the encoded matrix

            # # --- normalize `tuned_model` to a dict of models ---
            # models = {}
            # # You can add more models here later; for now just the Booster:
            # models["xgb_final"] = bst  # bst returned by retrain_xgb_model(...)

            # def predict_proba_any(model, X):
            #     # scikit-like estimators
            #     if hasattr(model, "predict_proba"):
            #         return model.predict_proba(X)[:, 1]
            #     # XGBoost Booster
            #     if isinstance(model, xgb.Booster):
            #         # Ensure X is numpy/pandas -> DMatrix
            #         dtest = xgb.DMatrix(X)
            #         # For binary:logistic this returns probabilities
            #         return model.predict(dtest)
            #     # decision_function fallback (e.g., linear SVM)
            #     if hasattr(model, "decision_function"):
            #         from scipy.special import expit
            #         return expit(model.decision_function(X))
            #     # last resort: predict hard labels (not ideal)
            #     yhat = model.predict(X)
            #     return np.asarray(yhat).astype(float)

            # # --- compute results ---
            # results = {}
            # for name, model in models.items():
            #     y_proba = predict_proba_any(model, X_for_pred)

            #     df_eval = X_test_original_groups.copy()
            #     df_eval["y_true"]  = pd.Series(y_true.values, index=df_eval.index)
            #     df_eval["y_proba"] = pd.Series(y_proba,       index=df_eval.index)

            #     model_tables = {}
            #     for gcol in group_cols:
            #         sub = df_eval[[gcol, "y_true", "y_proba"]].dropna()
            #         table, ref = group_fairness_table(sub, group_col=gcol, threshold=threshold)
            #         model_tables[gcol] = (table, ref)
            #     results[name] = model_tables

            # # --- Streamlit UI ---
            # st.title("Fairness Evaluation — Group-Based Metrics")
            # st.write("Global threshold for recall comparisons (equal opportunity): ", threshold)
            # for model_name, tables in results.items():
            #     with st.expander(f"Model: {model_name}", expanded=False):
            #         for gcol, (table, ref) in tables.items():
            #             st.subheader(f"Group column: {gcol} (ref: {ref})")
            #             st.dataframe(table, use_container_width=True)
            #             csv_buf = StringIO()
            #             table.to_csv(csv_buf, index=False)
            #             st.download_button(
            #                 label=f"Download CSV — {model_name} · {gcol}",
            #                 data=csv_buf.getvalue(),
            #                 file_name=f"fairness_{model_name}_{gcol}.csv",
            #                 mime="text/csv"
            #             )
            
            st.subheader("XGBoost Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_xgb_metrics(metrics_tab, row)
            get_internet_xgb_metrics_caption(column2)

            # Explainability models
            attribs = collect_attributions_for_tests(
                lr_model=None,
                X_train_scaled_bg=None,
                X_val_scaled=None,

                rf_model=None,
                xgb_clf=None,                           # you have a native Booster in `bst`
                X_train_enc=X_train_encoded,
                X_val_enc=X_test_encoded,

                xgb_booster=bst,                        # native Booster path (pred_contribs)
                use_streamlit=True
            )
            A_xgb_booster = attribs["xgb_booster"]
            
            val_tab, shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["Validation","SHAP", "LIME", "Counterfactuals"])
            shap_tab, lime_tab, counterfactual_tab = explainability_tab.tabs(["SHAP", "LIME", "Counterfactuals"])


            # Faithfulness
            
            # rep = faithfulness_report(
            #     model=XGBoosterAdapter(bst, objective="binary:logistic"),
            #     X_val=X_test_encoded,
            #     y_val=y_test,
            #     A=A_xgb_booster,
            #     feature_names=X_test_encoded.columns.tolist(),
            #     n_steps=10,
            #     seed=42,
            #     use_proba=True
            # )
            # val_tab.markdown("### Faithfulness")
            # val_tab.write(f"Spearman ρ (rank vs. single-feature impact): **{rep['rho']:.3f}**")
            # val_tab.dataframe(rep["curve"])

            # x_row = X_test_encoded.iloc[0]
            # a_row = pd.Series(A_xgb_booster[0], index=X_test_encoded.columns)

            # rep = local_faithfulness_report(
            #     model=bst,               # RF / LR / XGB model object
            #     x_row=x_row,
            #     a_row=a_row,
            #     background=background_data_encoded,
            #     n_draws=64,
            #     n_steps=10,
            #     replace_with="mean"              # or "draw"
            # )
            # val_tab.markdown("### Local Faithfulness (this customer)")
            # val_tab.write(f"Spearman ρ (local): **{rep['rho_local']:.3f}**")
            # val_tab.line_chart(pd.DataFrame({"p(k)": rep["deletion_curve"]["preds"]}, index=rep["deletion_curve"]["k"]))
            # val_tab.caption(
            #     f"Deletion curve AUC: {rep['deletion_curve']['auc']:.4f} (lower is better). "
            #     f"Baseline p0={rep['deletion_curve']['p0']:.3f}. "
            #     f"{'Flip at k=' + str(rep['flip_k']) if rep['flip_k'] is not None else ''}"
            # )
            # val_tab.dataframe(rep["single_feature_impacts"].head(12).to_frame("local_impact"))

            # X_train_for_lime = X_train_encoded  # use X_train_encoded for RF/XGB branches
            # X_test_for_lime = X_test_encoded
            # random_state = 42
            # # choose a row to test (must be same feature space as the explainer/model)
            # row_id = X_test_for_lime.index[0]           # e.g., scaled_X_test_features or X_test_encoded
            # x_row = X_test_for_lime.loc[row_id]
            # tuned_model = bst

            # if hasattr(tuned_model, "predict_proba"):
            #     # sklearn model (LR, RF, XGBClassifier)
            #     predict_fn = tuned_model.predict_proba
            #     class_names = [str(c) for c in getattr(tuned_model, "classes_", [0, 1])]
            # else:
            #     # native Booster (bst)
            #     booster = bst if isinstance(bst, xgb.Booster) else bst.get_booster()
            #     feat_names = X_train_for_lime.columns.tolist()
            #     def predict_fn(X_batch: np.ndarray) -> np.ndarray:
            #         dm = xgb.DMatrix(X_batch, feature_names=feat_names)
            #         p1 = booster.predict(dm).reshape(-1)
            #         return np.column_stack([1.0 - p1, p1])
            #     class_names = ["0", "1"]

            # # ---- Create LIME explainer on TRAIN data (same feature space as model) ----
            # lime_explainer = LimeTabularExplainer(
            #     training_data=X_train_for_lime.to_numpy(),
            #     feature_names=X_train_for_lime.columns.tolist(),
            #     class_names=class_names,
            #     mode="classification",
            #     discretize_continuous=True,
            #     sample_around_instance=True,
            #     random_state=42,
            # )

            # # ---- Run the tests (note: we pass predict_fn, not model) ----
            # lt = lime_xgb_local_deletion_test(
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     predict_fn=predict_fn,
            #     num_features=10,
            #     k_max=10,
            #     mask_strategy="mean",
            # )
            # val_tab.write(
            #     f"Deletion curve AUC: **{lt['auc']:.4f}** (lower is better). "
            #     f"Baseline p0={lt['p0']:.3f}. "
            #     f"{'Flip at k='+str(lt['flip_k']) if lt['flip_k'] is not None else 'No flip'}."
            # )

            # corr = lime_xgb_local_rank_vs_impact(
            #     explainer=lime_explainer,
            #     X_row=x_row,
            #     X_background=X_train_for_lime,
            #     predict_fn=predict_fn,
            #     num_features=10,
            #     mask_strategy="mean",
            # )
            # val_tab.write(
            #     f"Spearman ρ (LIME rank vs single-feature impact): **{corr['rho']:.3f}** "
            #     f"(p={corr['pval']:.3g})."
            # )

            # Stability
            # booster_for_shap = _patch_base_score_in_modelfile(bst)

            # # # ✅ take the schema the Booster was trained with
            # FEAT_TRAIN = booster_for_shap.feature_names or X_train_encoded.columns.tolist()

            # # # hard-schema attrib function that also normalizes dash variants
            # get_attr_xgb = make_get_attribs_xgb_booster_with_schema(booster_for_shap, FEAT_TRAIN)

            # # (optional but recommended) align the test frame before calling stability_report
            # from src.shap import align_to_training_columns  # use the one in shap.py
            # X_test_aligned = align_to_training_columns(X_test_encoded.copy(), FEAT_TRAIN)

            # stab = stability_report(get_attr_xgb, X_test_aligned, n_boot=20, sample_frac=0.8, seed=42)


            # val_tab.markdown("### Stability")
            # val_tab.write(f"Kendall’s τ (mean ± sd): **{stab['kendall_tau_mean']:.3f} ± {stab['kendall_tau_std']:.3f}**")

            # def _xgb_model_key(model) -> str:
            #     """
            #     Deterministic fingerprint for a trained XGBoost model.
            #     Works for xgb.Booster and xgb.XGBClassifier.
            #     """
            #     # get raw bytes of the trained trees
            #     if isinstance(model, xgb.Booster):
            #         raw = model.save_raw()              # bytearray
            #         raw_b = bytes(raw)                  # -> bytes
            #         base = raw_b
            #         extra = b""
            #     elif hasattr(model, "get_booster"):     # XGBClassifier
            #         booster = model.get_booster()
            #         raw = booster.save_raw()            # bytearray
            #         raw_b = bytes(raw)
            #         base = raw_b
            #         # include key hyperparams to be extra-safe
            #         hp_names = ["n_estimators", "max_depth", "learning_rate", "subsample",
            #                     "colsample_bytree", "colsample_bylevel", "colsample_bynode",
            #                     "reg_lambda", "reg_alpha", "gamma", "min_child_weight", "random_state"]
            #         hp_vals = tuple(getattr(model, n, None) for n in hp_names)
            #         extra = repr(hp_vals).encode()
            #     else:
            #         # fallback
            #         return hashlib.sha1(repr(model).encode()).hexdigest()

            #     # stable digest string (streamlit-friendly)
            #     return hashlib.sha1(base + extra).hexdigest()

            # # 2) Cache full-test contribs (pred_contribs). Ignore unhashable model via leading underscore.
            # @st.cache_data(show_spinner=False)
            # def _xgb_full_attr(
            #     _model,                 # not hashed by Streamlit
            #     model_key: str,         # hashable fingerprint to invalidate cache when the model changes
            #     feature_names: tuple,   # tuple(X_test_encoded.columns)
            #     X_val_index: tuple,     # tuple(X_test_encoded.index)
            #     X_val_values: np.ndarray
            # ) -> pd.DataFrame:
            #     # get Booster
            #     if isinstance(_model, xgb.Booster):
            #         booster = _model
            #     elif hasattr(_model, "get_booster"):
            #         check_is_fitted(_model)
            #         booster = _model.get_booster()
            #     else:
            #         raise TypeError("Unsupported XGBoost model type for _xgb_full_attr")

            #     feats = list(feature_names)
            #     dm = xgb.DMatrix(X_val_values, feature_names=feats)
            #     contribs = booster.predict(dm, pred_contribs=True)   # (n, p+1) last col = bias
            #     A = contribs[:, :-1]                                 # drop bias to align with features
            #     return pd.DataFrame(A, index=list(X_val_index), columns=feats)

            # # 3) Pick which model to use for explanations
            # try:
            #     xgb_used = bst_patched   # if you patched base_score earlier
            # except NameError:
            #     xgb_used = bst           # Booster returned by retrain_xgb_model(...)

            # # Build cache
            # xgb_key = _xgb_model_key(xgb_used)
            # A_xgb_full = _xgb_full_attr(
            #     _model=xgb_used,
            #     model_key=xgb_key,
            #     feature_names=tuple(X_test_encoded.columns.tolist()),
            #     X_val_index=tuple(X_test_encoded.index.tolist()),
            #     X_val_values=X_test_encoded.to_numpy(),
            # )

            # # Getter: reuse cache for existing rows; recompute contribs only for perturbed rows
            # def get_attr_xgb_cached(X_batch: pd.DataFrame) -> np.ndarray:
            #     try:
            #         return A_xgb_full.loc[X_batch.index].to_numpy()
            #     except KeyError:
            #         feats = X_batch.columns.tolist()
            #         booster = xgb_used if isinstance(xgb_used, xgb.Booster) else xgb_used.get_booster()
            #         dm = xgb.DMatrix(X_batch[feats].to_numpy(), feature_names=feats)
            #         contribs = booster.predict(dm, pred_contribs=True)
            #         return contribs[:, :-1]

            # 4) Local stability UI
            # with val_tab:
            #     st.markdown("### Local Stability (XGBoost)")
            #     xgb_row_id = st.selectbox("Pick a row (XGB):", X_test_encoded.index.tolist(), index=0, key="xgb_row_pick")
            #     n_pert = st.slider("Perturbations", 10, 100, 30, key="xgb_npert")
            #     noise = st.number_input("Numeric noise (σ)", 0.0, 1.0, 0.02, 0.01, key="xgb_noise")
            #     flip  = st.number_input("Flip prob (binary/one-hot)", 0.0, 1.0, 0.05, 0.01, key="xgb_flip")

            #     # One-hots are strictly {0,1}; the rest numeric
            #     binary_cols_xgb  = [c for c in X_test_encoded.columns
            #                         if set(pd.unique(X_test_encoded[c])).issubset({0, 1})]
            #     numeric_cols_xgb = [c for c in X_test_encoded.columns if c not in binary_cols_xgb]

            #     if st.button("Run local stability (XGB)"):
            #         report = local_stability_report_generic(
            #             get_attr_for_batch=get_attr_xgb_cached,
            #             X_matrix=X_test_encoded,
            #             row_id=xgb_row_id,
            #             numeric_cols=numeric_cols_xgb,
            #             binary_cols=binary_cols_xgb,
            #             onehot_groups=onehot_groups,   # your existing dict
            #             n_perturb=n_pert,
            #             noise_scale=noise,
            #             flip_prob=flip,
            #             k_list=(5, 10)
            #         )
            #         st.write(f"Kendall’s τ (mean): **{report['kendall_tau_mean']:.3f}**")
            #         st.write(f"Spearman ρ (mean): **{report['spearman_rho_mean']:.3f}**")
            #         st.write(f"Avg std(|attrib|): **{report['mean_attr_std']:.4f}**")
            #         st.write({f"Top-{k} overlap": f"{report['topk_overlap_mean'][k]:.2f}" for k in (5, 10)})
            #         st.dataframe(report["attr_std_by_feature"].head(12).to_frame("std").style.format({"std": "{:.4f}"}))

            # Sanity (label randomization)
            # san = sanity_report(
            #     model=bst,
            #     X_val=X_test_encoded,
            #     y_val=y_test,
            #     get_attribs_callable=get_attr_xgb,
            #     randomize="labels",
            #     seed=42
            # )
            # val_tab.markdown("### Sanity")
            # val_tab.write(f"Spearman ρ vs original after label randomization: **{san['spearman_rho_vs_original']:.3f}** "
            #               "(should drop toward 0 if explanations depend on learned signal)")



            # get_xgb_explanation(bst, background_data_encoded, X_test_encoded, shap_tab)
            xgb_local_shap_by_truth(bst, X_test_encoded, X_test, y_test, 0.42, shap_tab, background_data_encoded)
            get_lime_explanations_binary(bst, X_test, X_train_encoded, X_test_encoded, y_test, 0.42, lime_tab, title_prefix="Local LIME – XGBoost")

            adapted = XGBoosterAdapter(bst, objective="binary:logistic")
            xgb_for_dice = ThresholdedModel(adapted, threshold=0.42)

            continuous_features = [
                "subscription_age",
                "bill_avg",
                "service_failure_count",          
                "download_avg",
                "upload_avg",
                "total_usage",
                "services_count",
                "download_over_limit"
            ]

            features_to_vary = [
                "is_tv_subscriber", "is_movie_package_subscriber", 
                "bill_avg",
                "download_over_limit",
                "contract_stage_0.5-1y", "contract_stage_1-2y", "contract_stage_>2y", "contract_stage_no_contract"
            ]

            permitted_range = {"bill_avg": [15, 150]}

            results = get_counterfactual_analysis(
                y_test=y_test,
                X_test=X_test_encoded,          
                X_train=X_train_encoded,
                y_train=y_train,
                model=xgb_for_dice,              
                continuous_features=continuous_features,
                counterfactual_tab=counterfactual_tab,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,       # raw bounds
                scaler=None,                           
                numeric_feature_names_in_scaler_order=None,
                immutable_features=immutable,
                onehot_groups=onehot_groups
            )

            print(X_train_encoded.info())


# ######################## Modeling ####################################################################





# #################### Training and evaluation ########################################3
# ROC-AUC
    
#     # Get probability that that customer will churn
#     positive_class_index  = list(log_reg_model.classes_).index(1)   # robust: find where class "1" is
#     churn_probability = log_reg_model.predict_proba(X_test)[:, positive_class_index]

#     # AUC number

#     # Check how well the predicted probabilities line up with the actual churn labels
#     auc = roc_auc_score(y_test, churn_probability)

#     # ROC curve with Plotly

#     # x/y coordinates to plot the ROC curve.
#     # They’re computed from true labels (y_test) and predicted probabilities (churn_probability)
#     false_positive, true_positive, _ = roc_curve(y_test, churn_probability)

#     roc_line = px.line(
#         pd.DataFrame({"FPR": false_positive, "TPR": true_positive}),
#         x="FPR", y="TPR",
#         title=f"ROC Curve — Logistic Regression (AUC = {auc:.3f})",
#         labels={"FPR":"False Positive Rate", "TPR":"True Positive Rate"}
#     )
#     roc_line.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
#     roc_line.update_layout(xaxis_range=[0,1], yaxis_range=[0,1])

#     st.plotly_chart(roc_line, use_container_width=True)
#     st.markdown('The model shows strong ranking performance with AUC = 0.862, meaning ' \
#     'it can reliably score churners above non-churners. The blue curve sits well above the dashed ' \
#     'random baseline. Around FPR ≈ 10%, the TPR is ~60–65%; at FPR ≈ 20%, TPR exceeds ~75%, indicating ' \
#     'you can capture many churners with relatively few false alarms. The “knee” of the curve is roughly '
#     'in the 0.10–0.20 FPR range— a sensible region to choose an operating threshold depending on your tolerance for false positives.')



#     
#     # Explainability
#     st.subheader('Explainability')
#     tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])

#     # Small background sample for SHAP (keeps it fast & stable)
#     background_features = X_train.sample(min(1000, len(X_train)), random_state=42)

#     explainer = shap.Explainer(log_reg_model, background_features )

#     tab_global, tab_local = tab_shap.tabs(["Global importance", "Per-customer"])

#     # ---------- GLOBAL IMPORTANCE ----------
#     tab_global.subheader('Global Feature Importance — Logistic Regression Model (SHAP)')

#     # explain a small evaluation sample (speed)
#     features_for_global  = X_test.sample(min(500, len(X_test)), random_state=42)
#     shap_global_explanations  = explainer(features_for_global)  # shap.Explanation

#     mean_abs_contrib_by_feature = np.abs(shap_global_explanations.values).mean(axis=0)

#     importance_table = (
#         pd.DataFrame({
#             "feature": features_for_global.columns,
#             "avg_abs_contribution": mean_abs_contrib_by_feature
#         })
#         .sort_values("avg_abs_contribution", ascending=False)
#     )

#     global_importance_bar = px.bar(
#         importance_table.head(15).iloc[::-1],  # reverse so the biggest bar is on top
#         x="avg_abs_contribution",
#         y="feature",
#         orientation="h",
#         title=f"Top 15 Features by Global Importance (SHAP) — Logistic Regression Model",
#         labels={"avg_abs_contribution": "Average |contribution|", "feature": "Feature"}
#         )  
    
#     tab_global.plotly_chart(global_importance_bar, use_container_width=True)
#     tab_global.caption("Average absolute SHAP value = how much a feature moves predictions on average. Bigger bar → more influence overall.")

#     # ------------------- SHAP per customer --------------------------
#     row_explanation = explainer(X_test.iloc[[0]])

#     per_feature_contrib = (
#         pd.DataFrame({
#             "feature": X_test.columns,
#             "shap_value": row_explanation.values[0],       # signed effect per feature
#             "feature_value": X_test.iloc[[0]].iloc[0].values # the (scaled/encoded) input values
#         })
#         .assign(abs_effect=lambda d: d["shap_value"].abs())
#         .sort_values("abs_effect", ascending=False)
#         .drop(columns="abs_effect")
#     )

#     fig_shap_customer = px.bar(
#         per_feature_contrib.head(12).iloc[::-1],
#         x="shap_value",          # was shap_contribution
#         y="feature",             # was feature_name
#         orientation="h",
#         title=f"Top 12 feature contributions — row 0",
#         labels={"shap_value": "Contribution (→ churn + / ← non-churn −)", "feature": "Feature"}
#     )

#     tab_local.plotly_chart(fig_shap_customer, use_container_width=True)
#     tab_local.markdown('Each bar shows how a feature pushed this customer’s prediction: right = ' \
#     'toward churn, left = toward non-churn. The biggest push toward churn comes from tenure '
#     '(likely short tenure) and MonthlyCharges (relatively high). Smaller pushes come from factors like ' \
#     'PhoneService and Electronic check. The main factors reducing churn risk for this customer are ' \
#     'InternetService_Fiber optic and TotalCharges (their values pull the prediction left). Magnitude = ' \
#     'strength of influence; the signed contributions sum (with the baseline) to this customer’s final churn probability.')

#     tab_local.dataframe(per_feature_contrib.head(12), use_container_width=True)
#     tab_local.markdown('Each row shows how a feature moved this customer’s prediction: positive = pushes ' \
#     'toward churn, negative = pushes toward non-churn. The biggest pushes toward churn are tenure (+1.68) '
#     'and MonthlyCharges (+0.85), with smaller pushes from PhoneService, Contract_Two year, PaymentMethod_Electronic ' \
#     'check, InternetService_No, and Contract_One year. The main factors reducing churn risk are InternetService_Fiber ' \
#     'optic (–0.67), TotalCharges (–0.67), plus smaller negatives from MultipleLines, StreamingMovies, and StreamingTV. ' \
#     'The last number is the standardized feature value for this customer (negative ≈ below the dataset average).')

#     #----------------------- LIME -----------------------
#     lime_explainer = LimeTabularExplainer(
#         training_data=X_train.values,
#          feature_names=X_train.columns.tolist(),
#          class_names=["No churn","Churn"],
#          discretize_continuous=True,
#          mode="classification"
#     )

#     lime_explanation = lime_explainer.explain_instance(
#         data_row=X_test.iloc[[0]].iloc[0].values,
#         predict_fn=log_reg_model.predict_proba,
#         num_features=10
#      )
    
#     lime_items = lime_explanation.as_list()  # list of (feature_or_rule, weight)
#     lime_table = pd.DataFrame(lime_items, columns=["feature_or_rule", "lime_weight"])

#     fig_lime_customer = px.bar(
#         lime_table.iloc[::-1],
#         x="lime_weight", y="feature_or_rule",
#         orientation="h",
#         title=f"LIME local explanation — row 0",
#         labels={"lime_weight":"Local weight (→ churn + / ← non-churn −)", "feature_or_rule":"Feature / Rule"}
#     )

#     tab_lime.plotly_chart(fig_lime_customer, use_container_width=True)
#     tab_lime.markdown('Each bar shows a simple rule near this customer and its local weight: right = pushes prediction toward ' \
#     'churn, left = pushes away. Here, rules like tenure <= -0.95, MonthlyCharges <= -0.96, Contract_Two year <= -0.56, '
#     'and InternetService_No <= -0.53 increase churn risk for this customer, while TotalCharges <= -0.83, InternetService_Fiber ' \
#     'optic <= -0.89, and several others reduce it.')
#     tab_lime.dataframe(lime_table, use_container_width=True)
#     tab_lime.markdown('Each rule shows a simple condition near this customer and its local weight: positive = pushes prediction' \
#     ' toward churn, negative = pushes toward non-churn. \nBiggest risk drivers here: very short tenure (tenure ≤ −0.95), ' \
#     'below-average monthly charges (MonthlyCharges ≤ −0.96), not on a 2-year contract (Contract_Two year ≤ −0.56), and having ' \
#     'internet service (InternetService_No ≤ −0.53 ⇒ not “No internet”).\nRisk-reducing factors: lower total charges (TotalCharges ≤ −0.83),' \
#     ' not fiber-optic (InternetService_Fiber optic ≤ −0.89), plus small reductions from StreamingMovies and MultipleLines.')

# # Random Forest
# 

#     # ROC-AUC
    
#     # Get probability that that customer will churn
#     positive_class_index  = list(random_forest_model.classes_).index(1)   # robust: find where class "1" is
#     churn_probability = random_forest_model.predict_proba(X_test)[:, positive_class_index]

#     # AUC number

#     # Check how well the predicted probabilities line up with the actual churn labels
#     auc = roc_auc_score(y_test, churn_probability)

#     # ROC curve with Plotly

#     # x/y coordinates to plot the ROC curve.
#     # They’re computed from true labels (y_test) and predicted probabilities (churn_probability)
#     false_positive, true_positive, _ = roc_curve(y_test, churn_probability)

#     roc_line = px.line(
#         pd.DataFrame({"FPR": false_positive, "TPR": true_positive}),
#         x="FPR", y="TPR",
#         title=f"ROC Curve — Logistic Regression (AUC = {auc:.3f})",
#         labels={"FPR":"False Positive Rate", "TPR":"True Positive Rate"}
#     )
#     roc_line.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
#     roc_line.update_layout(xaxis_range=[0,1], yaxis_range=[0,1])

#     st.plotly_chart(roc_line, use_container_width=True)
#     st.markdown('The model shows good ranking ability with AUC = 0.843, well above the dashed ' \
#             'random baseline. The curve rises quickly at low false-positive rates: around FPR ≈ 10% the ' \
#             'TPR (recall) is roughly ~60%, and by FPR ≈ 20% it’s around ~70–75%.')

#     # Explainability
#     # tabs
#     st.subheader("Explainability")
#     tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])

#     with tab_shap:
#         # sub-tabs INSIDE the SHAP tab
#         sub_global, sub_local = st.tabs(["Global importance", "Per-customer"])

#     # --- Global importance ---
#     with sub_global:
#         explainer = shap.TreeExplainer(random_forest_model)  # RF -> TreeExplainer

#         # smaller sample to keep it fast
#         X_sample = X_test.sample(min(300, len(X_test)), random_state=42)

#         exp = explainer(X_sample)           # shap.Explanation
#         vals = exp.values                   # numpy array

#         # handle possible 3D output (rows, features, classes)
#         if vals.ndim == 3:
#             pos_idx = list(random_forest_model.classes_).index(1)
#             vals = vals[:, :, pos_idx]

#         avg_abs = np.abs(vals).mean(axis=0)
#         importance_df = (
#             pd.DataFrame({"feature": X_sample.columns, "avg_abs_contribution": avg_abs})
#             .sort_values("avg_abs_contribution", ascending=False)
#         )

#         # BUILD A FIGURE, then plot it
#         import plotly.express as px
#         fig = px.bar(
#             importance_df.head(15).iloc[::-1],
#             x="avg_abs_contribution", y="feature", orientation="h",
#             title="Top 15 Features by Global Importance (SHAP — Random Forest)",
#             labels={"avg_abs_contribution":"Average |contribution|", "feature":"Feature"}
#         )
#         sub_global.plotly_chart(fig, use_container_width=True)
#         sub_global.markdown('This chart ranks features by how much they influence churn predictions on average '
#         '(magnitude only). Tenure is the strongest driver, followed by Internet service type (e.g., Fiber optic), ' \
#         'contract length (especially Two year), Total/Monthly charges, and Payment method (Electronic check). ' \
#         'Billing options (PaperlessBilling) and service add-ons (OnlineSecurity, TechSupport) also matter.')

#     # --- Per-customer (example pattern) ---
#     with sub_local:
#         row_idx = 0
#         x_row = X_test.iloc[[row_idx]]
#         row_exp = explainer(x_row)
#         row_vals = row_exp.values
#         if row_vals.ndim == 3:
#             pos_idx = list(random_forest_model.classes_).index(1)
#             row_vals = row_vals[:, :, pos_idx]

#         contrib_df = (
#             pd.DataFrame({
#                 "feature": X_test.columns,
#                 "shap_value": row_vals[0],
#                 "feature_value": x_row.iloc[0].values
#             })
#             .assign(abs_=lambda d: d.shap_value.abs())
#             .sort_values("abs_", ascending=False)
#             .drop(columns="abs_")
#         )

#         fig_row = px.bar(
#             contrib_df.head(12).iloc[::-1],
#             x="shap_value", y="feature", orientation="h",
#             title=f"Top 12 Feature Contributions — Row {row_idx}",
#             labels={"shap_value":"Contribution (→ churn + / ← non-churn −)", "feature":"Feature"}
#         )
#         sub_local.plotly_chart(fig_row, use_container_width=True)
#         sub_local.markdown('Bars show how each feature pushed this customer’s prediction: right = toward churn, ' \
#         'left = toward non-churn. The largest pushes toward churn come from tenure and TotalCharges, with smaller ' \
#         'positive effects from PaymentMethod_Electronic check, PhoneService, and some contract/internet indicators. ' \
#         'The main factor reducing risk is InternetService_Fiber optic (left bar), with small reductions from SeniorCitizen '
#         'and PaperlessBilling. Bar length = strength of influence; the signed contributions combine (with the baseline) to ' \
#         'produce this customer’s final churn probability.')
#         sub_local.dataframe(contrib_df.head(12), use_container_width=True)
#         sub_local.markdown('Positive SHAP values push this prediction toward churn; negative values push away. For this ' \
#                        'customer, the largest push toward churn comes from TotalCharges (+0.137).Smaller pushes come ' \
#                        'from Electronic check (+0.054), PhoneService (+0.043), Contract_Two year (+0.043), ' \
#                        'InternetService_No (+0.035), plus OnlineBackup/OnlineSecurity and Contract_One year (small positives). ' \
#                         'The main factors reducing churn risk are InternetService_Fiber optic (−0.063) and, slightly, SeniorCitizen (−0.016).')

#     #----------------------- LIME -----------------------
#     with tab_lime:
            
#         lime_explainer = LimeTabularExplainer(
#             training_data=X_train.values,
#              feature_names=X_train.columns.tolist(),
#              class_names=["No churn","Churn"],
#              discretize_continuous=True,
#              mode="classification"
#         )

#         lime_explanation = lime_explainer.explain_instance(
#             data_row=X_test.iloc[[0]].iloc[0].values,
#             predict_fn=random_forest_model.predict_proba,
#             num_features=10
#          )

#         lime_items = lime_explanation.as_list()  # list of (feature_or_rule, weight)
#         lime_table = pd.DataFrame(lime_items, columns=["feature_or_rule", "lime_weight"])

#         fig_lime_customer = px.bar(
#             lime_table.iloc[::-1],
#             x="lime_weight", y="feature_or_rule",
#             orientation="h",
#             title=f"LIME local explanation — row 0",
#             labels={"lime_weight":"Local weight (→ churn + / ← non-churn −)", "feature_or_rule":"Feature / Rule"}
#         )

#         tab_lime.plotly_chart(fig_lime_customer, use_container_width=True)
#         tab_lime.markdown('Bars show simple rules near this customer and their local weight: right = pushes toward churn, ' \
#         'left = pushes toward non-churn.\nBiggest risk drivers: not on a 2-year contract (Contract_Two year ≤ −0.56) '
#         'and short tenure (tenure ≤ −0.95).\nOther factors nudging risk up: low total charges (TotalCharges ≤ −0.83), ' \
#         'not on a 1-year contract, has internet service (InternetService_No ≤ −0.53 ⇒ not “no internet”), electronic check ' \
#         'payment, and signals consistent with no TechSupport/OnlineSecurity.\nRisk reducer: not on fiber optic '
#         '(InternetService_Fiber optic ≤ −0.89) strongly pulls the prediction toward non-churn; StreamingMovies also reduces risk slightly.')
#         tab_lime.dataframe(lime_table, use_container_width=True)
#         tab_lime.markdown('Each rule is a simple condition near this customer; the weight shows its local effect (positive → pushes ' \
#         'toward churn, negative → pushes toward non-churn).\nBiggest risk drivers: not on a 2-year contract '
#         'and very short tenure (both strong positives).\nOther risk-increasing signals: lower total charges '
#         '(early lifecycle), not on a 1-year contract, has internet service (vs. “no internet”), pays by electronic ' \
#         'check, and no TechSupport/OnlineSecurity.\nRisk reducers: not using fiber-optic internet (largest negative)'
#         ' and StreamingMovies (small negative).')

def display_fairness_table_lr(X_test, tuned_model, scaled_X_test_features, threshold):
    def group_fairness_table(df, group_col, y_true='y_true', y_proba='y_proba', threshold=0.5832, reference=None):
        """Return a per-group fairness table and the chosen reference group."""
        # pick reference = largest group if not provided
        ref = reference or df[group_col].value_counts().idxmax()

        def recall_pos(g):
            mask = g[y_true] == 1
            if mask.sum() == 0:
                return np.nan
            y_hat = (g[y_proba] >= threshold).astype(int)
            return recall_score(g.loc[mask, y_true], y_hat[mask])

        agg = (
            df.groupby(group_col)
              .apply(lambda g: pd.Series({
                  'n': len(g),
                  'observed_churn_rate': g[y_true].mean(),
                  'mean_pred_proba': g[y_proba].mean(),
                  f'prediction_rate@{threshold:.3f}': (g[y_proba] >= threshold).mean(),
                  'recall_pos': recall_pos(g),
              }))
              .reset_index()
        )

        ref_mean = agg.loc[agg[group_col] == ref, 'mean_pred_proba'].values[0]
        agg[f'Δ_vs_{ref}'] = (agg['mean_pred_proba'] - ref_mean).abs()
        agg['stat_parity_ok(≤0.05)'] = agg[f'Δ_vs_{ref}'] <= 0.05
        return agg.sort_values('n', ascending=False), ref

    group_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]  
    threshold = 0.5832
    X_test_original_groups = X_test[group_cols].copy()
    if hasattr(y_test, "dtype") and y_test.dtype == object:
        y_true = y_test.map({"No": 0, "Yes": 1}).astype(int)
    elif isinstance(y_test, pd.DataFrame) and "Churn" in y_test.columns:
        y_true = y_test["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    else:
        y_true = y_test.astype(int)  # already 0/1

    if hasattr(tuned_model, "predict_proba"):           # single estimator
        models = {"logreg_final": tuned_model}
    elif isinstance(tuned_model, (list, tuple)):        # list/tuple of estimators
        models = {f"model_{i}": m for i, m in enumerate(tuned_model)}
    elif isinstance(tuned_model, dict):                  # already a dict
        models = tuned_model
    else:
        raise TypeError("tuned_model must be an estimator, list/tuple of estimators, or dict.")

    X_for_pred = scaled_X_test_features

    results = {}
    for name, model in models.items():
        # get probabilities for the positive class
        y_proba = model.predict_proba(X_for_pred)[:, 1]

        # build evaluation frame aligned by index
        df_eval = X_test_original_groups.copy()
        df_eval["y_true"] = pd.Series(y_true.values, index=df_eval.index)
        df_eval["y_proba"] = pd.Series(y_proba, index=df_eval.index)

    # per-group tables for this model
        model_tables = {}
        for gcol in group_cols:
            sub = df_eval[[gcol, "y_true", "y_proba"]].dropna()
            table, ref = group_fairness_table(
                sub, group_col=gcol, y_true="y_true", y_proba="y_proba", threshold=threshold
            )
            model_tables[gcol] = (table, ref)
        results[name] = model_tables

    st.title("Fairness Evaluation — Group-Based Metrics")
    st.write("Global threshold for recall comparisons (equal opportunity): ", threshold)

    for model_name, tables in results.items():
        with st.expander(f"Model: {model_name}", expanded=False):
            for gcol, (table, ref) in tables.items():
                st.subheader(f"Group column: {gcol} (ref: {ref})")
                st.dataframe(table, use_container_width=True)

                # CSV download
                csv_buf = StringIO()
                table.to_csv(csv_buf, index=False)
                st.download_button(
                    label=f"Download CSV — {model_name} · {gcol}",
                    data=csv_buf.getvalue(),
                    file_name=f"fairness_{model_name}_{gcol}.csv",
                    mime="text/csv"
                )
