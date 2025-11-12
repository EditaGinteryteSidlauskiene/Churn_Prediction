"""Streamlit dashboard for telco and internet churn datasets.

The module focuses on clarity and reuse by structuring the data preparation,
analysis, and modelling flows into dedicated helper functions.  This keeps the
main `main()` entry point compact and makes it easier to tweak individual
sections without touching the rest of the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd
import streamlit as st

from src.config import data_path
from src.features import convert_str_to_numeric, fill_na_values_with_mean, create_churn_flag
from src.exploratory_data_analysis import show_balance_between_yes_and_no
from src.internet_data_analysis import (
    engineer_internet_features,
    get_internet_data_analysis,
    get_internet_data_missing_values_caption,
    get_internet_data_null_heatmap,
    get_internet_data_null_hist,
    get_internet_dataset_null_percentage,
    get_internet_pie_caption,
    show_internet_categorical_drivers,
    show_internet_correlations_and_interactions,
    show_internet_engagement_and_churn,
    show_internet_numeric_drivers,
)
from src.internet_data_preprocessing import (
    encode_internet_data,
    get_scaled_internet_features,
    split_internet_data,
)
from src.lime import get_lime_explanations_binary
from src.logistic_regression import (
    display_lr_metrics,
    get_internet_lr_metrics_caption,
    get_telco_lr_metrics_caption,
    retrain_lg_model,
)
from src.random_forest import (
    display_rf_metrics,
    get_internet_rf_metrics_caption,
    get_telco_rf_metrics_caption,
    retrain_rf_model,
)
from src.shap import lr_local_shap_by_truth, rf_local_shap_by_truth, xgb_local_shap_by_truth
from src.telco_data_analysis import (
    get_correlation_heatmap_with_engineered_features,
    get_telco_data_analysis,
    get_text_about_engineered_features,
    get_telco_pie_caption,
    show_telco_categorical_drivers,
    show_telco_correlations_and_interactions,
    show_telco_demographic_drivers,
    show_telco_engagement_and_churn,
    show_telco_numeric_drivers,
)
from src.telco_data_preprocessing import (
    encode_telco_data,
    engineer_telco_data_features,
    get_scaled_telco_features,
    map_telco_data_features,
    split_telco_data,
)
from src.threshold_model import ThresholdedModel
from src.xgboost import (
    display_xgb_metrics,
    get_internet_xgb_metrics_caption,
    get_telco_xgb_metrics_caption,
    retrain_xgb_model,
)
from src.XGBooster_adapter import XGBoosterAdapter
from src.dice import get_counterfactual_analysis


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TelcoContext:
    """Preprocessed artefacts for the telco dataset."""

    cleaned: pd.DataFrame
    engineered: pd.DataFrame
    encoded_full: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_encoded: pd.DataFrame
    X_test_encoded: pd.DataFrame
    scaled_train: pd.DataFrame
    scaled_test: pd.DataFrame
    scaler: object
    numeric_columns: Iterable[str]
    background_scaled: pd.DataFrame
    background_encoded: pd.DataFrame


@dataclass
class InternetContext:
    """Preprocessed artefacts for the internet dataset."""

    prepared: pd.DataFrame
    engineered: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_encoded: pd.DataFrame
    X_test_encoded: pd.DataFrame
    scaled_train: pd.DataFrame
    scaled_test: pd.DataFrame
    scaler: object
    numeric_columns: Iterable[str]
    background_scaled: pd.DataFrame
    background_encoded: pd.DataFrame


# ---------------------------------------------------------------------------
# Constants and configuration helpers
# ---------------------------------------------------------------------------

TELCO_ANALYSIS_OPTIONS = [
    "Choose Analysis Type",
    "Numeric Drivers of Churn",
    "Categorical Drivers of Churn",
    "Service Engagement and Churn",
    "Demographic Drivers of Churn",
    "Correlations and Interactions",
]

INTERNET_ANALYSIS_OPTIONS = [
    "Choose Analysis Type",
    "Numeric Drivers of Churn",
    "Categorical Drivers of Churn",
    "Service Engagement and Churn",
    "Correlations and Interactions",
]

MODEL_OPTIONS = [
    "Choose Prediction Model",
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]

TELCO_IMMUTABLE = {
    "Is_female",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "TotalCharges",
    "IsLongTermContract",
    "AvgMonthlyCharge",
    "AvgPricePerService",
    "OnlineServiceCount",
}

TELCO_ONEHOT_GROUPS = {
    "Contract": ["Contract_One year", "Contract_Two year"],
    "PaymentMethod": [
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
    ],
    "InternetService": [
        "InternetService_DSL",
        "InternetService_Fiber optic",
        "InternetService_No",
    ],
}

TELCO_CONTINUOUS_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlyCharge",
    "OnlineServiceCount",
    "AvgPricePerService",
]

TELCO_FEATURES_TO_VARY = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "InternetService_Fiber optic",
    "InternetService_No",
]

INTERNET_IMMUTABLE = {
    "subscription_age",
    "download_avg",
    "upload_avg",
    "total_usage",
    "usage_tier_Medium",
    "usage_tier_Heavy",
}

INTERNET_ONEHOT_GROUPS = {
    "total_usage": ["usage_tier_Medium", "usage_tier_Heavy"],
    "subscription_age": [
        "contract_stage_0.5-1y",
        "contract_stage_1-2y",
        "contract_stage_>2y",
        "contract_stage_no_contract",
    ],
}

INTERNET_CONTINUOUS_FEATURES = [
    "subscription_age",
    "bill_avg",
    "service_failure_count",
    "download_avg",
    "upload_avg",
    "total_usage",
    "services_count",
    "download_over_limit",
]

INTERNET_FEATURES_TO_VARY = [
    "is_tv_subscriber",
    "is_movie_package_subscriber",
    "bill_avg",
    "download_over_limit",
    "contract_stage_0.5-1y",
    "contract_stage_1-2y",
    "contract_stage_>2y",
    "contract_stage_no_contract",
]

TELCO_THRESHOLDS = {
    "lr": 0.5832,
    "rf": 0.5418,
    "xgb": 0.58,
}

INTERNET_THRESHOLDS = {
    "lr": 0.408,
    "rf": 0.426,
    "xgb": 0.42,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_datasets() -> Dict[str, pd.DataFrame]:
    """Load the raw datasets from disk."""

    telco = pd.read_csv(data_path("telco_data.csv"))
    internet = pd.read_csv(data_path("internet_service_churn.csv"))
    internet = internet.rename(columns={"reamining_contract": "remaining_contract"})
    return {"Telco Dataset": telco, "Internet Dataset": internet}


def sample_background(X: pd.DataFrame, y: pd.Series, target_size: int) -> pd.DataFrame:
    """Create a class-balanced background sample for explanation tooling."""

    if X.empty:
        return X

    if isinstance(y, pd.DataFrame):
        y_series = y.iloc[:, 0]
    elif isinstance(y, pd.Series):
        y_series = y
    else:
        y_series = pd.Series(y)
    y_series = y_series.reindex(X.index)

    if len(X) <= target_size:
        return X.copy()

    frac = min(1.0, target_size / len(X))
    sampled = (
        X.assign(Churn=y_series.values)
        .groupby("Churn", group_keys=False)
        .apply(lambda frame: frame.sample(frac=frac, random_state=42))
        .drop(columns="Churn")
    )
    return sampled


def render_dataset_overview(dataset_name: str, df: pd.DataFrame, about_text: str) -> None:
    """Display high-level dataset information and preview."""

    st.markdown("#### Data Exploratory Analysis")
    st.markdown(f"##### {dataset_name}")
    with st.expander("About this dataset"):
        st.write(about_text)

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    n_rows, n_cols = df.shape
    info_col1.metric("Rows", f"{n_rows:,}")
    info_col2.metric("Columns", f"{n_cols:,}")
    info_col3.metric("Total Missing", f"{int(df.isna().sum().sum()):,}")
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    info_col4.metric("Memory", f"{memory_mb:.1f} MB")

    st.markdown("###### First five rows of the dataset")
    st.write(df.head())


def create_sidebar_selectbox(label: str, options: Iterable[str]) -> str:
    """Create a collapsed select box in the sidebar."""

    return st.sidebar.selectbox(
        options=options,
        label=label,
        label_visibility="collapsed",
        index=0,
    )


def create_metric_tabs(
    title: str,
    metrics,
    display_fn,
    caption_fn=None,
):
    """Render metric and explanation tabs and return the explanation tabs."""

    st.subheader(title)
    metrics_tab, explanation_tab = st.tabs(["Metrics", "Explanation"])
    caption_column = display_fn(metrics_tab, metrics)
    if caption_fn is not None:
        caption_fn(caption_column)
    shap_tab, lime_tab, counter_tab = explanation_tab.tabs(["SHAP", "LIME", "Counterfactuals"])
    return shap_tab, lime_tab, counter_tab


# ---------------------------------------------------------------------------
# Telco dataset preparation and rendering
# ---------------------------------------------------------------------------


def prepare_telco_context(telco: pd.DataFrame) -> TelcoContext:
    """Perform preprocessing, encoding, and scaling for the telco dataset."""

    cleaned = telco.copy()
    cleaned["TotalCharges"] = convert_str_to_numeric(cleaned["TotalCharges"])
    cleaned["TotalCharges"] = fill_na_values_with_mean(cleaned["TotalCharges"])
    cleaned["ChurnFlag"] = create_churn_flag(cleaned["Churn"])

    map_telco_data_features(cleaned)
    engineered = engineer_telco_data_features(cleaned)
    encoded_full = encode_telco_data(cleaned, engineered)

    X_train, X_test, y_train, y_test = split_telco_data(engineered)
    X_train_encoded = encode_telco_data(cleaned, X_train)
    X_test_encoded = encode_telco_data(cleaned, X_test)

    scaler, scaled_train, scaled_test, numeric_columns = get_scaled_telco_features(
        X_train_encoded, X_test_encoded
    )

    background_scaled = sample_background(scaled_train, y_train, target_size=500)
    background_encoded = sample_background(X_train_encoded, y_train, target_size=500)

    return TelcoContext(
        cleaned=cleaned,
        engineered=engineered,
        encoded_full=encoded_full,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_encoded=X_train_encoded,
        X_test_encoded=X_test_encoded,
        scaled_train=scaled_train,
        scaled_test=scaled_test,
        scaler=scaler,
        numeric_columns=numeric_columns,
        background_scaled=background_scaled,
        background_encoded=background_encoded,
    )


def render_telco_analysis(selection: str, context: TelcoContext) -> None:
    """Render exploratory analysis panels for the telco dataset."""

    if selection == "Numeric Drivers of Churn":
        st.subheader(selection)
        show_telco_numeric_drivers(context.cleaned)
    elif selection == "Categorical Drivers of Churn":
        st.subheader(selection)
        show_telco_categorical_drivers(context.cleaned)
    elif selection == "Service Engagement and Churn":
        st.subheader(selection)
        show_telco_engagement_and_churn(context.cleaned)
    elif selection == "Demographic Drivers of Churn":
        st.subheader(selection)
        show_telco_demographic_drivers(context.cleaned)
    elif selection == "Correlations and Interactions":
        st.subheader(selection)
        get_text_about_engineered_features()
        get_correlation_heatmap_with_engineered_features(context.encoded_full)
        show_telco_correlations_and_interactions(context.cleaned)


def render_telco_logistic(context: TelcoContext) -> None:
    metrics, model = retrain_lg_model(
        context.scaled_train,
        context.y_train,
        context.scaled_test,
        context.y_test,
        TELCO_THRESHOLDS["lr"],
        0.00316,
    )

    shap_tab, lime_tab, counter_tab = create_metric_tabs(
        "Logistic Regression Performance Analysis",
        metrics,
        display_lr_metrics,
        get_telco_lr_metrics_caption,
    )

    lr_local_shap_by_truth(
        lr_model=model,
        background_data=context.background_scaled,
        X_test_scaled=context.scaled_test,
        X_test=context.X_test,
        y_test=context.y_test,
        shap_tab=shap_tab,
        threshold=TELCO_THRESHOLDS["lr"],
        top_display=12,
    )
    get_lime_explanations_binary(
        model,
        context.X_test,
        context.scaled_train,
        context.scaled_test,
        context.y_test,
        TELCO_THRESHOLDS["lr"],
        lime_tab,
        title_prefix="Local LIME – Logistic Regression",
    )

    lr_for_dice = ThresholdedModel(model, threshold=TELCO_THRESHOLDS["lr"])
    get_counterfactual_analysis(
        y_test=context.y_test,
        X_test=context.scaled_test,
        X_train=context.scaled_train,
        y_train=context.y_train,
        model=lr_for_dice,
        continuous_features=TELCO_CONTINUOUS_FEATURES,
        counterfactual_tab=counter_tab,
        outcome_name="Churn",
        total_CFs=6,
        features_to_vary=TELCO_FEATURES_TO_VARY,
        permitted_range={"MonthlyCharges": [15, 150]},
        scaler=context.scaler,
        numeric_feature_names_in_scaler_order=context.numeric_columns,
        immutable_features=TELCO_IMMUTABLE,
        onehot_groups=TELCO_ONEHOT_GROUPS,
    )


def render_telco_random_forest(context: TelcoContext) -> None:
    metrics, model = retrain_rf_model(
        context.X_train_encoded,
        context.y_train,
        context.X_test_encoded,
        context.y_test,
        TELCO_THRESHOLDS["rf"],
        477,
        17,
        12,
        0.3,
        None,
    )

    shap_tab, lime_tab, counter_tab = create_metric_tabs(
        "Random Forest Performance Analysis",
        metrics,
        display_rf_metrics,
        get_telco_rf_metrics_caption,
    )

    rf_local_shap_by_truth(
        model,
        context.X_test_encoded,
        context.X_test,
        context.y_test,
        TELCO_THRESHOLDS["rf"],
        shap_tab,
        context.background_encoded,
    )
    get_lime_explanations_binary(
        model,
        context.X_test,
        context.X_train_encoded,
        context.X_test_encoded,
        context.y_test,
        TELCO_THRESHOLDS["rf"],
        lime_tab,
        title_prefix="Local LIME – Random Forest",
    )

    rf_for_dice = ThresholdedModel(model, threshold=TELCO_THRESHOLDS["rf"])
    get_counterfactual_analysis(
        y_test=context.y_test,
        X_test=context.X_test_encoded,
        X_train=context.X_train_encoded,
        y_train=context.y_train,
        model=rf_for_dice,
        continuous_features=TELCO_CONTINUOUS_FEATURES,
        counterfactual_tab=counter_tab,
        features_to_vary=TELCO_FEATURES_TO_VARY,
        permitted_range={"MonthlyCharges": [15, 150]},
        scaler=None,
        numeric_feature_names_in_scaler_order=None,
        immutable_features=TELCO_IMMUTABLE,
        onehot_groups=TELCO_ONEHOT_GROUPS,
    )


def render_telco_xgboost(context: TelcoContext) -> None:
    metrics_row, booster, _ = retrain_xgb_model(
        context.X_train_encoded,
        context.y_train,
        context.X_test_encoded,
        context.y_test,
        TELCO_THRESHOLDS["xgb"],
        0.05,
        5,
        0.872,
        0.853,
        0.715,
        17.402,
        0.006,
    )

    shap_tab, lime_tab, counter_tab = create_metric_tabs(
        "XGBoost Performance Analysis",
        metrics_row,
        display_xgb_metrics,
        get_telco_xgb_metrics_caption,
    )

    xgb_local_shap_by_truth(
        booster,
        context.X_test_encoded,
        context.X_test,
        context.y_test,
        TELCO_THRESHOLDS["xgb"],
        shap_tab,
        context.background_encoded,
    )
    get_lime_explanations_binary(
        booster,
        context.X_test,
        context.X_train_encoded,
        context.X_test_encoded,
        context.y_test,
        TELCO_THRESHOLDS["xgb"],
        lime_tab,
        title_prefix="Local LIME – XGBoost",
    )

    adapted = XGBoosterAdapter(booster, objective="binary:logistic")
    xgb_for_dice = ThresholdedModel(adapted, threshold=TELCO_THRESHOLDS["xgb"])
    get_counterfactual_analysis(
        y_test=context.y_test,
        X_test=context.X_test_encoded,
        X_train=context.X_train_encoded,
        y_train=context.y_train,
        model=xgb_for_dice,
        continuous_features=TELCO_CONTINUOUS_FEATURES,
        counterfactual_tab=counter_tab,
        features_to_vary=TELCO_FEATURES_TO_VARY,
        permitted_range={"MonthlyCharges": [15, 150]},
        scaler=None,
        numeric_feature_names_in_scaler_order=None,
        immutable_features=TELCO_IMMUTABLE,
        onehot_groups=TELCO_ONEHOT_GROUPS,
    )


def render_telco_models(selection: str, context: TelcoContext) -> None:
    if selection == "Logistic Regression":
        render_telco_logistic(context)
    elif selection == "Random Forest":
        render_telco_random_forest(context)
    elif selection == "XGBoost":
        render_telco_xgboost(context)


def render_telco_dashboard(telco_df: pd.DataFrame) -> None:
    analysis_choice = create_sidebar_selectbox("Choose Analysis Type", TELCO_ANALYSIS_OPTIONS)
    model_choice = create_sidebar_selectbox("Choose Prediction Model", MODEL_OPTIONS)

    context = prepare_telco_context(telco_df)

    pie_col1, pie_col2 = st.columns(2)
    pie_col1.plotly_chart(show_balance_between_yes_and_no(context.cleaned))
    pie_col2.caption(get_telco_pie_caption(), unsafe_allow_html=True)

    render_telco_analysis(analysis_choice, context)

    if model_choice != MODEL_OPTIONS[0]:
        render_telco_models(model_choice, context)


# ---------------------------------------------------------------------------
# Internet dataset preparation and rendering
# ---------------------------------------------------------------------------


def prepare_internet_context(internet: pd.DataFrame) -> InternetContext:
    """Perform preprocessing, encoding, and scaling for the internet dataset."""

    prepared = internet.copy()
    prepared = prepared.rename(columns={"churn": "Churn"})
    prepared = prepared.dropna(subset=["upload_avg", "download_avg"])

    engineered = engineer_internet_features(prepared)
    engineered["remaining_contract"] = engineered["remaining_contract"].fillna(0)

    X_train, X_test, y_train, y_test = split_internet_data(engineered)
    X_train_encoded = encode_internet_data(X_train)
    X_test_encoded = encode_internet_data(X_test)

    scaler, scaled_train, scaled_test, numeric_columns = get_scaled_internet_features(
        X_train_encoded, X_test_encoded
    )

    background_scaled = sample_background(scaled_train, y_train, target_size=1000)
    background_encoded = sample_background(X_train_encoded, y_train, target_size=1000)

    return InternetContext(
        prepared=prepared,
        engineered=engineered,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_encoded=X_train_encoded,
        X_test_encoded=X_test_encoded,
        scaled_train=scaled_train,
        scaled_test=scaled_test,
        scaler=scaler,
        numeric_columns=numeric_columns,
        background_scaled=background_scaled,
        background_encoded=background_encoded,
    )


def render_internet_analysis(selection: str, context: InternetContext) -> None:
    dataset_for_plots = context.prepared.copy()
    dataset_for_plots["Churn"] = dataset_for_plots["Churn"].map({1: "Yes", 0: "No"})

    if selection == "Numeric Drivers of Churn":
        st.subheader(selection)
        show_internet_numeric_drivers(dataset_for_plots)
    elif selection == "Categorical Drivers of Churn":
        st.subheader(selection)
        show_internet_categorical_drivers(context.engineered)
    elif selection == "Service Engagement and Churn":
        st.subheader(selection)
        show_internet_engagement_and_churn(context.engineered)
    elif selection == "Correlations and Interactions":
        st.subheader(selection)
        show_internet_correlations_and_interactions(context.engineered)


def render_internet_logistic(context: InternetContext) -> None:
    metrics, model = retrain_lg_model(
        context.scaled_train,
        context.y_train,
        context.scaled_test,
        context.y_test,
        INTERNET_THRESHOLDS["lr"],
        0.01,
    )

    shap_tab, lime_tab, counter_tab = create_metric_tabs(
        "Logistic Regression Performance Analysis",
        metrics,
        display_lr_metrics,
        get_internet_lr_metrics_caption,
    )

    lr_local_shap_by_truth(
        lr_model=model,
        background_data=context.background_scaled,
        X_test_scaled=context.scaled_test,
        X_test=context.X_test,
        y_test=context.y_test,
        shap_tab=shap_tab,
        threshold=INTERNET_THRESHOLDS["lr"],
        top_display=12,
    )
    get_lime_explanations_binary(
        model,
        context.X_test,
        context.scaled_train,
        context.scaled_test,
        context.y_test,
        INTERNET_THRESHOLDS["lr"],
        lime_tab,
        title_prefix="Local LIME – Logistic Regression",
    )

    lr_for_dice = ThresholdedModel(model, threshold=INTERNET_THRESHOLDS["lr"])
    get_counterfactual_analysis(
        y_test=context.y_test,
        X_test=context.scaled_test,
        X_train=context.scaled_train,
        y_train=context.y_train,
        model=lr_for_dice,
        continuous_features=INTERNET_CONTINUOUS_FEATURES,
        counterfactual_tab=counter_tab,
        outcome_name="Churn",
        total_CFs=6,
        features_to_vary=INTERNET_FEATURES_TO_VARY,
        permitted_range={"MonthlyCharges": [15, 150]},
        scaler=context.scaler,
        numeric_feature_names_in_scaler_order=context.numeric_columns,
        immutable_features=INTERNET_IMMUTABLE,
        onehot_groups=INTERNET_ONEHOT_GROUPS,
    )


def render_internet_random_forest(context: InternetContext) -> None:
    metrics, model = retrain_rf_model(
        context.X_train_encoded,
        context.y_train,
        context.X_test_encoded,
        context.y_test,
        INTERNET_THRESHOLDS["rf"],
        609,
        10,
        3,
        0.9,
        28,
    )

    shap_tab, lime_tab, counter_tab = create_metric_tabs(
        "Random Forest Performance Analysis",
        metrics,
        display_rf_metrics,
        get_internet_rf_metrics_caption,
    )

    rf_local_shap_by_truth(
        model,
        context.X_test_encoded,
        context.X_test,
        context.y_test,
        INTERNET_THRESHOLDS["rf"],
        shap_tab,
        context.background_encoded,
    )
    get_lime_explanations_binary(
        model,
        context.X_test,
        context.X_train_encoded,
        context.X_test_encoded,
        context.y_test,
        INTERNET_THRESHOLDS["rf"],
        lime_tab,
        title_prefix="Local LIME – Random Forest",
    )

    rf_for_dice = ThresholdedModel(model, threshold=INTERNET_THRESHOLDS["rf"])
    get_counterfactual_analysis(
        y_test=context.y_test,
        X_test=context.X_test_encoded,
        X_train=context.X_train_encoded,
        y_train=context.y_train,
        model=rf_for_dice,
        continuous_features=INTERNET_CONTINUOUS_FEATURES,
        counterfactual_tab=counter_tab,
        features_to_vary=INTERNET_FEATURES_TO_VARY,
        permitted_range={"MonthlyCharges": [15, 150]},
        scaler=None,
        numeric_feature_names_in_scaler_order=None,
        immutable_features=INTERNET_IMMUTABLE,
        onehot_groups=INTERNET_ONEHOT_GROUPS,
    )


def render_internet_xgboost(context: InternetContext) -> None:
    metrics_row, booster, _ = retrain_xgb_model(
        context.X_train_encoded,
        context.y_train,
        context.X_test_encoded,
        context.y_test,
        INTERNET_THRESHOLDS["xgb"],
        0.01423,
        9,
        2.7843,
        0.7848,
        0.6645,
        1.0097,
        0.0041,
    )

    shap_tab, lime_tab, counter_tab = create_metric_tabs(
        "XGBoost Performance Analysis",
        metrics_row,
        display_xgb_metrics,
        get_internet_xgb_metrics_caption,
    )

    xgb_local_shap_by_truth(
        booster,
        context.X_test_encoded,
        context.X_test,
        context.y_test,
        INTERNET_THRESHOLDS["xgb"],
        shap_tab,
        context.background_encoded,
    )
    get_lime_explanations_binary(
        booster,
        context.X_test,
        context.X_train_encoded,
        context.X_test_encoded,
        context.y_test,
        INTERNET_THRESHOLDS["xgb"],
        lime_tab,
        title_prefix="Local LIME – XGBoost",
    )

    adapted = XGBoosterAdapter(booster, objective="binary:logistic")
    xgb_for_dice = ThresholdedModel(adapted, threshold=INTERNET_THRESHOLDS["xgb"])
    get_counterfactual_analysis(
        y_test=context.y_test,
        X_test=context.X_test_encoded,
        X_train=context.X_train_encoded,
        y_train=context.y_train,
        model=xgb_for_dice,
        continuous_features=INTERNET_CONTINUOUS_FEATURES,
        counterfactual_tab=counter_tab,
        features_to_vary=INTERNET_FEATURES_TO_VARY,
        permitted_range={"MonthlyCharges": [15, 150]},
        scaler=None,
        numeric_feature_names_in_scaler_order=None,
        immutable_features=INTERNET_IMMUTABLE,
        onehot_groups=INTERNET_ONEHOT_GROUPS,
    )


def render_internet_models(selection: str, context: InternetContext) -> None:
    if selection == "Logistic Regression":
        render_internet_logistic(context)
    elif selection == "Random Forest":
        render_internet_random_forest(context)
    elif selection == "XGBoost":
        render_internet_xgboost(context)


def render_internet_dashboard(internet_df: pd.DataFrame) -> None:
    analysis_choice = create_sidebar_selectbox("Choose Analysis Type", INTERNET_ANALYSIS_OPTIONS)
    model_choice = create_sidebar_selectbox("Choose Prediction Model", MODEL_OPTIONS)

    context = prepare_internet_context(internet_df)

    pie_col1, pie_col2 = st.columns(2)
    pie_col1.plotly_chart(show_balance_between_yes_and_no(context.prepared))
    pie_col2.caption(get_internet_pie_caption(), unsafe_allow_html=True)

    if (
        analysis_choice == INTERNET_ANALYSIS_OPTIONS[0]
        and model_choice == MODEL_OPTIONS[0]
    ):
        st.markdown("##### Exploring Missing Data")
        heatmap_col1, heatmap_col2 = st.columns(2)
        figure, _ = get_internet_data_null_heatmap(internet_df)
        heatmap_col1.pyplot(figure)
        heatmap_col2.plotly_chart(get_internet_data_null_hist(internet_df))

        missing_download = get_internet_dataset_null_percentage(internet_df["download_avg"])
        missing_contract = get_internet_dataset_null_percentage(internet_df["remaining_contract"])
        st.caption(
            get_internet_data_missing_values_caption(missing_download, missing_contract),
            unsafe_allow_html=True,
        )
    else:
        render_internet_analysis(analysis_choice, context)

    if model_choice != MODEL_OPTIONS[0]:
        render_internet_models(model_choice, context)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
    st.title("Churn Prediction Dashboard")
    st.sidebar.subheader("Visualization Settings")

    datasets = load_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Choose a dataset:",
        list(datasets.keys()),
    )

    df = datasets[selected_dataset]
    about_text = (
        get_telco_data_analysis() if selected_dataset == "Telco Dataset" else get_internet_data_analysis()
    )
    render_dataset_overview(selected_dataset, df, about_text)

    if selected_dataset == "Telco Dataset":
        render_telco_dashboard(df)
    else:
        render_internet_dashboard(df)


if __name__ == "__main__":
    main()
