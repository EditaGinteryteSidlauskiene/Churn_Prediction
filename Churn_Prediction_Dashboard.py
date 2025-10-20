import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
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
from src.logistic_regression import get_internet_metrics_caption, get_telco_metrics_caption, display_lr_metrics, perform_primary_lg_training, cross_validate_lg_model, hyperparameter_tune_lg, retrain_lg_model
from src.random_forest import get_internet_metrics_caption, get_telco_metrics_caption, display_rf_metrics, perform_primary_rf_training, cross_validate_rf_model, hyperparameter_tune_rf, retrain_rf_model
from src.xgboost import get_internet_metrics_caption, get_telco_metrics_caption, display_xgb_metrics, perform_primary_xgb_training, cross_validate_xgb_model, xgb_random_search, xgb_grid_refine, retrain_xgb_model
from sklearn.model_selection import StratifiedKFold
from src.shap import get_lr_explanation, get_rf_explanation


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

        scaled_X_train_features, scaled_X_test_features = get_scaled_telco_features(X_train_encoded, X_test_encoded)
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
        if selected_prediction_model == 'Logistic Regression':
            
            # perform_primary_lg_training(scaled_X_train_features, y_train, scaled_X_test_features, y_test, "balanced")
            # cross_validate_lg_model(X_train_encoded, y_train, cross_validation, "balanced")
            # hyperparameter_tune_lg(X_train_encoded, y_train, cross_validation)
            metrics, tuned_model = retrain_lg_model(scaled_X_train_features, y_train, scaled_X_test_features, y_test, 0.5832, 0.00316)
            st.subheader("Logistic Regression Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_lr_metrics(metrics_tab, metrics)
            get_telco_metrics_caption(column2)
            
            # Explainability models
            shap_tab, lime_tabe = explainability_tab.tabs(["SHAP", "LIME"])
            get_lr_explanation(tuned_model, background_data_scaled, scaled_X_test_features, shap_tab)
    

        elif selected_prediction_model == 'Random Forest':
            
            # perform_primary_rf_training(X_train_encoded, y_train, X_test_encoded, y_test, "balanced")
            # cross_validate_rf_model(X_train_encoded, y_train, cross_validation, "balanced")
            # hyperparameter_tune_rf(X_train_encoded, y_train, cross_validation)
            metrics, tuned_model = retrain_rf_model(X_train_encoded, y_train, X_test_encoded, y_test, 0.5418, 477, 17, 12, 0.3, None)
            st.subheader("Random Forest Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_rf_metrics(metrics_tab, metrics)
            get_telco_metrics_caption(column2)

            # Explainability models
            shap_tab, lime_tabe = explainability_tab.tabs(["SHAP", "LIME"])
            get_rf_explanation(tuned_model, background_data_encoded, X_test_encoded, shap_tab)

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
            st.subheader("XGBoost Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_xgb_metrics(metrics_tab, row)
            get_telco_metrics_caption(column2)
        
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

        scaled_X_train_features, scaled_X_test_features = get_scaled_internet_features(X_train_encoded, X_test_encoded)

        #------------------- Train models -----------------------------

        cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if selected_prediction_model == 'Logistic Regression':
            
            # perform_primary_lg_training(scaled_X_train_features, y_train, scaled_X_test_features, y_test, None)
            # cross_validate_lg_model(X_train_encoded, y_train, cross_validation, None)
            # hyperparameter_tune_lg(X_train_encoded, y_train, cross_validation)
            metrics = retrain_lg_model(scaled_X_train_features, y_train, scaled_X_test_features, y_test, 0.408, 0.01)
            st.subheader("Logistic Regression Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_lr_metrics(metrics_tab, metrics)
            get_internet_metrics_caption(column2)

            models = ["Logistic Regression", "Random Forest", "XGBoost"]
            f1_scores = [0.88, 0.92, 0.92]

            df = pd.DataFrame({"Model": models, "F1": f1_scores})

            # Highlight the best bar
            best = df["F1"].max()
            df["color"] = df["F1"].apply(lambda x: "#6baed6" if x == best else "#b0b0b0")  # blue for best

            # Plotly Express bar chart
            fig = px.bar(
                df, x="Model", y="F1",
                text=df["F1"].map(lambda x: f"{x:.3f}"),
                color="color", color_discrete_map="identity",
                range_y=[0, 1.0]
            )

            fig.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>F1=%{y:.3f}<extra></extra>")
            fig.update_layout(showlegend=False, yaxis_title="F1-score", xaxis_title="Model", xaxis_tickangle=-90, margin=dict(b=80))

            # In a script: fig.show()
            # In Streamlit:
            # import streamlit as st
            column3, col4 = st.columns([1, 1])
            column3.plotly_chart(fig, use_container_width=True)

        elif selected_prediction_model == 'Random Forest':
            
            # perform_primary_rf_training(X_train_encoded, y_train, X_test_encoded, y_test, None)
            # cross_validate_rf_model(X_train_encoded, y_train, cross_validation, None)
            # hyperparameter_tune_rf(X_train_encoded, y_train, cross_validation)
            metrics = retrain_rf_model(X_train_encoded, y_train, X_test_encoded, y_test, 0.426, 609, 10, 3, 0.9, 28)
            st.subheader("Random Forest Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_rf_metrics(metrics_tab, metrics)
            get_internet_metrics_caption(column2)

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
            st.subheader("XGBoost Performance Analysis")
            metrics_tab, explainability_tab = st.tabs(["Metrics", "Explanation"])
            column2 = display_xgb_metrics(metrics_tab, row)
            get_internet_metrics_caption(column2)





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

