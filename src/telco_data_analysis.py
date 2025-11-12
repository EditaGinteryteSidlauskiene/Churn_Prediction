import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

#Custom color map
color_map = {
    'Yes': "#442020",
    'No': "#314A31"
}

def display_select_boxes(sidebar):
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

def get_telco_data_analysis():
    """
    Provides a concise textual summary of the Techo Customer Churn dataset.
    """
    return 'The Telco Customer Churn dataset contains information on 7,043 customers, ' \
    'including demographic attributes (e.g., gender, senior citizen, dependents), account ' \
    'details (e.g., contract type, billing method, tenure), and service subscriptions ' \
    '(e.g., phone, internet, streaming).\nThe target variable is Churn, indicating whether a ' \
    'customer left the company.'

def get_telco_pie_caption():
    """
    Generates a caption describing the churn rate distribution.
    """
    return ('<br><br>The dataset shows a moderate class imbalance:'
    '<br>- 26.5% of customers have churned.'
    '<br>- 73.5% remain active.'
    '<br><br>This imbalance can bias machine learning models '
    'toward predicting the majority class ("No churn").')

def get_text_about_engineered_features():
    """
    Display an informational message about the engineered features in the Teclo Churn dataset.
    """
    st.info('''
            To enhance the model’s predictive performance and interpretability, feature engineering was applied
            to derive more informative attributes that better capture customer behavior patterns.
            
            **The newly created features include:**
            - **AvgMonthlyCharge** – calculated by dividing “TotalCharges” by “Tenure,” representing the average monthly expenditure per
            customer.
            - **OnlineServiceCount** – the total number of online services a customer subscribes to, reflecting their
            level of digital engagement. 
            - **IsLongServiceContract** – a binary indicator marking customers with contracts or tenures
            of at least one year, associated with stronger customer commitment. 
            - **AvgPriceService** – measuring the average cost per active service, helping identify customers with higher price sensitivity.
            ''')

def show_telco_numeric_drivers(dataset):

    """
    Visualizes numeric churn drivers in the Telco Customer Churn dataset.

    This function displays both histograms and boxplots for the main numeric variables
    (Tenure, MonthlyCharges, and TotalCharges) segmented by churn status. The visualizations
    help identify how these continuous features differ between churned and retained customers.
    """

    # Create three columns for histogram layout
    tenure_hist_col, monthly_charges_hist_col, total_charges_hist_col = st.columns([1, 1, 1])

    # Histogram of tenure distribution by churn
    tenure_hist = px.histogram(
        data_frame=dataset,
        x='tenure',
        color='Churn',
        color_discrete_map=color_map,
        nbins=50,
        histnorm="percent",
        title='Tenure Distribution by Churn Status',
        labels={
            "tenure": "Tenure (months)"
        }
    )

    # Histogram of monthly charges distribution by churn
    monthly_charges_hist = px.histogram(
        data_frame=dataset,
        x='MonthlyCharges',
        color='Churn',
        color_discrete_map=color_map,
        nbins=50,
        histnorm="percent",
        title='Monthly Charges Distribution by Churn Status',
        labels={
            "MonthlyCharges": "Monthly Charges ($)"
        }
    )

    # Histogram of total charges distribution by churn
    total_charges_hist = px.histogram(
        data_frame=dataset,
        x='TotalCharges',
        color='Churn',
        color_discrete_map=color_map,
        nbins=50,
        histnorm="percent",
        title='Total Charges Distribution by Churn Status',
        labels={
            "TotalCharges": "Total Charges ($)"
        }
    )

    # Update axis titles for uniformity
    tenure_hist.update_layout(yaxis_title='Percentage of Customers')
    monthly_charges_hist.update_layout(yaxis_title='Percentage of Customers')
    total_charges_hist.update_layout(yaxis_title='Percentage of Customers')
    
    # Display the histograms in Streamlit columns
    tenure_hist_col.plotly_chart(tenure_hist)
    monthly_charges_hist_col.plotly_chart(monthly_charges_hist)
    total_charges_hist_col.plotly_chart(total_charges_hist)

    # Caption summarizing insights from histograms
    st.caption(body='Churned customers cluster at shorter' \
    ' tenures and lower lifetime total chanrges, while showing ' \
    'higher monthly charges on average - indicating early-stage ' \
    'attrition and possible price sensitivity. Distributions ' \
    'are showed as percentages of customers.', unsafe_allow_html=True)
        
    # Create three columns for boxplot layout    
    tenure_box_col, monthly_charges_box_col, total_charges_box_col = st.columns([1, 1, 1])
    
    # Boxplots comparing churn groups
    tenure_box = px.box(
        data_frame=dataset,
        x=dataset['Churn'],
        y=dataset['tenure'],
        color='Churn',
        color_discrete_map=color_map,
        title='Tenure Distribution by Churn Status'
    )
    monthly_charges_box = px.box(
        data_frame=dataset,
        x=dataset['Churn'],
        y=dataset['MonthlyCharges'],
        color='Churn',
        color_discrete_map=color_map,
        title='Monthly Charges Distribution by Churn Status'
    )
    total_charges_box = px.box(
        data_frame=dataset,
        x=dataset['Churn'],
        y=dataset['TotalCharges'],
        color='Churn',
        color_discrete_map=color_map,
        title='Total Charges Distribution by Churn Status'
    )

    # Update Y-axis titles
    tenure_box.update_layout(yaxis_title='Tenure (Months)')
    monthly_charges_box.update_layout(yaxis_title='Monthly Charges ($)')
    total_charges_box.update_layout(yaxis_title='Total Charges ($)')

    # Display the boxplots
    tenure_box_col.plotly_chart(tenure_box)
    monthly_charges_box_col.plotly_chart(monthly_charges_box)
    total_charges_box_col.plotly_chart(total_charges_box)

    # Caption summarizing insights from boxplots
    st.caption(body='Customers who did not churn ("No") generally have longer tenures, higher total charges' \
    ', and slightly lower monthly charges than those who left ("Yes").' \
    ' This patterns suggests that loyal customers stay longer, accumulate more total charges over time,' \
    'and may benefit from more stable or lower monthly.)', unsafe_allow_html=True)

def show_telco_categorical_drivers(dataset):
        """
        Visualize categorical churn drivers in the Telco Customer Churn dataset.

        This function generates a set of bar charts to explore how different categorical
        variables relate to customer churn behavior. Each plot compares churned and 
        retained customers across several categorical dimensions such as contract type, 
        payment method, billing type, and internet service.
        """
        
        # --- First row of categorical charts: Contract and Payment Method ---
        categorical1, categorical2 = st.columns([1, 1])

        # Contract type vs churn
        contract_bar = px.bar(
            data_frame=dataset,
            x='Contract',
            color='Churn',
            color_discrete_map=color_map,
            title='Churn Rates by Contract Type'
        )

        contract_bar.update_layout(yaxis_title='Number of Customers')

        categorical1.plotly_chart(contract_bar)

        categorical1.caption(body='Customers on month-to-month contracts are far more likely to churn than those ' \
        'on one-year or two-year contracts. Longer contracts appear to reduce churn, likely because they lock '
        'in commitment and possibly offer discounts.', unsafe_allow_html=True)

        # Payment method vs churn
        payment_method_bar = px.bar(
            data_frame=dataset,
            x='PaymentMethod',
            color='Churn',
            color_discrete_map=color_map,
            title='Churn Rates by Payment Method'
        )

        contract_bar.update_layout(yaxis_title='Number of Customers')

        categorical2.plotly_chart(payment_method_bar)

        categorical2.caption(body='Churn is highest among customers using electronic checks, while those paying by ' \
        'credit card or bank transfer (automatic) churn less. This may reflect both customer demographics and ' \
        'the convenience/reliability of automated payments.', unsafe_allow_html=True)

        # --- Second row of categorical charts: Paperless Billing and Internet Service ---
        categorical3, categorical4 = st.columns([1, 1])

        # Paperless billing vs churn
        paperless_billing_bar = px.bar(
            data_frame=dataset,
            x='PaperlessBilling',
            color='Churn',
            color_discrete_map=color_map,
            title='Churn Rates by Paperless Billing',
            barmode='group'
        )

        paperless_billing_bar.update_layout(yaxis_title='Number of Customers')

        categorical3.plotly_chart(paperless_billing_bar)

        categorical3.caption(body='Customers with paperless billing have noticeably higher churn than those receiving paper bills. ' \
        'This might indicate differences in customer profiles: paperless billing users could be younger, more price-sensitive, '
        'or more likely to switch providers.', unsafe_allow_html=True)

        # Internet service vs churn
        internet_service_bar = px.bar(
            data_frame=dataset,
            x='InternetService',
            color='Churn',
            color_discrete_map=color_map,
            title='Churn Rates by Internet Service',
            barmode='group'
        )

        internet_service_bar.update_layout(yaxis_title='Number of Customers')

        categorical4.plotly_chart(internet_service_bar)

        categorical4.caption(body='Churn is most pronounced among Fiber optic users, while DSL customers show lower churn, '
        'and those without internet service churn the least. This suggests that fiber customers may have higher' \
        ' expectations for service or face more competitive alternatives.', unsafe_allow_html=True)

def show_telco_engagement_and_churn(dataset):
    """
    Visualize the relationship between customer engagement level and churn.

    This function analyzes how the number of subscribed services influences 
    customer churn. It calculates the churn rate for customers with different 
    counts of active services (e.g., OnlineSecurity, TechSupport, StreamingTV) 
    and visualizes the trend using a line chart.
    """

    # Create two columns: one for the chart, one for the explanatory note
    services_count_line_col, services_count_note_col = st.columns([5, 2])

    # Define the list of service-related columns to measure engagement
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines"
    ]
    
    # Compute the number of active services per customer ("Yes" values)
    dataset["ServicesCount"] = dataset[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

    # Group data by service count and compute churn rate percentage
    churn_rate_by_services = (
        dataset.groupby("ServicesCount")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)  # percentage
        .reset_index(name="ChurnRate")
    )
    
    # Create a line chart showing churn rate by number of subscribed services
    services_count_line = px.line(
        churn_rate_by_services,
        x="ServicesCount",
        y="ChurnRate",
        markers=True,
        title="Churn Rate by Number of Services",
        labels={
            "ServicesCount": "Number of Services Subscribed",
            "ChurnRate": "Churn Rate (%)"
        }
)
    # Display chart in Streamlit
    services_count_line_col.plotly_chart(services_count_line)

    # Add explanatory note next to the chart
    services_count_note_col.caption(body='<br><br>Customers with fewer subscribed services are more likely to churn, ' \
    'with churn rates peaking among those with only one or two services. As the number of services ' \
    'increases, the churn rate steadily declines, suggesting that broader service engagement creates ' \
    'stronger customer loyalty and reduces the likelihood of leaving.', unsafe_allow_html=True)

def show_telco_demographic_drivers(dataset):
    """
    Visualize churn rates across key demographic groups.

    This function computes churn rates (in %) for specified demographic features
    (e.g., SeniorCitizen, Partner, Dependents) and renders a grouped bar chart
    comparing categories within each feature (e.g., Yes vs No). It is designed
    for Streamlit dashboards and prints an explanatory caption alongside the chart.
    """

    # --- Layout setup: create two columns ---
    # Left column (wider) → bar chart
    # Right column (narrower) → explanatory text
    demographic_drivers_bar_col, demographic_drivers_note_col = st.columns([5, 2])

    # --- Add readable flag column for senior citizens ---
    # Original column is numeric (1 = Yes, 0 = No); map it to human-friendly labels.
    dataset['SeniorCitizenFlag'] = dataset['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    
    # --- Demographic columns to analyze ---
    # Each of these columns will be plotted separately.
    demo_cols = ["SeniorCitizenFlag", "Partner", "Dependents"]

    # --- Define consistent colors for categories ---
    # 'Yes' = orange tone, 'No' = green tone.
    color_map = {
    'Yes': "#a26c5c", 
    "No": "#69988C"           
    }
    
    # --- Compute churn rates by category for each demographic feature ---
    frames = []
    for col in demo_cols:
        # Group by category (e.g., Partner=Yes/No), calculate % churned.
        tmp = (
            dataset
            .groupby(col)["Churn"]
            .apply(lambda s: (s == "Yes").mean()*100)
            .reset_index()
            .rename(columns={col: "Category", "Churn": "ChurnRate"})
        )
        tmp["Feature"] = col
        frames.append(tmp)

    # Combine results for all demographic features into one table
    rates = pd.concat(frames, ignore_index=True)

    # --- Create grouped bar chart ---
    demographic_drivers_bar = px.bar(
        rates,
        x="Feature",
        y="ChurnRate",
        color="Category",
        color_discrete_map=color_map,
        barmode="group",
        text=rates["ChurnRate"].round(1).astype(str) + "%",
        title="Who Churns More (Churn Rate by Demographic Group)",
        labels={"Feature": "Demographic Feature", "ChurnRate": "Churn Rate (%)"}
    )

    # --- Improve appearance ---
    # Place percentage labels above the bars.
    demographic_drivers_bar.update_traces(textposition="outside")

    # Set y-axis range slightly above max churn rate (for visual breathing room).
    demographic_drivers_bar.update_layout(yaxis_range=[0, max(5, rates["ChurnRate"].max()*1.15)])

    # --- Render bar chart in the left column ---
    demographic_drivers_bar_col.plotly_chart(demographic_drivers_bar)

    # --- Add explanatory note in the right column ---
    # Note: The churn percentages here are hardcoded and might not match current data.
    # Consider computing them dynamically from `rates`.
    demographic_drivers_note_col.caption(body='<br><br>Demographic factors play an important role in churn behavior. ' \
    'Senior citizens show the highest churn rate (41.7%), indicating they are more likely to leave. ' \
    'In contrast, customers with a partner (19.7%) or dependents (15.5%) are less likely to churn ' \
    'compared to those without. This suggests that customers with family responsibilities tend to be ' \
    'more stable and loyal, while older customers may be more prone to switching providers.', unsafe_allow_html=True)

def show_telco_correlations_and_interactions(dataset):
    """
    Visualize numeric feature correlations and tenure-related churn behavior.

    This function generates two key visual analyses for telecom churn data:
    1. A **correlation heatmap** showing relationships among numeric variables
       (tenure, monthly charges, total charges, and churn flag).
    2. A **tenure bucket analysis** that illustrates how churn rates change
       as customer tenure increases.

    Both visuals are displayed in two side-by-side Streamlit columns.
    """

    # --- Layout setup: split page into two equal columns ---
    # Left column: correlation heatmap and interpretation.
    # Right column: churn–tenure relationship and discussion.
    corr_column1, corr_column2 = st.columns([1, 1])

    # --- Select numeric columns for correlation analysis ---
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]

    # --- Compute correlation matrix (Pearson correlation) ---
    # numeric_only=True ensures non-numeric columns are ignored.
    correlation_matrix = dataset[numeric_cols].corr(numeric_only=True)

    # --- Create correlation heatmap with Plotly ---
    # Each cell shows correlation between two numeric variables.
    # Blue → positive correlation, red → negative correlation.
    heatmap = px.imshow(
        correlation_matrix.round(2),
        text_auto=True,                  # show correlation values
        color_continuous_scale="icefire", # red = negative, blue = positive
        title="Correlation Heatmap of Numeric Features"
    )

    # Add axis labels for clarity.
    heatmap.update_layout(
        xaxis_title="Features",
        yaxis_title="Features"
    )

    # --- Display correlation heatmap in the left column ---
    corr_column1.plotly_chart(heatmap)

    # --- Add interpretive note under the heatmap ---
    # This static caption summarizes general observed relationships.
    # Ideally, in a production dashboard, these insights could be computed dynamically.
    corr_column1.caption(body='The heatmap shows strong positive correlations between tenure and total charges (0.82), ' \
    'as expected, since customers who stay longer accumulate higher charges. Monthly charges are moderately ' \
    'correlated with total charges (0.65) but less related to tenure (0.25). Importantly, churn flag is negatively ' \
    'correlated with tenure (-0.35), meaning longer-tenure customers are less likely to churn, while its correlation ' \
    'with monthly charges (0.19) suggests higher monthly fees slightly increase churn risk.', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECOND PART: Churn rate by tenure buckets
    # -------------------------------------------------------------------------

    # --- Define tenure ranges (buckets) for easier interpretation ---
    # Adjust bin edges if your data has higher or lower max tenure.
    bins = [0, 6, 12, 24, 48, 72]              
    labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]

    # --- Categorize customers into tenure buckets ---
    # pd.cut creates labeled intervals for analysis.
    dataset["TenureBucket"] = pd.cut(
        dataset["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    # --- Compute churn rate for each tenure bucket ---
    # (Number of churned customers / total in bucket) × 100.
    # Assumes 'Churn' column contains "Yes"/"No" values.
    bucket_rates = (
        dataset.groupby("TenureBucket", observed=True)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    # --- Ensure buckets display in correct order ---
    bucket_rates["TenureBucket"] = pd.Categorical(bucket_rates["TenureBucket"], categories=labels, ordered=True)
    bucket_rates = bucket_rates.sort_values("TenureBucket")

    # --- (a) Create a bar chart to visualize churn rates ---
    # Each bar = one tenure range; height = churn percentage.
    fig_bar = px.bar(
        bucket_rates,
        x="TenureBucket",
        y="ChurnRate",
        text=bucket_rates["ChurnRate"].round(1).astype(str) + "%",
        title="Churn Rate by Tenure Bucket",
        labels={"TenureBucket": "Tenure (months)", "ChurnRate": "Churn Rate (%)"},
    )

    # Place percentage labels above each bar.
    fig_bar.update_traces(textposition="outside")

    # Add headroom above the tallest bar for readability.
    fig_bar.update_layout(yaxis_range=[0, max(5, bucket_rates["ChurnRate"].max() * 1.15)])

    # Note: .show() opens a separate Plotly window in some environments.
    # In Streamlit, this line can be omitted; use st.plotly_chart instead.
    fig_bar.show()

    # --- (b) Create a line chart to highlight churn trend ---
    # Helpful for showing the downward trajectory of churn as tenure increases.
    fig_line = px.line(
        bucket_rates,
        x="TenureBucket",
        y="ChurnRate",
        markers=True,
        title="Churn Rate by Tenure Bucket (Trend)",
        labels={"TenureBucket": "Tenure (months)", "ChurnRate": "Churn Rate (%)"},
    )

    # Match Y-axis scale with bar chart for consistent interpretation.
    fig_line.update_layout(yaxis_range=[0, max(5, bucket_rates["ChurnRate"].max() * 1.15)])

    # --- Render the line chart in the right-hand column ---
    corr_column2.plotly_chart(fig_line)

    # --- Add interpretive caption for the trend chart ---
    corr_column2.caption(body='Churn risk decreases sharply as customer tenure increases. Customers in their first ' \
    '6 months churn at over 50%, while those with 1–2 years of tenure churn around 30%. After 2 years, ' \
    'churn steadily declines, reaching under 10% for customers with more than 4 years of tenure. This highlights ' \
    'the importance of retention strategies in the early stages of the customer lifecycle, where churn risk is highest.', unsafe_allow_html=True)

def get_correlation_heatmap_with_engineered_features(telco_data_encoded):

    corr_features = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'ChurnFlag',
    'AvgMonthlyCharge', 'OnlineServiceCount',
    'IsLongTermContract', 'AvgPricePerService'
    ]

    # Keep only those columns that exist (in case of missing)
    corr_features = [f for f in corr_features if f in telco_data_encoded.columns]

    # Compute correlation matrix
    corr_matrix = telco_data_encoded[corr_features].corr(numeric_only=True).round(2)

    # Plot
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="icefire",
        title="Correlation Heatmap of Engineered Numeric Features",
        width=900,
        height=500
    )
    fig.update_layout(xaxis_title="Features", yaxis_title="Features")
    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(fig)
    col2.caption(body='<br><br>The correlation heatmap reveals that long-term contracts are strongly associated with ' \
    'customer retention, whereas higher per-service and monthly charges show moderate positive ' \
    'correlations with churn. These engineered variables capture deeper behavioral and financial ' \
    'patterns – such as commitment, engagement, and cost sensitivity – offering richer insights ' \
    'beyond the base relationships found in “Tenure” and “Charges” alone.', unsafe_allow_html=True)


