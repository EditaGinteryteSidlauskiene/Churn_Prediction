import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

colors = {
            "download_avg": "#0072B2",  
            "upload_avg":   "#D55E00",  
        }

#Custom color map
color_map = {
    'Yes': "#442020",
    'No': "#314A31"
}

def get_internet_data_analysis():
    return 'The Internet Service Customer Churn dataset contains information on 72,274 ' \
    'customers, capturing details about subscription type, billing, service usage, and ' \
    'contract status. It includes indicators of additional services such as TV and movie ' \
    'packages, measures of customer tenure (subscription age), and typical billing levels ' \
    '(average bill). Service quality and usage are reflected in features like service failure ' \
    'count, average download and upload volumes, and whether customers exceeded their data limit.\n' \
    'A key contractual attribute is remaining contract, which records the number of months left on ' \
    'fixed-term agreements. Notably, almost 30% of the values in this column are missing, which means ' \
    'that the customers does not have a contract. The target variable ' \
    'is Churn, encoded as a binary flag showing whether a customer left the service provider.'

def get_internet_dataset_null_percentage(dataset_column):
    return round(dataset_column.isna().mean() * 100, 2)

def get_internet_data_null_heatmap(dataset):
    # make the plot
    figure, axes = plt.subplots()
    #figure.set_size_inches(10, 10)
    
    sns.heatmap(
        dataset.isnull(), 
        yticklabels=False, 
        cbar=False, 
        cmap='viridis',
        ax=axes)
    
    axes.set_title('Missing Values Heatmap', fontdict={'fontsize': 10, 'fontweight': 'bold'}, pad=5)

    return figure, axes

def get_internet_data_missing_values_caption(
        missing_download_avg_percentage, 
        missing_remaining_contract_percentage):
    return (f'Heatmap of missing values across features: most missingness is in remaining_contract ' \
    f'({missing_remaining_contract_percentage}%), while download_avg and upload_avg each have only ' \
    f'{missing_download_avg_percentage}% missing.<br>'
    f'Distribution of download_avg vs upload_avg (trimmed ' \
    f'at the 99th percentile) shows heavy right-skew and the expected download≫upload pattern.<br>'
    f'Because ' \
    f'download/upload are primary usage signals and their distributions are skewed, mean imputation could ' \
    f'distort the density and the download-to-upload ratio. Given the very small missing rate ({missing_download_avg_percentage}%),' \
    f' removing those rows has negligible impact on sample size and preserves the original distributional relationships. ' \
    f'Missingness in remaining_contract is likely informative, so it is retained via a missing-indicator feature.')

def get_internet_data_null_hist(dataset):
    # put both columns into one long column + label
        long = dataset[['download_avg', 'upload_avg']].melt(
            var_name='metric', value_name='value'
        ).dropna()

        # zoom x-axis to the central mass (change p if you want)
        p = 0.99
        xmax = long['value'].quantile(p)

        fig = px.histogram(
            long, x='value', color='metric',
            nbins=100,                 # more bars
            range_x=[0, xmax],         # avoid the extreme tail
            histnorm='percent',         # y-axis in %
            color_discrete_map=colors
        )

        fig.update_xaxes(tickmode='linear', tick0=0, dtick=20)

        fig.update_layout(
            title=f"Download vs Upload Usage Distribution (≤ {int(p*100)}th percentile)",
            xaxis_title="Value",
            yaxis_title="Percent",
            barmode='overlay'          # overlay both series
        )
        fig.update_traces(opacity=0.6, marker_line_color="white", marker_line_width=0.5) # see overlap better
        fig.update_yaxes(ticksuffix='%')

        return fig

def get_internet_pie_caption():
    """
    Generates a caption describing the churn rate distribution.
    """
    return ('<br><br>- 55.4% of customers have churned.'
    '<br>- 44.6% remain active.'
    '<br><br>This relatively high churn proportion indicates a potential retention problem within the customer base.')

def engineer_internet_features(dataset):
    data_copy = dataset.copy()

    data_copy["has_contract"] = data_copy["remaining_contract"].notnull().astype(int)

    # --- Build contract_stage ---
    data_copy["contract_stage"] = pd.Series(index=data_copy.index, dtype="object")

    # 1) No contract
    data_copy.loc[data_copy["has_contract"] == 0, "contract_stage"] = "no_contract"

    # 2) Positive remaining years → bucket
    mask_pos = (data_copy["has_contract"] == 1) & (data_copy["remaining_contract"] > 0)
    data_copy.loc[mask_pos, "contract_stage"] = pd.cut(
        data_copy.loc[mask_pos, "remaining_contract"],
        bins=[0, 0.5, 1, 2, np.inf],
        labels=["0-0.5y", "0.5-1y", "1-2y", ">2y"],
        right=True
    ).astype(str)

    # --- 1) Define bill buckets ---
    # Example boundaries: adjust as needed for your dataset’s scale
    data_copy["bill_bucket"] = pd.cut(
        data_copy["bill_avg"],   # or "TotalCharges", depending on your metric
        bins=[0, 20, 50, 80, 120, 200, np.inf],
        labels=["<20", "20–50", "50–80", "80–120", "120–200", ">200"],
        right=False
    ).astype(str)

    data_copy["bill_bucket"] = pd.Categorical(data_copy["bill_bucket"], categories=["<20", "20–50", "50–80", "80–120", "120–200", ">200"], ordered=True)

    # --- 1) Compute total usage and define tiers ---
    data_copy["total_usage"] = data_copy["download_avg"] + data_copy["upload_avg"]

    # Use quantile-based bins (or fixed thresholds if you prefer)
    # Example with fixed thresholds:
    data_copy["usage_tier"] = pd.cut(
        data_copy["total_usage"],
        bins=[0, 50, 200, np.inf],
        labels=["Light", "Medium", "Heavy"],
        right=False
    ).astype(str)

    data_copy["usage_tier"] = pd.Categorical(data_copy["usage_tier"], categories=["Light", "Medium", "Heavy"], ordered=True)

    # --- 1) Define fail count buckets ---
    data_copy["fail_count_bucket"] = pd.cut(
        dataset["service_failure_count"],
        bins=[-1, 0, 1, 3, np.inf],
        labels=["0", "1", "2–3", "4+"],
        right=True
    ).astype(str)

    # Preserve category order for x-axis
    data_copy["fail_count_bucket"] = pd.Categorical(data_copy["fail_count_bucket"], categories=["0", "1", "2–3", "4+"], ordered=True)

    # Define the list of service-related columns to measure engagement
    service_cols = [
        "is_tv_subscriber",
        "is_movie_package_subscriber"
    ]
    
    # Compute the number of active services per customer ("Yes" values)
    data_copy["services_count"] = data_copy[service_cols].apply(lambda row: (row == 1).sum(), axis=1)

    return data_copy



def show_internet_numeric_drivers(dataset):
    # Create three columns for histogram layout
    subscription_age_hist_col, bill_avg_hist_col, service_failure_count_hist_col = st.columns([1, 1, 1])

    # Histogram of tenure distribution by churn
    subscription_age_hist = px.histogram(
        data_frame=dataset,
        x='subscription_age',
        color='Churn',
        color_discrete_map=color_map,
        nbins=50,
        histnorm="percent",
        title='Tenure Distribution by Churn Status',
        labels={
            "subscription_age": "Tenure (years)"
        }
    )

    # Histogram of monthly charges distribution by churn
    bill_avg_hist = px.histogram(
        data_frame=dataset,
        x='bill_avg',
        color='Churn',
        color_discrete_map=color_map,
        nbins=50,
        histnorm="percent",
        title='Average Bill Distribution by Churn Status',
        labels={
            "bill_avg": "Average bill for the past 3 months ($)"
        }
    )

    # Histogram of total charges distribution by churn
    service_failure_count_hist = px.histogram(
        data_frame=dataset,
        x='service_failure_count',
        color='Churn',
        color_discrete_map=color_map,
        nbins=50,
        histnorm="percent",
        title='Total Failed Services Distribution by Churn Status',
        labels={
            "service_failure_count": "Total failed services"
        }
    )

    # Update axis titles for uniformity
    subscription_age_hist.update_layout(yaxis_title='Percentage of Customers')
    bill_avg_hist.update_layout(yaxis_title='Percentage of Customers')
    service_failure_count_hist.update_layout(yaxis_title='Percentage of Customers')
    
    # Display the histograms in Streamlit columns
    subscription_age_hist_col.plotly_chart(subscription_age_hist)
    bill_avg_hist_col.plotly_chart(bill_avg_hist)
    service_failure_count_hist_col.plotly_chart(service_failure_count_hist)

    # Caption summarizing insights from histograms
    st.caption(body='The three distributions suggest that churn is driven primarily by service reliability, ' \
    'pricing, and tenure. Customers with more failed services are disproportionately likely to churn. ' \
    'Higher average bills are also associated with a greater share of churned customers, indicating price ' \
    'sensitivity. Finally, churn is most common among newer customers and decreases with longer tenure, ' \
    'underscoring the value of early retention efforts.', unsafe_allow_html=True)
        
    # Create three columns for boxplot layout    
    subscription_age_box_col, bill_avg_box_col, service_failure_count_box_col = st.columns([1, 1, 1])
    
    # Boxplots comparing churn groups
    subscription_age_box = px.box(
        data_frame=dataset,
        x=dataset['Churn'],
        y=dataset['subscription_age'],
        color='Churn',
        color_discrete_map=color_map,
        title='Tenure Distribution by Churn Status'
    )
    bill_avg_box = px.box(
        data_frame=dataset,
        x=dataset['Churn'],
        y=dataset['bill_avg'],
        color='Churn',
        color_discrete_map=color_map,
        title='Average Bill Distribution by Churn Status'
    )
    service_failure_count_box = px.box(
        data_frame=dataset,
        x=dataset['Churn'],
        y=dataset['service_failure_count'],
        color='Churn',
        color_discrete_map=color_map,
        title='Total Failed Services Distribution by Churn Status'
    )

    # Update Y-axis titles
    subscription_age_box.update_layout(yaxis_title='Tenure (Years)')
    bill_avg_box.update_layout(yaxis_title='Average Bill For The Last Three Months ($)')
    service_failure_count_box.update_layout(yaxis_title='Total Failed Services')

    # Display the boxplots
    subscription_age_box_col.plotly_chart(subscription_age_box)
    bill_avg_box_col.plotly_chart(bill_avg_box)
    service_failure_count_box_col.plotly_chart(service_failure_count_box)

    # Caption summarizing insights from boxplots
    st.caption(body='Churned customers tend to have more failed services and higher average bills,' \
    ' with wider spreads and higher maxima than retained customers. They also have shorter tenure, ' \
    'indicating that service reliability, pricing, and early-lifecycle risk are key drivers of churn.', 
    unsafe_allow_html=True)

def show_internet_categorical_drivers(dataset):
    # --- Stacked counts by contract_stage & churn (to match the picture) ---
    # (Assumes Churn is 0/1; tweak the mapping if it's already "Yes"/"No")
    churn_label = {0: "No", 1: "Yes"}
    dataset["churn_str"] = dataset["Churn"].map(churn_label).fillna(dataset["Churn"])

    counts = (
        dataset.dropna(subset=["contract_stage"])
               .groupby(["contract_stage", "churn_str"], observed=True)
               .size()
               .reset_index(name="n")
    )

    # preserve desired order on x and legend (No first, then Yes)
    counts["contract_stage"] = pd.Categorical(counts["contract_stage"], categories=["no_contract", "0-0.5y", "0.5-1y", "1-2y", ">2y"], ordered=True)
    legend_order = ["No", "Yes"]
    counts["churn_str"] = pd.Categorical(counts["churn_str"], categories=legend_order, ordered=True)

    fig = px.bar(
        counts,
        x="contract_stage",
        y="n",
        color="churn_str",
        color_discrete_map=color_map,
        barmode="stack",
        category_orders={"contract_stage": ["no_contract", "0-0.5y", "0.5-1y", "1-2y", ">2y"], "churn_str": legend_order},
        labels={"contract_stage": "Contract Stage", "n": "Number of Customers", "churn_str": "Churn"},
        title="Churn by Contract Stage",
    )

    # optional: small visual tweaks similar to the example
    fig.update_layout(
        bargap=0.25,
        legend_title_text="Churn",
    )
    fig.update_yaxes(tickformat=",")  # comma separators like 1,000

    categorical1_col, categorical2_col = st.columns([1, 1])
    categorical1_col.plotly_chart(fig, use_container_width=True)

    # --- 2) Prepare churn labels ---
    churn_label = {0: "No", 1: "Yes"}
    dataset["churn_str"] = dataset["Churn"].map(churn_label).fillna(dataset["Churn"])

    # --- 3) Aggregate counts ---
    counts = (
        dataset.dropna(subset=["bill_bucket"])
               .groupby(["bill_bucket", "churn_str"], observed=True)
               .size()
               .reset_index(name="n")
    )

    legend_order = ["No", "Yes"]
    counts["churn_str"] = pd.Categorical(counts["churn_str"], categories=legend_order, ordered=True)

    # --- 4) Plot stacked bar ---
    # Preserve order for consistent plotting

    fig = px.bar(
        counts,
        x="bill_bucket",
        y="n",
        color="churn_str",
        color_discrete_map=color_map,
        barmode="stack",
        category_orders={"bill_bucket": ["<20", "20–50", "50–80", "80–120", "120–200", ">200"], "churn_str": legend_order},
        labels={"bill_bucket": "Monthly Bill ($)", "n": "Number of Customers", "churn_str": "Churn"},
        title="Churn by Bill Bucket",
    )

    fig.update_layout(
        bargap=0.25,
        legend_title_text="Churn",
    )
    fig.update_yaxes(tickformat=",")
    categorical2_col.plotly_chart(fig, use_container_width=True)

    categorical1_caption_col, categorical2_caption_col = st.columns([1, 1])
    categorical1_caption_col.caption(body='Customers without a contract exhibit the highest churn, while churn rates ' \
    'drop sharply among those with longer active contracts, particularly beyond one year.', unsafe_allow_html=True)
    categorical2_caption_col.caption(body='Churn is most prevalent among customers with lower bills (under $50), while ' \
    'those paying higher monthly charges show minimal churn, indicating that higher-value customers tend to remain more loyal.',
    unsafe_allow_html=True)

    # --- 2) Prepare churn labels ---
    churn_label = {0: "No", 1: "Yes"}
    dataset["churn_str"] = dataset["Churn"].map(churn_label).fillna(dataset["Churn"])

    # --- 3) Aggregate counts ---
    counts = (
        dataset.dropna(subset=["usage_tier"])
               .groupby(["usage_tier", "churn_str"], observed=True)
               .size()
               .reset_index(name="n")
    )

    legend_order = ["No", "Yes"]
    counts["churn_str"] = pd.Categorical(counts["churn_str"], categories=legend_order, ordered=True)

    # --- 4) Plot stacked bar ---
    fig = px.bar(
        counts,
        x="usage_tier",
        y="n",
        color="churn_str",
        color_discrete_map=color_map,
        barmode="stack",
        category_orders={"usage_tier": ["Light", "Medium", "Heavy"], "churn_str": legend_order},
        labels={"usage_tier": "Usage Tier", "n": "Number of Customers", "churn_str": "Churn"},
        title="Churn by Usage Tier",
    )

    fig.update_layout(
        bargap=0.25,
        legend_title_text="Churn",
    )
    fig.update_yaxes(tickformat=",")

    categorical3_col, categorical4_col = st.columns([1, 1])
    categorical3_col.plotly_chart(fig, use_container_width=True)

    counts = (dataset.groupby(["fail_count_bucket", "churn_str"], observed=True)
                 .size().reset_index(name="n"))


    counts = counts.set_index(["fail_count_bucket","churn_str"]).reindex(
    pd.MultiIndex.from_product([["0", "1", "2–3", "4+"], ["No","Yes"]], names=["fail_count_bucket","churn_str"]),
    fill_value=0
    ).reset_index()

    # 4) Plot (force categorical axis)
    fig = px.bar(
        counts, x="fail_count_bucket", y="n", 
        color="churn_str",
        color_discrete_map=color_map,
        barmode="stack", 
        category_orders={"fail_count_bucket": ["0", "1", "2–3", "4+"], 
                         "churn_str": ["No","Yes"]},
        labels={"fail_count_bucket":"Number of Failed Services","n":"Number of Customers","churn_str":"Churn"},
        title="Churn by Fail Count Bucket"
    )

    fig.update_xaxes(type="category", categoryorder="array", categoryarray=["0", "1", "2–3", "4+"])
    categorical4_col.plotly_chart(fig, use_container_width=True)

    categorical3_caption_col, categorical4_caption_col = st.columns([1, 1])
    categorical3_caption_col.caption(body='Light users show the highest churn rate, suggesting ' \
    'lower engagement or service dependence, while medium and heavy users demonstrate greater retention, ' \
    'likely due to stronger reliance on the service.', unsafe_allow_html=True)

    categorical4_caption_col.caption(body='Most churned customers fall in the “0” failures bucket simply ' \
    'because most customers have no failures (it’s the largest group). The total counts drop sharply for ' \
    '1, 2–3, and 4+ failures; however, the share of churn within each bucket is higher in the higher-failure ' \
    'buckets, indicating reliability issues are associated with greater churn risk.', unsafe_allow_html=True)

    # Then drop it if you like
    dataset.drop(columns=["churn_str"], inplace=True)

def show_internet_engagement_and_churn(dataset):

    # Create two columns: one for the chart, one for the explanatory note
    services_count_line_col, services_count_note_col = st.columns([5, 2])

    # Group data by service count and compute churn rate percentage
    churn_rate_by_services = (
        dataset.groupby("services_count")["Churn"]
        .apply(lambda x: (x == 1).mean() * 100)  # percentage
        .reset_index(name="ChurnRate")
    )
    
    # Create a line chart showing churn rate by number of subscribed services
    services_count_line = px.line(
        churn_rate_by_services,
        x="services_count",
        y="ChurnRate",
        markers=True,
        title="Churn Rate by Number of Services",
        labels={
            "services_count": "Number of Services Subscribed",
            "ChurnRate": "Churn Rate (%)"
        }
)
    # Display chart in Streamlit
    services_count_line_col.plotly_chart(services_count_line)

    # Add explanatory note next to the chart
    services_count_note_col.caption(body='<br><br>Customers with no additional services exhibit the ' \
    'highest churn rate, while those subscribed to one or two services are significantly more likely to ' \
    'remain, indicating that broader service engagement enhances customer retention.', unsafe_allow_html=True)


def show_internet_correlations_and_interactions(dataset):

    # --- Layout setup: split page into two equal columns ---
    # Left column: correlation heatmap and interpretation.
    # Right column: churn–tenure relationship and discussion.
    corr_column1, corr_column2 = st.columns([1, 1])

    # --- Select numeric columns for correlation analysis ---
    numeric_cols = ["subscription_age", "bill_avg", "remaining_contract", "service_failure_count","download_over_limit",
                    "total_usage", "services_count", "Churn"]

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
        title="Correlation Heatmap of Engineered Numeric Features",
        width=900,
        height=500
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
    corr_column2.caption(body='<br><br><br><br><br><br>The heatmap illustrates relationships among key numerical variables and churn. ' \
    'Strong negative correlations are observed between remaining_contract and Churn, as well as between services_count '
    'and Churn, indicating that customers with longer contracts or more subscribed services are less likely to churn. ' \
    'In contrast, most other features show weak or minimal correlations with churn.', unsafe_allow_html=True)

