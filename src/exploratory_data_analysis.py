import plotly.express as px
import streamlit as st
import pandas as pd

#Custom color map
color_map = {
    'Yes': "#442020",
    1 : "#442020",
    'No': "#314A31",
    0: "#314A31"
}

def show_balance_between_yes_and_no(dataset):
    """
    Create and return a pie chart showing the distribution of churned
    vs non-churned customers.
    """
    dataset_copy = dataset.copy()

    # Standardize churn labels
    unique_values = set(dataset_copy['Churn'].unique())

    if unique_values <= {0, 1}:  # if only 0s and 1s
        dataset_copy['Churn'] = dataset_copy['Churn'].map({1: 'Yes', 0: 'No'})

    pie = px.pie(
        data_frame=dataset_copy,
        names='Churn',
        title='Churn Rate Review',
        color='Churn',
        color_discrete_map=color_map,
        width=500
    )

    return pie

