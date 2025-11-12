import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def encode_internet_data(dataset_to_encode):
    dataset_to_encode = dataset_to_encode.drop(columns=["id", "has_contract", "remaining_contract"])

    cat_cols = ["bill_bucket", "usage_tier", "fail_count_bucket", "contract_stage"]

    internet_data_encoded = pd.get_dummies(
        dataset_to_encode,
        columns=cat_cols,
        drop_first=True,
        dtype="int8"     # <- dummies come out numeric already
    )

    return internet_data_encoded

def split_internet_data(dataset):
    # Split the data
    X = dataset.drop(columns=["Churn"])
    y = dataset['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test

def get_scaled_internet_features(train_data, test_data):
    # Scaling numerical data
    scaler = StandardScaler()

    # Identify numeric columns
    train_numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    test_numeric_cols = test_data.select_dtypes(include=[np.number]).columns

    # Identify binary columns (only 0 and 1)
    train_binary_cols = [col for col in train_numeric_cols if train_data[col].dropna().isin([0, 1]).all()]
    test_binary_cols = [col for col in test_numeric_cols if test_data[col].dropna().isin([0, 1]).all()]

    # Continuous numeric columns (those that are not binary)
    train_continuous_cols = [col for col in train_numeric_cols if col not in train_binary_cols]
    test_continuous_cols = [col for col in test_numeric_cols if col not in test_binary_cols]

    train_scaled_continuous = scaler.fit_transform(train_data[train_continuous_cols])
    test_scaled_continuous = scaler.transform(test_data[test_continuous_cols])

    # Replace the continuous columns with their scaled versions
    train_features_scaled = train_data.copy()
    train_features_scaled[train_continuous_cols] = train_scaled_continuous

    test_features_scaled = test_data.copy()
    test_features_scaled[test_continuous_cols] = test_scaled_continuous

    return scaler, train_features_scaled, test_features_scaled, list(train_continuous_cols)

