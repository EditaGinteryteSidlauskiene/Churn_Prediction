import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def map_telco_data_features(dataset):
    mapping = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}

    cols_to_map = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MultipleLines'
    ]

    for col in cols_to_map:
        if col == 'gender':
            dataset[col] = dataset[col].map({'Female': 1, 'Male': 0})
        else:
            dataset[col] = dataset[col].map(mapping)

def engineer_telco_data_features(dataset):
    # Feature engieering
    data_copy = dataset.copy()

    # But avoid division-by-zero during calculations
    data_copy['AvgMonthlyCharge'] = np.where(
        data_copy['tenure'] > 0,
        data_copy['TotalCharges'] / data_copy['tenure'],
        data_copy['MonthlyCharges']  # fallback for new customers
    )

    # Service bundle count (number of active services)
    service_cols = [
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    data_copy['OnlineServiceCount'] = data_copy[service_cols].sum(axis=1)

    # Long term contract indicator
    data_copy['IsLongTermContract'] = data_copy['Contract'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)

    # Average price per service (captures values perception)
    data_copy['AvgPricePerService'] = data_copy['MonthlyCharges'] / (data_copy['OnlineServiceCount'] + 1)
    return data_copy

def encode_telco_data(dataset_original, dataset_to_encode):
    # Drop customer id column
    dataset_to_encode = dataset_to_encode.drop(['customerID', 'Churn'], axis=1)
    dataset_to_encode.rename(columns={"gender": "Is_female"}, inplace=True)

    # Encode multi-category columns
    telco_data_encoded = pd.get_dummies(
        data=dataset_to_encode,
        columns=["Contract","PaymentMethod", "InternetService"],
        drop_first=True
    )

    # force dummies to int, leave other columns untouched
    dummy_cols = telco_data_encoded.columns.difference(dataset_original.columns)
    telco_data_encoded[dummy_cols] = telco_data_encoded[dummy_cols].replace([np.inf, -np.inf], np.nan)
    telco_data_encoded[dummy_cols] = telco_data_encoded[dummy_cols].fillna(0)

    telco_data_encoded[dummy_cols] = telco_data_encoded[dummy_cols].astype(int)

    return telco_data_encoded

def get_scaled_telco_features(train_data, test_data):
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

def split_telco_data(dataset):
    # Split the data
    X = dataset.drop(columns=["ChurnFlag"])
    y = dataset['ChurnFlag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test