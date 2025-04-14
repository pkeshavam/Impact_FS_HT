import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(data_dir, filename):
    """
    Load dataset from a specified CSV file in the data directory.
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def preprocess_data(data, target_column):
    """
    Preprocess the data:
    - Separate features and target.
    - Split into train/test sets.
    - Scale the feature values.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is not in the dataset.")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test



