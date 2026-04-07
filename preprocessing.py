import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(cleaned_df, final_df, pay_df):
    """
    Preprocess the final dataset for model training.
    Only final_df is used because it already contains
    engineered features and is aligned for modeling.

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Use final_df as the main dataset
    df = final_df.copy()

    # Target variable
    y = df["price"]

    # Feature matrix
    X = df.drop(columns=["price"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test