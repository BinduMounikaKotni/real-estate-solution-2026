import pickle
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model(X_train, y_train, model_path="Models/model.pkl"):
    """
    Train a RandomForest model and save it to disk.
    """

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model