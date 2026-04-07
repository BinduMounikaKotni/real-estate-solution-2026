from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    Returns a dictionary of evaluation results.
    """

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nMODEL EVALUATION RESULTS")
    print("------------------------")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }