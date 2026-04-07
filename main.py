# main.py
from Source.loader_data import load_final_df
from Source.preprocessing import preprocess_data
from Source.model_training import train_and_save_model
from Source.model_evaluation import evaluate_model

print("\n--- STEP 1: Loading Data ---")
final_df = load_final_df()
print("Data loaded successfully")

print("\n--- STEP 2: Preprocessing ---")
X_train, X_test, y_train, y_test = preprocess_data(None, final_df, None)
print("Preprocessing complete")

print("\n--- STEP 3: Training Model ---")
model = train_and_save_model(X_train, y_train)
print("Model training complete and saved")

print("\n--- STEP 4: Evaluating Model ---")
results = evaluate_model(model, X_test, y_test)
print("Evaluation complete")