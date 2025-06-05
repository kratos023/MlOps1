import pandas as pd
import mlflow
import mlflow.sklearn # Ensure this is imported for mlflow.sklearn.log_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Set MLflow experiment name
mlflow.set_experiment("Iris_Classification")

# Load Data
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Definitions
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Create a directory to save local model files temporarily
os.makedirs("models", exist_ok=True)

best_model_name = None
best_model_score = 0.0
best_model_uri = None # This will store the MLflow URI of the best model's artifact

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)

        # Save model locally as a .pkl file
        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)

        # Log the locally saved .pkl file as an artifact
        # This makes the raw .pkl file accessible in the MLflow UI
        mlflow.log_artifact(model_path)

        # Log the model using mlflow.sklearn.log_model
        # This logs the model in MLflow's native format, making it easier to serve and deploy
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Check if the current model is the best so far
        if accuracy > best_model_score:
            best_model_score = accuracy
            best_model_name = model_name
            # Update best_model_uri to point to the MLflow logged model for THIS run
            best_model_uri = f"runs:/{run.info.run_id}/model" # 'model' is the artifact_path used in mlflow.sklearn.log_model
            print(f"New best model found: {best_model_name} with accuracy: {best_model_score:.4f}")

# --- IMPORTANT CHANGE: Register the best model AFTER the loop completes ---
# This ensures only the single best model from all runs is registered
if best_model_uri:
    print(f"\nRegistering the overall best model: {best_model_name}...")
    try:
        registered_model = mlflow.register_model(
            model_uri=best_model_uri,
            name="Best_Iris_Classifier_Model"
        )
        print(f"Successfully registered model version {registered_model.version} for '{registered_model.name}'.")
    except Exception as e:
        print(f"Error registering model: {e}")
else:
    print("No best model found to register.")


print(f"\nFinal Best model: {best_model_name} with accuracy: {best_model_score:.4f}")