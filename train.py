import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- ADD THESE PRINT STATEMENTS FOR DIAGNOSTICS ---
print(f"Current Working Directory: {os.getcwd()}")
print(f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI', 'Not Set')}")
print(f"MLFLOW_ARTIFACT_URI: {os.environ.get('MLFLOW_ARTIFACT_URI', 'Not Set')}")
# --- END DIAGNOSTIC PRINTS ---

#Set Experiment
mlflow.set_experiment("Iris_Classification")

#Load Data
df = pd.read_csv("data/iris.csv")
x = df.drop("target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

#Model Definitions
models ={
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

os.makedirs("models", exist_ok=True)
best_model_name  = None
best_model_score = 0.0
best_model_uri   = None

for model_name , model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)

        #Save and Log Artifact
        model_path = f"models/{model_name}.pkl"

        # --- ADD THIS PRINT STATEMENT HERE ---
        print(f"Attempting to save and log artifact from: {model_path}")
        # --- END DIAGNOSTIC PRINT ---

        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        #Log the model in MLflow

        mlflow.sklearn.log_model(model, artifact_path="model")
        if acc>best_model_score:
            best_model_score=acc
            best_model_name = model_name
            best_model_uri=f"runs:/{run.info.run_id}/model"

if best_model_uri:
    result=mlflow.register_model(
        model_uri=best_model_uri,
        name="Best_Iris_Model"
    )
    print(f"Registered best model:{best_model_name}(accuracy:{best_model_score:.4f})")