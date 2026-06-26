import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def load_data():
    # Fetching directly from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    df = pd.read_csv(url)
    
    # Feature Engineering & Cleaning
    # Drop unique identifiers
    df = df.drop(columns=['UDI', 'Product ID'])
    
    # One-hot encode the 'Type' column (L, M, H quality variants)
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)
    
    # Target and features splits
    X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    y = df['Machine failure']
    
    return X, y

def train_and_track():
    mlflow.set_tracking_uri("http://localhost:5001") if os.getenv("LOCAL_MLFLOW") else mlflow.set_tracking_uri("local")
    mlflow.set_experiment("Predictive_Maintenance_RF")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler artifact to be bundled with serving layer
    os.makedirs("artifacts", exist_ok=True)
    import joblib
    joblib.dump(scaler, "artifacts/scaler.joblib")

    # Simple hyperparameter grid space
    param_grid = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10}
    ]

    for params in param_grid:
        with mlflow.start_run():
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"], 
                max_depth=params["max_depth"], 
                random_state=42,
                class_weight="balanced"
            )
            model.fit(X_train_scaled, y_train)
            
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            
            # Logging to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_artifact("artifacts/scaler.joblib")
            
            # Log the model instance
            mlflow.sklearn.log_model(model, "predictive_maintenance_model")
            joblib.dump(model, "artifacts/model.joblib")
            print(f"Logged run with n_estimators: {params['n_estimators']}, F1: {f1:.4f}")

if __name__ == "__main__":
    train_and_track()