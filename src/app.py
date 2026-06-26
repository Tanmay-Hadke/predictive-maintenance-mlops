import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator    

app = FastAPI(title="Predictive Maintenance Inference Service")

# Initialize the instrumentator
instrumentator = Instrumentator()

# Hook it up to your FastAPI app
instrumentator.instrument(app).expose(app)

# Prometheus Metrics definition
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests.")
ANOMALY_COUNT = Counter("prediction_anomalies_total", "Total number of failure anomalies predicted.")
LATENCY_HISTOGRAM = Histogram("prediction_latency_seconds", "Inference latency distribution.")
DRIFT_GAUGE = Gauge("input_feature_mean_torque", "Running monitor for feature distribution shift (Torque).")

# Mock local load or download from MLflow registry
# For portability, we load a pre-trained fallback if artifact paths aren't overridden
try:
    model = joblib.load("artifacts/model.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
except FileNotFoundError:
    # Fallback to dummy model for container initialization tests
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier().fit(np.random.rand(10, 7), np.random.randint(0, 2, 10))
    scaler = joblib.load("artifacts/scaler.joblib") if os.path.exists("artifacts/scaler.joblib") else None

class InputData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type_M: int
    type_L: int

@app.get("/healthz")
def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(data: InputData):
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    # Monitor input distribution drift
    DRIFT_GAUGE.set(data.torque)
    
    # Prepare payload structured matching training shape
    # Prepare payload structured matching training shape
    input_df = pd.DataFrame([{
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
        "Type_L": data.type_L,  # <-- Type_L is now here
        "Type_M": data.type_M   # <-- Type_M is now here
    }])
    
    if scaler:
        features = scaler.transform(input_df)
    else:
        features = input_df.to_numpy()
        
    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][prediction])
    
    if prediction == 1:
        ANOMALY_COUNT.inc()
        
    LATENCY_HISTOGRAM.observe(time.time() - start_time)
    
    return {
        "machine_failure_prediction": prediction,
        "confidence": confidence
    }