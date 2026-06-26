## Predictive Maintenance MLOps Platform

An end-to-end Machine Learning pipeline for predicting industrial machine failure, featuring automated model serving, real-time observability, and proactive alerting.
## 🚀 Overview

This project monitors industrial torque, rotational speed, and temperature to predict machine failure before it occurs. The infrastructure is designed for high availability on Kubernetes, with a fully integrated MLOps observability stack that monitors model performance and infrastructure health in real-time.
## 🏗️ Architecture

* Inference Engine: FastAPI

* Orchestration: Kubernetes (Minikube)

* Observability: Prometheus & Grafana

* Alerting: Alertmanager (Custom PrometheusRules)

## 🛠️ Key Features

* Real-time Inference: FastAPI-based REST API for millisecond-latency predictions.

* Custom Metrics: Built-in instrumentation for `prediction_latency`, `input_feature_mean_torque`, and `prediction_anomalies_total`.

* Proactive Monitoring: ** Real-time dashboard for visualizing feature drift and model latency.

        Automated HighTorqueAnomaly alerts that notify stakeholders via Alertmanager.

* Infrastructure as Code: Fully declarative monitoring configurations deployed directly into the cluster.

## 📁 Repository Structure
```

predictive-maintenance-mlops/
├── .github/workflows/  # CI/CD deployment pipelines
├── artifacts/          # Versioned model weights/scalers
├── k8s/                # Kubernetes deployment, HPA, and Service manifests
├── monitoring/         # Centralized observability configs
│   ├── alerts/         # Prometheus alert rules
│   └── prometheus-config.yaml
├── src/                # FastAPI source code & training scripts
└── Dockerfile          # Production container setup
```
## ⚙️ Quick Start
1. Deploy the Infrastructure

Deploy the API and observability stack to your Kubernetes cluster:
```bash

kubectl apply -f k8s/
kubectl apply -f monitoring/
```
2. Verify Observability

Access your Grafana dashboard to view real-time model telemetry:
```bash

kubectl port-forward -n default svc/grafana 3000:3000
```
3. Simulate Machine Stress

Use the following command to test the anomaly detection pipeline:
```bash

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature": 304.0, "process_temperature": 313.0, "rotational_speed": 1300, "torque": 95.0, "tool_wear": 210, "type_M": 0, "type_L": 1}'
```
## 📈 Observability & Alerts

The system is configured with a `HighTorqueAnomaly` alert rule. If the average torque exceeds 90 units, the system triggers a critical severity alert to the configured Alertmanager endpoint.

*Note:* Project developed as an MLOps reference implementation for predictive industrial maintenance.