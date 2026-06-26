FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# Ensure temporary/placeholder artifact workspace exists
RUN mkdir -p artifacts

# Fallback generation of dummy artifact to ensure container builds regardless of registry connection state
RUN python -c "import joblib, os; from sklearn.ensemble import RandomForestClassifier; import numpy as np; \
               os.makedirs('artifacts', exist_ok=True); \
               joblib.dump(RandomForestClassifier().fit(np.random.rand(5,7), [0,1,0,1,0]), 'artifacts/model.joblib')"

EXPOSE 8000

ENV PYTHONPATH=/app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]