import os, joblib, pandas as pd
from threading import Lock
from django.conf import settings

_PATH = os.path.join(settings.BASE_DIR, "models", "live_pipeline.pkl")
_MODEL = None
_LOCK  = Lock()

def get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                _MODEL = joblib.load(_PATH)
    return _MODEL

def predict_clusters(df: pd.DataFrame):
    bundle   = get_model()
    X        = df[bundle["columns"]]
    X_scaled = bundle["scaler"].transform(X)
    X_pca    = bundle["pca"].transform(X_scaled)
    clusters = bundle["kmeans"].predict(X_pca)
    df["cluster_label"] = clusters
    return df
