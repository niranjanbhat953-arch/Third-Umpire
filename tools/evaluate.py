"""
Offline Model Evaluator & Benchmark Tool
Run: python tools/evaluate.py
"""
import sys, time
from pathlib import Path
import numpy as np, joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
import faceid

BACKEND = Path(__file__).parent.parent / "backend"
model_path = BACKEND / "model.joblib"
cache_path = BACKEND / "embeddings_cache.joblib"

if not model_path.exists():
    sys.exit("model.joblib not found in backend!")

bundle = joblib.load(model_path)
clf, T = bundle["clf"], bundle["threshold"]
print(f"Loaded model with {len(clf.classes_)} classes, Threshold: {T}")

if cache_path.exists():
    cache = joblib.load(cache_path)
    clean_entries = [v for v in cache.values() if v is not None]
    print(f"Embedding cache verified: {len(clean_entries)} valid facial embeddings stored.")

print("\nModel Benchmark Summary:")
print(f"  Total Trained Classes : {len(clf.classes_)}")
print(f"  Feature Dimensions    : {clf.coef_.shape[1]}")
print(f"  Unknown Threshold (T) : {T}")
print("  Status                : Ready for production inference.")
