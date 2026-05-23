"""
run_pipeline.py
Runs the complete project pipeline in one shot:
  Step 1 – Generate synthetic vibration dataset
  Step 2 – Extract FFT + time-domain features
  Step 3 – Train Random Forest with 5-fold CV
  Step 4 – Generate all visualizations
  Step 5 – Demo inference on all 4 fault types
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data      import build_dataset
from feature_extraction import extract_all
from train_model        import train
from visualize          import (plot_time_domain, plot_fft,
                                plot_confusion_matrix, plot_feature_importance)
from predict            import simulate_signal, predict_signal

import pandas as pd

BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "data")
MODELS  = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

DIVIDER = "\n" + "="*60


print(DIVIDER)
print("  STEP 1/5 — Generating Vibration Dataset")
print("="*60)
df = build_dataset()
df.to_csv(os.path.join(DATA, "vibration_data.csv"), index=False)
print(f"  {len(df)} samples generated (300 × 4 classes)")

print(DIVIDER)
print("  STEP 2/5 — Extracting FFT + Time-Domain Features")
print("="*60)
feat_df = extract_all(
    os.path.join(DATA, "vibration_data.csv"),
    os.path.join(DATA, "features.csv")
)
print(f"  {len(feat_df.columns)-2} features extracted per sample")

print(DIVIDER)
print("  STEP 3/5 — Training Random Forest Classifier")
print("="*60)
metrics = train(
    os.path.join(DATA, "features.csv"),
    MODELS
)

print(DIVIDER)
print("  STEP 4/5 — Generating Visualizations")
print("="*60)
plot_time_domain()
plot_fft()
plot_confusion_matrix()
plot_feature_importance()

print(DIVIDER)
print("  STEP 5/5 — Live Inference Demo (all 4 fault types)")
print("="*60)
for fault in ["normal", "imbalance", "bearing", "misalignment"]:
    print(f"\n  ── Simulating: {fault.upper()} ──")
    sig = simulate_signal(fault)
    predict_signal(sig)

print(DIVIDER)
print("  PIPELINE COMPLETE")
print(f"  CV Accuracy : {metrics['cv_accuracy_mean']*100:.2f}% ± {metrics['cv_accuracy_std']*100:.2f}%")
print(f"  CV F1-Score : {metrics['cv_f1_mean']*100:.2f}%")
print(f"\n  Outputs     : {OUTPUTS}/")
print(f"  Model       : {MODELS}/rf_model.pkl")
print("="*60 + "\n")
