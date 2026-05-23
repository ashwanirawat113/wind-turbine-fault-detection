"""
train_model.py
Trains a Random Forest classifier on the extracted FFT features.
  - Stratified 5-fold cross-validation
  - Final model trained on full training set
  - Saves model + scaler to /models/
  - Prints classification report + confusion matrix
"""

import numpy as np
import pandas as pd
import pickle, os, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

FEATURE_COLS = [
    "rms", "peak", "crest_factor", "kurtosis", "skewness",
    "variance", "mean_abs", "dom_freq", "dom_mag", "spec_centroid",
    "top_mag_1", "top_mag_2", "top_mag_3", "top_mag_4", "top_mag_5",
    "e_rotor_norm", "e_shaft_norm", "e_2shaft_norm", "e_3shaft_norm",
    "e_bpfo_norm", "e_bpfi_norm",
]
CLASS_NAMES = ["Normal", "Blade_Imbalance", "Bearing_Wear", "Misalignment"]
RANDOM_SEED = 42


def train(features_csv: str, model_dir: str):
    df = pd.read_csv(features_csv)
    X  = df[FEATURE_COLS].values
    y  = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # ── 5-fold cross-validation ───────────────────────────────
    print("\n── Stratified 5-Fold Cross-Validation ──")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv  = cross_validate(clf, X_scaled, y, cv=skf,
                         scoring=["accuracy", "f1_weighted"],
                         return_train_score=True)

    acc_mean = cv["test_accuracy"].mean()
    acc_std  = cv["test_accuracy"].std()
    f1_mean  = cv["test_f1_weighted"].mean()
    f1_std   = cv["test_f1_weighted"].std()

    print(f"  CV Accuracy : {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"  CV F1-Score : {f1_mean*100:.2f}% ± {f1_std*100:.2f}%")

    # ── train on full dataset ─────────────────────────────────
    print("\n── Training final model on full dataset ──")
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)

    train_acc = accuracy_score(y, y_pred)
    train_f1  = f1_score(y, y_pred, average="weighted")
    print(f"  Train Accuracy : {train_acc*100:.2f}%")
    print(f"  Train F1-Score : {train_f1*100:.2f}%")

    print("\n── Classification Report (full data) ──")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))

    print("── Confusion Matrix ──")
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(cm_df.to_string())

    # ── feature importance ────────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    print("\n── Top 10 Feature Importances ──")
    print(importances.head(10).to_string())

    # ── save artifacts ────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metrics = {
        "cv_accuracy_mean": round(acc_mean, 4),
        "cv_accuracy_std":  round(acc_std,  4),
        "cv_f1_mean":       round(f1_mean,  4),
        "cv_f1_std":        round(f1_std,   4),
        "train_accuracy":   round(train_acc, 4),
        "train_f1":         round(train_f1,  4),
        "feature_importance": importances.head(10).round(4).to_dict(),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n Model + scaler + metrics saved → {model_dir}")
    return metrics


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    train(
        os.path.join(base, "../data/features.csv"),
        os.path.join(base, "../models")
    )
