"""
predict.py
Run inference on a new vibration signal.

Usage:
    python predict.py --mode simulate --fault normal
    python predict.py --mode simulate --fault bearing
    python predict.py --mode csv --file my_signal.csv

The --mode simulate option generates a synthetic signal so you can demo
without any hardware attached.
"""

import numpy as np
import pickle, os, argparse
import sys

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features

FS       = 1600
CLASS_NAMES = {0: "Normal", 1: "Blade_Imbalance", 2: "Bearing_Wear", 3: "Misalignment"}
SEVERITY    = {0: "✅ HEALTHY",
               1: "⚠️  WARNING – Blade Imbalance",
               2: "🔴 ALERT   – Bearing Wear Detected",
               3: "⚠️  WARNING – Shaft Misalignment"}
ACTION      = {
    0: "No action required. Continue routine monitoring.",
    1: "Schedule blade balance inspection within 7 days. Monitor vibration trend.",
    2: "URGENT: Inspect main bearing immediately. Risk of catastrophic failure.",
    3: "Check coupling alignment and foundation bolts within 48 hours.",
}

BASE = os.path.join(os.path.dirname(__file__), "..")


def load_model():
    with open(os.path.join(BASE, "models/rf_model.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(BASE, "models/scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return clf, scaler


def simulate_signal(fault: str) -> np.ndarray:
    """Generate a test signal — useful for demo without hardware."""
    t = np.linspace(0, 1.0, FS, endpoint=False)
    np.random.seed(np.random.randint(0, 9999))

    if fault in ("normal", "0"):
        s  = 0.30 * np.sin(2*np.pi*5*t)
        s += 0.15 * np.sin(2*np.pi*22*t)
        s += np.random.normal(0, 0.05, FS)
    elif fault in ("imbalance", "1"):
        s  = 0.80 * np.sin(2*np.pi*5*t + 0.3)
        s += 0.40 * np.sin(2*np.pi*10*t)
        s += np.random.normal(0, 0.08, FS)
    elif fault in ("bearing", "2"):
        s  = 0.25 * np.sin(2*np.pi*5*t)
        s += 0.35 * np.sin(2*np.pi*83*t)
        imp = np.random.choice(FS, 12, replace=False)
        s[imp] += np.random.uniform(1.5, 2.5, 12)
        s += np.random.normal(0, 0.10, FS)
    elif fault in ("misalignment", "3"):
        s  = 0.25 * np.sin(2*np.pi*5*t)
        s += 0.55 * np.sin(2*np.pi*44*t + 0.5)
        s += 0.40 * np.sin(2*np.pi*66*t)
        s += np.random.normal(0, 0.07, FS)
    else:
        raise ValueError(f"Unknown fault type: {fault}")
    return s


def predict_signal(signal: np.ndarray):
    clf, scaler = load_model()
    feats = extract_features(signal)

    feat_order = [
        "rms", "peak", "crest_factor", "kurtosis", "skewness",
        "variance", "mean_abs", "dom_freq", "dom_mag", "spec_centroid",
        "top_mag_1", "top_mag_2", "top_mag_3", "top_mag_4", "top_mag_5",
        "e_rotor_norm", "e_shaft_norm", "e_2shaft_norm", "e_3shaft_norm",
        "e_bpfo_norm", "e_bpfi_norm",
    ]
    X = np.array([[feats[k] for k in feat_order]])
    X_s = scaler.transform(X)

    pred  = clf.predict(X_s)[0]
    proba = clf.predict_proba(X_s)[0]

    print("\n" + "="*52)
    print("  WIND TURBINE FAULT DETECTION SYSTEM")
    print("  Nordex EDGE GET – Ashwani Rawat")
    print("="*52)
    print(f"\n  Prediction  : {CLASS_NAMES[pred]}")
    print(f"  Status      : {SEVERITY[pred]}")
    print(f"  Confidence  : {proba[pred]*100:.1f}%")
    print(f"\n  Action      : {ACTION[pred]}")
    print("\n  Class Probabilities:")
    for i, name in CLASS_NAMES.items():
        bar = "█" * int(proba[i] * 30)
        print(f"    {name:<20} {proba[i]*100:5.1f}%  {bar}")
    print("="*52 + "\n")
    return pred, proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["simulate","csv"], default="simulate")
    parser.add_argument("--fault", default="normal",
                        help="normal | imbalance | bearing | misalignment")
    parser.add_argument("--file",  default=None, help="CSV with one column 'signal'")
    args = parser.parse_args()

    if args.mode == "simulate":
        signal = simulate_signal(args.fault)
    else:
        import pandas as pd
        signal = pd.read_csv(args.file)["signal"].values

    predict_signal(signal)
