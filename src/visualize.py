"""
visualize.py
Generates 4 publication-quality figures saved to /outputs/:
  1. time_domain_signals.png   – raw waveforms for all 4 classes
  2. fft_spectra.png           – FFT magnitude spectra with fault markers
  3. confusion_matrix.png      – confusion matrix heatmap
  4. feature_importance.png    – top feature importances bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import rfft, rfftfreq
import json, os

BASE    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FS       = 1600
COLORS   = {"Normal":"#2ecc71","Blade_Imbalance":"#f39c12",
            "Bearing_Wear":"#e74c3c","Misalignment":"#3498db"}
CLASS_NAMES = ["Normal","Blade_Imbalance","Bearing_Wear","Misalignment"]


def load_one_sample(class_name):
    df   = pd.read_csv(os.path.join(BASE, "data/vibration_data.csv"))
    row  = df[df["class_name"] == class_name].iloc[0]
    cols = [c for c in df.columns if c.startswith("t")]
    return row[cols].values.astype(float)


# ── Fig 1: Time-domain waveforms ────────────────────────────
def plot_time_domain():
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Wind Turbine Vibration Signals – Time Domain\n(Simulated ADXL345 Accelerometer Data)",
                 fontsize=13, fontweight="bold", y=1.01)
    t = np.linspace(0, 1.0, FS, endpoint=False)[:800]   # show 0.5 s

    for ax, name in zip(axes, CLASS_NAMES):
        sig = load_one_sample(name)[:800]
        ax.plot(t, sig, color=COLORS[name], linewidth=0.8)
        ax.set_ylabel("Accel (g)", fontsize=9)
        ax.set_title(name.replace("_", " "), fontsize=10, fontweight="bold",
                     color=COLORS[name], loc="left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3.5, 3.5)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "time_domain_signals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Fig 2: FFT spectra ──────────────────────────────────────
def plot_fft():
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("FFT Magnitude Spectra – Fault Frequency Analysis",
                 fontsize=13, fontweight="bold", y=1.01)

    fault_freqs = {"Rotor 1×":5,"Shaft":22,"Shaft 2×":44,"Shaft 3×":66,
                   "BPFO":83,"BPFI":117}

    for ax, name in zip(axes, CLASS_NAMES):
        sig  = load_one_sample(name)
        N    = len(sig)
        mag  = np.abs(rfft(sig)) / N
        freq = rfftfreq(N, d=1.0/FS)
        mask = freq <= 200
        ax.plot(freq[mask], mag[mask], color=COLORS[name], linewidth=0.9)
        ax.set_ylabel("|X(f)|", fontsize=8)
        ax.set_title(name.replace("_", " "), fontsize=10, fontweight="bold",
                     color=COLORS[name], loc="left")
        ax.grid(True, alpha=0.3)
        for label, fq in fault_freqs.items():
            ax.axvline(fq, color="grey", linestyle="--", linewidth=0.6, alpha=0.7)
            if name == CLASS_NAMES[0]:
                ax.text(fq+0.5, ax.get_ylim()[1]*0.7, label,
                        fontsize=6, color="grey", rotation=90)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fft_spectra.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Fig 3: Confusion matrix ─────────────────────────────────
def plot_confusion_matrix():
    with open(os.path.join(BASE, "models/metrics.json")) as f:
        metrics = json.load(f)

    cm     = np.array(metrics["confusion_matrix"])
    labels = [c.replace("_", "\n") for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix\n"
        f"CV Accuracy: {metrics['cv_accuracy_mean']*100:.2f}% ± {metrics['cv_accuracy_std']*100:.2f}%  |  "
        f"F1: {metrics['cv_f1_mean']*100:.2f}%",
        fontsize=11, fontweight="bold"
    )

    thresh = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Fig 4: Feature importance ───────────────────────────────
def plot_feature_importance():
    with open(os.path.join(BASE, "models/metrics.json")) as f:
        metrics = json.load(f)

    fi      = metrics["feature_importance"]
    names   = list(fi.keys())
    values  = list(fi.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names[::-1], values[::-1],
                   color=["#e74c3c" if "e_b" in n else
                          "#3498db" if "e_" in n else
                          "#2ecc71" for n in names[::-1]])
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Top Feature Importances – Random Forest\n"
                 "(Red = Bearing bands  |  Blue = Shaft bands  |  Green = Time-domain)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Generating visualizations...")
    plot_time_domain()
    plot_fft()
    plot_confusion_matrix()
    plot_feature_importance()
    print("All figures saved to /outputs/")
