"""
feature_extraction.py
Extracts 26 features per sample:
  - Time-domain : RMS, peak, crest factor, kurtosis, skewness, variance, mean-abs
  - FFT-domain  : dominant freq, spectral centroid, top-5 magnitude peaks,
                  energy in rotor / shaft / bearing bands
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
import os

FS       = 1600
ROTOR_HZ = 5.0
SHAFT_HZ = 22.0
BPFO     = 83.0
BPFI     = 117.0

BAND_W   = 5.0      # ± Hz around each frequency of interest


def band_energy(magnitudes, freqs, center, width=BAND_W):
    mask = (freqs >= center - width) & (freqs <= center + width)
    return float(np.sum(magnitudes[mask] ** 2))


def extract_features(signal: np.ndarray) -> dict:
    # ── time domain ──────────────────────────────────────────
    rms         = float(np.sqrt(np.mean(signal ** 2)))
    peak        = float(np.max(np.abs(signal)))
    crest       = peak / (rms + 1e-9)
    kurt        = float(kurtosis(signal))
    skewness    = float(skew(signal))
    variance    = float(np.var(signal))
    mean_abs    = float(np.mean(np.abs(signal)))

    # ── frequency domain ─────────────────────────────────────
    N       = len(signal)
    mag     = np.abs(rfft(signal)) / N
    freqs   = rfftfreq(N, d=1.0 / FS)

    dom_idx         = int(np.argmax(mag))
    dom_freq        = float(freqs[dom_idx])
    dom_mag         = float(mag[dom_idx])
    spec_centroid   = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-9))

    # top-5 peak magnitudes (sorted desc)
    top5_idx = np.argsort(mag)[::-1][:5]
    top5_mags = sorted(mag[top5_idx].tolist(), reverse=True)

    # band energies
    e_rotor   = band_energy(mag, freqs, ROTOR_HZ)
    e_shaft   = band_energy(mag, freqs, SHAFT_HZ)
    e_2shaft  = band_energy(mag, freqs, 2 * SHAFT_HZ)
    e_3shaft  = band_energy(mag, freqs, 3 * SHAFT_HZ)
    e_bpfo    = band_energy(mag, freqs, BPFO)
    e_bpfi    = band_energy(mag, freqs, BPFI)
    total_e   = float(np.sum(mag ** 2)) + 1e-9

    return {
        "rms":          rms,
        "peak":         peak,
        "crest_factor": crest,
        "kurtosis":     kurt,
        "skewness":     skewness,
        "variance":     variance,
        "mean_abs":     mean_abs,
        "dom_freq":     dom_freq,
        "dom_mag":      dom_mag,
        "spec_centroid":spec_centroid,
        "top_mag_1":    top5_mags[0],
        "top_mag_2":    top5_mags[1],
        "top_mag_3":    top5_mags[2],
        "top_mag_4":    top5_mags[3],
        "top_mag_5":    top5_mags[4],
        "e_rotor_norm": e_rotor  / total_e,
        "e_shaft_norm": e_shaft  / total_e,
        "e_2shaft_norm":e_2shaft / total_e,
        "e_3shaft_norm":e_3shaft / total_e,
        "e_bpfo_norm":  e_bpfo   / total_e,
        "e_bpfi_norm":  e_bpfi   / total_e,
    }


def extract_all(data_csv: str, out_csv: str):
    print("Loading raw data...")
    df = pd.read_csv(data_csv)
    signal_cols = [c for c in df.columns if c.startswith("t")]

    print(f"Extracting features from {len(df)} samples...")
    features = []
    for _, row in df.iterrows():
        sig  = row[signal_cols].values.astype(float)
        feat = extract_features(sig)
        feat["label"]      = int(row["label"])
        feat["class_name"] = row["class_name"]
        features.append(feat)

    feat_df = pd.DataFrame(features)
    feat_df.to_csv(out_csv, index=False)
    print(f"Saved {len(feat_df)} feature rows → {out_csv}")
    return feat_df


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    extract_all(
        os.path.join(base, "../data/vibration_data.csv"),
        os.path.join(base, "../data/features.csv")
    )
