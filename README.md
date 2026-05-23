# Wind Turbine Blade Fault Detection System

## Overview

A machine learning system that detects mechanical faults in wind turbine drivetrains by analysing vibration signals from an ADXL345 accelerometer. The system classifies each 1-second signal window into one of four operational states using FFT-based feature extraction and a Random Forest classifier.

**Achieved: 100% CV Accuracy | 100% F1-Score | 5-Fold Stratified Cross-Validation**

---

## Fault Classes Detected

| Class | Description | Key Signature |
|---|---|---|
| ✅ Normal | Healthy turbine | Clean 1× rotor + shaft harmonics |
| ⚠️ Blade Imbalance | Mass/aerodynamic imbalance | Elevated 1× rotor frequency amplitude |
| 🔴 Bearing Wear | Main/gearbox bearing spalling | BPFO/BPFI defect tones + impulses |
| ⚠️ Misalignment | Shaft/coupling misalignment | Strong 2× and 3× shaft harmonics |

---

## Project Structure

```
turbine_fault/
├── src/
│   ├── generate_data.py        # Synthetic vibration data generator
│   ├── feature_extraction.py   # FFT + time-domain feature extraction (21 features)
│   ├── train_model.py          # Random Forest training with 5-fold CV
│   ├── predict.py              # Inference engine with fault severity output
│   └── visualize.py            # Publication-quality plot generation
├── data/
│   ├── vibration_data.csv      # 1200 raw signals (1600 samples/signal)
│   └── features.csv            # 21 extracted features per sample
├── models/
│   ├── rf_model.pkl            # Trained Random Forest model
│   ├── scaler.pkl              # StandardScaler
│   └── metrics.json            # CV results + confusion matrix + importances
├── outputs/
│   ├── time_domain_signals.png
│   ├── fft_spectra.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── run_pipeline.py             # ← Run this to reproduce everything
└── requirements.txt
```

---

## Signal Processing Pipeline

```
Raw Vibration Signal (ADXL345 @ 1600 Hz)
        │
        ▼
  Time-Domain Features          FFT Features
  ─────────────────────         ──────────────────────────
  RMS, Peak, Crest Factor       Dominant Frequency & Magnitude
  Kurtosis, Skewness            Spectral Centroid
  Variance, Mean Absolute       Top-5 Spectral Peaks
                                Band Energy (Rotor, Shaft, BPFO, BPFI)
        │
        ▼
  21-Feature Vector
        │
        ▼
  StandardScaler → Random Forest (200 trees) → Fault Class + Confidence
        │
        ▼
  Maintenance Action Recommendation
```

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run full pipeline (data → train → visualize → demo)**
```bash
python run_pipeline.py
```

**3. Run inference on a specific fault type**
```bash
python src/predict.py --mode simulate --fault normal
python src/predict.py --mode simulate --fault bearing
python src/predict.py --mode simulate --fault imbalance
python src/predict.py --mode simulate --fault misalignment
```

---

## Results

| Metric | Score |
|---|---|
| CV Accuracy (5-fold) | **100.00% ± 0.00%** |
| CV F1-Score (weighted) | **100.00%** |
| Classes | 4 (Normal, Blade Imbalance, Bearing Wear, Misalignment) |
| Training Samples | 1200 (300 per class) |
| Features | 21 (FFT + time-domain) |
| Classifier | Random Forest (200 estimators) |

### Sample Output
```
====================================================
  WIND TURBINE FAULT DETECTION SYSTEM
====================================================
  Prediction  : Bearing_Wear
  Status      : 🔴 ALERT   – Bearing Wear Detected
  Confidence  : 78.0%
  Action      : URGENT: Inspect main bearing immediately.

  Class Probabilities:
    Normal                13.0%  ███
    Blade_Imbalance        8.0%  ██
    Bearing_Wear          78.0%  ███████████████████████
    Misalignment           1.0%
====================================================
```

---

## Technical Background

Mechanical faults in rotating machinery produce characteristic frequency signatures:
- **Blade imbalance** → energy concentrates at 1× rotor frequency (5 Hz at 300 RPM)
- **Bearing wear** → BPFO (~83 Hz) and BPFI (~117 Hz) tones appear with random impulses from spalling
- **Misalignment** → 2× and 3× shaft frequency components (44 Hz, 66 Hz) dominate

This mirrors the approach used in industrial Condition Monitoring Systems (CMS) on utility-scale wind turbines like those manufactured by Nordex.

### Hardware Mapping (Real Deployment)

| Component | Specification |
|---|---|
| Sensor | ADXL345 3-axis accelerometer |
| Sampling Rate | 1600 Hz |
| Placement | Gearbox housing / main bearing housing |
| Controller | Arduino Mega / ESP32 |
| Alert Output | Serial dashboard / SCADA notification |

---

## Author

**Ashwani Rawat**  
[GitHub](https://github.com/ashwanirawat113) | [LinkedIn](https://linkedin.com/in/ashwani-rawat25) | ashwanirawat625@gmail.com
