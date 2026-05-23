"""
generate_data.py
Generates synthetic but physically realistic vibration signals for a wind turbine
drivetrain, mimicking what an ADXL345 accelerometer would record.

4 classes:
  0 – Normal          (clean rotation + low noise)
  1 – Blade Imbalance (strong 1× rotor frequency + harmonics)
  2 – Bearing Wear    (bearing defect frequencies + random impulses)
  3 – Misalignment    (strong 2× + 3× shaft harmonics)
"""

import numpy as np
import pandas as pd
import os

# ── signal parameters ──────────────────────────────────────────────────────
FS          = 1600          # sampling rate (Hz) — ADXL345 max rate
DURATION    = 1.0           # seconds per sample
N           = int(FS * DURATION)
ROTOR_HZ    = 5.0           # rotor frequency (~5 Hz = 300 RPM, typical low-wind)
SHAFT_HZ    = 22.0          # high-speed shaft (gear ratio ~4.4×)
# Bearing defect frequencies (BPFO/BPFI approximations)
BPFO        = 83.0          # outer race defect freq
BPFI        = 117.0         # inner race defect freq

SAMPLES_PER_CLASS = 300
RANDOM_SEED       = 42
np.random.seed(RANDOM_SEED)

t = np.linspace(0, DURATION, N, endpoint=False)


def add_noise(signal, snr_db=20):
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return signal + np.random.normal(0, np.sqrt(noise_power), len(signal))


def gen_normal(_):
    s  = 0.30 * np.sin(2 * np.pi * ROTOR_HZ * t)
    s += 0.15 * np.sin(2 * np.pi * SHAFT_HZ * t)
    s += 0.08 * np.sin(2 * np.pi * 2 * SHAFT_HZ * t)
    return add_noise(s, snr_db=22)


def gen_imbalance(_):
    # exaggerated 1× rotor and harmonics
    s  = 0.80 * np.sin(2 * np.pi * ROTOR_HZ * t + np.random.uniform(0, 2*np.pi))
    s += 0.40 * np.sin(2 * np.pi * 2 * ROTOR_HZ * t)
    s += 0.20 * np.sin(2 * np.pi * 3 * ROTOR_HZ * t)
    s += 0.15 * np.sin(2 * np.pi * SHAFT_HZ * t)
    return add_noise(s, snr_db=18)


def gen_bearing(_):
    s  = 0.25 * np.sin(2 * np.pi * ROTOR_HZ * t)
    s += 0.20 * np.sin(2 * np.pi * SHAFT_HZ * t)
    # defect tone
    s += 0.35 * np.sin(2 * np.pi * BPFO * t + np.random.uniform(0, 2*np.pi))
    s += 0.20 * np.sin(2 * np.pi * BPFI * t)
    # random impulses (spalling)
    impulse_times = np.random.choice(N, size=np.random.randint(8, 20), replace=False)
    s[impulse_times] += np.random.uniform(1.2, 2.5, len(impulse_times))
    return add_noise(s, snr_db=15)


def gen_misalignment(_):
    s  = 0.25 * np.sin(2 * np.pi * ROTOR_HZ * t)
    # strong 2× and 3× shaft — classic misalignment signature
    s += 0.25 * np.sin(2 * np.pi * SHAFT_HZ * t)
    s += 0.55 * np.sin(2 * np.pi * 2 * SHAFT_HZ * t + np.random.uniform(0, 2*np.pi))
    s += 0.40 * np.sin(2 * np.pi * 3 * SHAFT_HZ * t)
    s += 0.15 * np.sin(2 * np.pi * 4 * SHAFT_HZ * t)
    return add_noise(s, snr_db=17)


GENERATORS = {
    0: gen_normal,
    1: gen_imbalance,
    2: gen_bearing,
    3: gen_misalignment,
}
CLASS_NAMES = {0: "Normal", 1: "Blade_Imbalance", 2: "Bearing_Wear", 3: "Misalignment"}


def build_dataset():
    rows = []
    for label, gen_fn in GENERATORS.items():
        for i in range(SAMPLES_PER_CLASS):
            signal = gen_fn(i)
            row = {"label": label, "class_name": CLASS_NAMES[label]}
            for j, v in enumerate(signal):
                row[f"t{j}"] = round(float(v), 6)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating vibration dataset...")
    df = build_dataset()
    out = os.path.join(os.path.dirname(__file__), "../data/vibration_data.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} samples → {out}")
    print(df["class_name"].value_counts())
