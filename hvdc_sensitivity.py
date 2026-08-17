"""
HVDC Protection — Sensitivity Parameter Analysis
==================================================
Systematic evaluation of model performance under:
  1. Fault Resistance variation     : 0 → 500 Ω  [231, 237, 358]
  2. SNR variation                  : 10 → 40 dB  [244, 245]
  3. Sampling Frequency variation   : 1 → 50 kHz  [219, 222, 358]
  4. Fault Location variation       : 5% → 95%    [231, 240]

Each sensitivity sweep uses a held-out evaluation set generated
specifically for that parameter condition (models are NOT retrained).
This protocol reflects real-world deployment constraints where a
protection model trained under nominal conditions is evaluated under
degraded conditions [335].

The approach mirrors the single-parameter variation methodology
of [244, 245, 231] while extending it to AI/ML model comparison,
as called for by the gap identified in Section 5.2 of the review.
"""

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from hvdc_signal_generator import (
    build_dataset, generate_fault_signal,
    extract_dwt_features, F_S_DEFAULT, FAULT_TYPES
)

N_SWEEP   = 150   # evaluation samples per sweep point (×6 classes = 900)
SEED_BASE = 1000


def _make_sweep_data(fault_type_list, R_fault, snr_db, fs, loc_frac,
                     n_per_class=N_SWEEP):
    """
    Generate evaluation data for a specific parameter combination.
    All six fault types included to maintain class balance.
    """
    X_v, X_i, Y = [], [], []
    rng = np.random.default_rng(SEED_BASE)
    for ft in fault_type_list:
        for k in range(n_per_class):
            seed_k = int(rng.integers(0, 1e7))
            _, v, i, label = generate_fault_signal(
                fault_type=ft, R_fault=R_fault,
                fault_location_frac=loc_frac,
                fs=fs, snr_db=snr_db, seed=seed_k
            )
            X_v.append(v); X_i.append(i); Y.append(label)
    X = np.stack([np.array(X_v), np.array(X_i)], axis=-1)
    return X, np.array(Y)


def _flat_feats(X):
    from scipy.stats import kurtosis, skew
    feats_dwt  = extract_dwt_features(X)
    feats_stat = []
    for sample in X:
        row = []
        for ch in range(2):
            s = sample[:, ch]
            row += [np.mean(s), np.std(s), np.max(np.abs(s)),
                    np.sqrt(np.mean(s**2)), kurtosis(s), skew(s)]
        feats_stat.append(row)
    return np.hstack([feats_dwt, np.array(feats_stat)])


def _seq_feats(X, n_steps=200):
    from scipy.signal import resample
    return resample(X, n_steps, axis=1).astype(np.float32)


def _evaluate_all(models, scalers, X_eval, Y_eval):
    """Evaluate all models on given evaluation set. Returns dict of accuracies."""
    results = {}
    X_flat = _flat_feats(X_eval)
    X_seq  = _seq_feats(X_eval)
    seq_scaler = scalers.get('_seq_scaler')
    if seq_scaler is not None:
        X_seq = (X_seq - seq_scaler['mean']) / seq_scaler['std']

    for name, model in models.items():
        if name in ('SVM', 'RF'):
            sc = scalers[name]
            Y_pred = model.predict(sc.transform(X_flat))
        else:
            Y_prob = model.predict(X_seq, verbose=0)
            Y_pred = np.argmax(Y_prob, axis=1)

        results[name] = {
            'acc': accuracy_score(Y_eval, Y_pred),
            'f1':  f1_score(Y_eval, Y_pred, average='macro')
        }
    return results


# ── Sweep 1: Fault Resistance ─────────────────────────────────────────────────

def sweep_fault_resistance(models, scalers,
                            R_range=None):
    """
    Evaluate accuracy vs. fault resistance (0–500 Ω).
    Uses SNR=35 dB (representative) and loc=0.3 (moderately remote).
    Fault types 1,2,3 tested (DC-side faults most resistance-sensitive [231]).
    """
    if R_range is None:
        R_range = [0, 25, 50, 100, 150, 200, 300, 400, 500]

    sweep_results = {m: [] for m in models}
    ft_list = list(range(6))

    for R in R_range:
        print(f"  R_fault = {R:4.0f} Ω", end="  ")
        X_ev, Y_ev = _make_sweep_data(ft_list, R_fault=R, snr_db=35.0,
                                       fs=F_S_DEFAULT, loc_frac=0.3)
        ev = _evaluate_all(models, scalers, X_ev, Y_ev)
        for m in models:
            sweep_results[m].append(ev[m]['acc'])
            print(f"{m}={ev[m]['acc']:.3f}", end="  ")
        print()

    return R_range, sweep_results


# ── Sweep 2: SNR ──────────────────────────────────────────────────────────────

def sweep_snr(models, scalers, snr_range=None):
    """
    Evaluate accuracy vs. SNR (10–40 dB) at nominal conditions.
    SNR range aligned with [244, 245]: 25 dB is critical threshold.
    """
    if snr_range is None:
        snr_range = [10, 15, 20, 25, 30, 35, 40]

    sweep_results = {m: [] for m in models}
    ft_list = list(range(6))

    for snr in snr_range:
        print(f"  SNR = {snr:2d} dB", end="  ")
        X_ev, Y_ev = _make_sweep_data(ft_list, R_fault=50.0, snr_db=float(snr),
                                       fs=F_S_DEFAULT, loc_frac=0.3)
        ev = _evaluate_all(models, scalers, X_ev, Y_ev)
        for m in models:
            sweep_results[m].append(ev[m]['acc'])
            print(f"{m}={ev[m]['acc']:.3f}", end="  ")
        print()

    return snr_range, sweep_results


# ── Sweep 3: Sampling Frequency ───────────────────────────────────────────────

def sweep_sampling_freq(models, scalers, fs_range=None):
    """
    Evaluate accuracy vs. sampling frequency (1–50 kHz).
    Models retrained at 50 kHz; evaluated at lower rates by resampling.
    This simulates deployment on hardware with constrained ADC rate [219, 222].
    The n_steps for DL models adjusts proportionally.
    """
    if fs_range is None:
        fs_range = [1000, 2000, 5000, 10000, 20000, 50000]

    sweep_results = {m: [] for m in models}
    ft_list = list(range(6))

    for fs in fs_range:
        print(f"  fs = {fs:6d} Hz", end="  ")
        X_ev, Y_ev = _make_sweep_data(ft_list, R_fault=50.0, snr_db=35.0,
                                       fs=fs, loc_frac=0.3)
        # DL models need fixed sequence length; resample to 200 steps
        ev = _evaluate_all(models, scalers, X_ev, Y_ev)
        for m in models:
            sweep_results[m].append(ev[m]['acc'])
            print(f"{m}={ev[m]['acc']:.3f}", end="  ")
        print()

    return fs_range, sweep_results


# ── Sweep 4: Fault Location ───────────────────────────────────────────────────

def sweep_fault_location(models, scalers, loc_range=None):
    """
    Evaluate accuracy vs. fault location (5%–95% of line length).
    Captures near-end (close-in) and far-end (remote) degradation [240].
    """
    if loc_range is None:
        loc_range = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    sweep_results = {m: [] for m in models}
    ft_list = list(range(6))

    for loc in loc_range:
        print(f"  Location = {loc*100:4.0f}%", end="  ")
        X_ev, Y_ev = _make_sweep_data(ft_list, R_fault=50.0, snr_db=35.0,
                                       fs=F_S_DEFAULT, loc_frac=loc)
        ev = _evaluate_all(models, scalers, X_ev, Y_ev)
        for m in models:
            sweep_results[m].append(ev[m]['acc'])
            print(f"{m}={ev[m]['acc']:.3f}", end="  ")
        print()

    return loc_range, sweep_results


# ── Combined stress test ──────────────────────────────────────────────────────

def combined_stress_test(models, scalers):
    """
    Evaluate under simultaneous parameter stress:
      R_fault = {0, 200, 500} Ω  ×  SNR = {40, 25, 15} dB
    This is the multi-parameter interaction test called for in Section 5.2.
    Produces a 3×3 performance heatmap per model.
    """
    R_vals   = [0, 200, 500]
    snr_vals = [40, 25, 15]
    ft_list  = list(range(6))

    # results[model][R_idx][snr_idx] = accuracy
    results = {m: np.zeros((3, 3)) for m in models}

    for ri, R in enumerate(R_vals):
        for si, snr in enumerate(snr_vals):
            print(f"  R={R:3d}Ω, SNR={snr:2d}dB", end="  ")
            X_ev, Y_ev = _make_sweep_data(ft_list, R_fault=float(R),
                                           snr_db=float(snr),
                                           fs=F_S_DEFAULT, loc_frac=0.3,
                                           n_per_class=100)
            ev = _evaluate_all(models, scalers, X_ev, Y_ev)
            for m in models:
                results[m][ri, si] = ev[m]['acc']
                print(f"{m}={ev[m]['acc']:.3f}", end="  ")
            print()

    return R_vals, snr_vals, results


if __name__ == "__main__":
    from hvdc_signal_generator import build_dataset
    from hvdc_models import run_baseline_comparison

    print("Building main dataset...")
    X, Y = build_dataset(n_per_class=300, seed=42)

    print("\nRunning baseline comparison...")
    baseline_results, models = run_baseline_comparison(X, Y)

    scalers = baseline_results['_scalers']

    print("\n" + "=" * 60)
    print("SENSITIVITY SWEEP 1: Fault Resistance")
    print("=" * 60)
    R_range, res_R = sweep_fault_resistance(models, scalers)

    print("\n" + "=" * 60)
    print("SENSITIVITY SWEEP 2: SNR")
    print("=" * 60)
    snr_range, res_snr = sweep_snr(models, scalers)

    print("\n" + "=" * 60)
    print("SENSITIVITY SWEEP 3: Sampling Frequency")
    print("=" * 60)
    fs_range, res_fs = sweep_sampling_freq(models, scalers)

    print("\n" + "=" * 60)
    print("SENSITIVITY SWEEP 4: Fault Location")
    print("=" * 60)
    loc_range, res_loc = sweep_fault_location(models, scalers)

    print("\n" + "=" * 60)
    print("COMBINED STRESS TEST")
    print("=" * 60)
    R_vals, snr_vals, stress_results = combined_stress_test(models, scalers)

    print("\nAll sweeps complete.")
