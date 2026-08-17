"""
Corrected, unified HVDC pipeline run.
Single source of truth for ALL numbers/figures — no separate
hvdc_confusion_matrix_v2.py path is used, eliminating the two-codebase
divergence that produced inconsistent Fig.2 / Fig.6 accuracies.
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/hvdc_fixed')

import numpy as np
t0 = time.time()

from hvdc_signal_generator import build_dataset, F_S_DEFAULT, N_SM_HALF_ARM
from hvdc_models import run_baseline_comparison
from hvdc_sensitivity import (sweep_fault_resistance, sweep_snr,
                               sweep_sampling_freq, sweep_fault_location,
                               combined_stress_test)

print(f"[physics check] N_SM_HALF_ARM={N_SM_HALF_ARM}  C_eff=6*C_SM/N (Cigre WG B4-57)")

print("Building dataset (200/class x 6 = 1200, matches Appendix A.4)...")
X, Y = build_dataset(n_per_class=200, seed=42, snr_db=32.0, r_fault_range=(0, 500))
print("Dataset:", X.shape, dict(zip(*np.unique(Y, return_counts=True))))

print("\nTraining all 5 models (single run, used for every figure)...")
baseline_results, models = run_baseline_comparison(X, Y)
scalers = baseline_results['_scalers']

out = {}
out['baseline'] = {m: {'test_acc': float(baseline_results[m]['test_acc']),
                        'test_f1':  float(baseline_results[m]['test_f1']),
                        'cm': baseline_results[m]['cm'].tolist()}
                    for m in ['SVM','RF','CNN','LSTM','CNN-LSTM']}

print("\n=== Sweep: Fault Resistance ===")
R_range, res_R = sweep_fault_resistance(models, scalers)
out['sweep_R'] = {'x': list(R_range), 'y': {m: res_R[m] for m in res_R}}

print("\n=== Sweep: SNR ===")
snr_range, res_snr = sweep_snr(models, scalers)
out['sweep_snr'] = {'x': list(snr_range), 'y': {m: res_snr[m] for m in res_snr}}

print("\n=== Sweep: Sampling Frequency ===")
fs_range, res_fs = sweep_sampling_freq(models, scalers)
out['sweep_fs'] = {'x': list(fs_range), 'y': {m: res_fs[m] for m in res_fs}}

print("\n=== Sweep: Fault Location ===")
loc_range, res_loc = sweep_fault_location(models, scalers)
out['sweep_loc'] = {'x': list(loc_range), 'y': {m: res_loc[m] for m in res_loc}}

print("\n=== Combined R x SNR stress test ===")
R_vals, snr_vals, stress = combined_stress_test(models, scalers)
out['stress'] = {'R': R_vals, 'snr': snr_vals,
                  'acc': {m: stress[m].tolist() for m in stress}}

with open('/home/claude/hvdc_fixed/corrected_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved corrected_results.json")

print("\n--- Baseline test accuracy (single source for Fig.2 AND Fig.6) ---")
for m in ['SVM','RF','CNN','LSTM','CNN-LSTM']:
    print(f"  {m:10s} acc={out['baseline'][m]['test_acc']*100:.2f}%  f1={out['baseline'][m]['test_f1']*100:.2f}%")

print("\n--- Combined stress worst-case (R=500, SNR=15) ---")
for m in ['SVM','RF','CNN','LSTM','CNN-LSTM']:
    print(f"  {m:10s} {out['stress']['acc'][m][2][2]*100:.2f}%")
