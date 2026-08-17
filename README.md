# -hvdc-sensitivity-aware-protection-
# HVDC Sensitivity-Aware Protection — Reproducible Code

This directory contains the exact code used to generate every number, table,
and figure in Section 5 and the Appendix of the accompanying manuscript.

## Files
- `hvdc_signal_generator.py` — Cigre WG B4-57 parameterized synthetic fault
  signal generator (corrected arm-capacitance derivation, adaptive DWT level).
- `hvdc_models.py` — SVM/RF/CNN/LSTM/CNN-LSTM training pipeline (corrected
  CNN architecture, fixed-length resampling, per-channel standardization).
- `hvdc_sensitivity.py` — Four single-parameter sensitivity sweeps + combined
  R_f x SNR stress test.
- `run_corrected.py` — Single entry point that reproduces every baseline,
  sweep, and stress-test number reported in the paper.
- `lstm_seeds.py` — Multi-seed (3-seed) LSTM training-stability check
  reported in Section 5.3.1.
- `tune_cnn.py` — Controlled A/B ablation that identified and fixed the
  CNN BatchNormalization/learning-rate instability (Appendix A.1).
- `corrected_results.json` — Raw output of `run_corrected.py`, the single
  source used to generate every figure in this paper.

## Reproducing the paper's results
```bash
pip install numpy scipy scikit-learn tensorflow-cpu pywavelets matplotlib --break-system-packages
python3 run_corrected.py       # baseline + 4 sweeps + combined stress test
python3 lstm_seeds.py          # 3-seed LSTM stability check (Section 5.3.1)
```

Dataset-generation seed is fixed (`numpy.random.seed(42)`); deep-learning
training seeds are 42 (single-run figures) and 1/2/3 (multi-seed LSTM check).
Expect ~20-25 minutes total runtime on CPU (no GPU required).

## Change log relative to a prior, withdrawn version of this study
See Appendix A.1 of the manuscript for the full, itemized list of
correctness defects identified and fixed (arm-capacitance derivation,
DWT decomposition level, fixed-length deep-learning input, input
normalization, CNN architecture, and retirement of a divergent
confusion-matrix script that had pre-set target accuracies in its own
docstring).

## License
MIT (suggested — adjust to your institution's policy before publishing).
