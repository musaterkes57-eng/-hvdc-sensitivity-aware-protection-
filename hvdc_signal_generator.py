"""
HVDC Fault Signal Generator
============================
Based on:
- Cigré B4 DC Grid Test System (2014)
- Muniappan (2021) [18] — three-stage fault evolution model
- Xiang et al. (2021) [63] — MMC-HVDC fault characteristic equations
- Liu et al. (2023) [231] — VSC-HVDC traveling waveform parameters
- Ikhide et al. (2018) [244] — noise characterization in HVDC protection

Fault Evolution Model (three stages):
  Stage 1: Capacitor discharge    i_c(t) = V0/Z0 * exp(-alpha*t) * sin(omega*t)
  Stage 2: Diode freewheeling     i_d(t) = I_peak * exp(-t/tau_d)
  Stage 3: AC grid current feed   i_g(t) = I_ac * (1 - exp(-t/tau_g))

System Parameters (Cigré B4 bipolar ±320 kV, 800 MW):
  V_dc = 320e3 V  (pole voltage)
  C_sm = 6e-3 F   (submodule capacitance per arm, MMC)
  L_arm = 50e-3 H (arm inductance)
  L_line = 40e-3 H/100km (line inductance)
  R_line = 0.01  Ohm/km
  f_s = 50 kHz   (default sampling frequency)
"""

import numpy as np
from scipy.signal import butter, filtfilt
import pywt

# ── System constants (Cigré B4 / [18, 63, 231]) ─────────────────────────────
V_DC       = 320e3        # Pole-to-ground DC voltage [V]
C_SM       = 10e-3        # Per-submodule capacitance [F] — Cigre WG B4-57 spec
N_SM_HALF_ARM = 400        # Submodules per half-arm — Cigre WG B4-57 spec
L_ARM      = 50e-3        # Arm inductance [H]
L_SMOOTH   = 80e-3        # Smoothing reactor [H]  [18]
R_LINE_KM  = 0.01         # Line resistance per km [Ω/km]
L_LINE_KM  = 40e-3 / 100  # Line inductance per km [H/km]
Z0         = 300.0        # Surge impedance [Ω]   [231]
V_WAVE     = 2.0e8        # Wave propagation speed [m/s]  [231]
LINE_LEN   = 300          # Default line length [km]
F_S_DEFAULT= 50_000       # Default sampling frequency [Hz]
T_WINDOW   = 0.02         # Observation window [s]  20 ms

# Fault types
FAULT_TYPES = {
    0: "Normal",
    1: "Pole-to-Ground (P2G)",
    2: "Pole-to-Pole (P2P)",
    3: "Double-Pole-to-Ground (DP2G)",
    4: "Commutation Fault",
    5: "Lightning Transient",
}

# ── Core signal generation ────────────────────────────────────────────────────

def _damped_oscillation(t, V0, R_f, L, C):
    """Stage 1: Capacitor discharge — damped oscillatory current [18, 63]"""
    alpha  = R_f / (2 * L)
    omega0 = 1.0 / np.sqrt(L * C)
    if omega0 > alpha:
        omega_d = np.sqrt(omega0**2 - alpha**2)
        return (V0 / (L * omega_d)) * np.exp(-alpha * t) * np.sin(omega_d * t)
    else:
        # Over-damped case
        beta = np.sqrt(alpha**2 - omega0**2)
        return (V0 / (2 * L * beta)) * (np.exp(-(alpha - beta) * t) -
                                         np.exp(-(alpha + beta) * t))


def _freewheeling(t, I_peak, tau_d=2e-3):
    """Stage 2: Diode freewheeling current [18]"""
    return I_peak * np.exp(-t / tau_d)


def _ac_infeed(t, I_ac, tau_g=5e-3):
    """Stage 3: AC grid current feeding [18]"""
    return I_ac * (1.0 - np.exp(-t / tau_g))


def generate_fault_signal(fault_type, R_fault=0.0, fault_location_frac=0.5,
                           fs=F_S_DEFAULT, snr_db=40.0,
                           line_len_km=LINE_LEN, seed=None):
    """
    Generate a synthetic HVDC fault current signal.

    Parameters
    ----------
    fault_type        : int  — 0=Normal, 1=P2G, 2=P2P, 3=DP2G, 4=CF, 5=Lightning
    R_fault           : float — fault resistance [Ω]
    fault_location_frac: float — fault location as fraction of line length [0,1]
    fs                : float — sampling frequency [Hz]
    snr_db            : float — signal-to-noise ratio [dB]
    line_len_km       : float — line length [km]
    seed              : int   — random seed

    Returns
    -------
    t   : np.ndarray  — time vector [s]
    i_v : np.ndarray  — voltage signal (normalized, p.u.)
    i_c : np.ndarray  — current signal (normalized, p.u.)
    label: int        — fault type label
    """
    rng = np.random.default_rng(seed)
    N   = int(T_WINDOW * fs)
    t   = np.linspace(0, T_WINDOW, N)

    # Fault inception at t = 5 ms (allows pre-fault baseline)
    t_fault = 5e-3
    dt      = t - t_fault
    mask    = dt >= 0

    # Line parameters at fault location
    R_line = R_LINE_KM * fault_location_frac * line_len_km
    L_line = L_LINE_KM * fault_location_frac * line_len_km
    L_eff  = L_ARM + L_line + L_SMOOTH
    # Equivalent arm capacitance from the official Cigré B4 DC grid test
    # system formula (Cigré WG B4-57, "The Cigré B4 DC grid test system"):
    # C_eff = 6*C_SM/N, N = submodules per half-arm. With the Cigré B4
    # spec values (C_SM=10 mF, N=400) this gives C_eff=150 uF, matching
    # the value published in that document. This replaces a prior
    # unreferenced C_eff = C_SM/6 that had no derivation and produced an
    # oscillation frequency (~14 Hz) far below any real HVDC fault
    # transient.
    C_eff  = 6.0 * C_SM / N_SM_HALF_ARM
    R_eff  = R_line + R_fault

    # Traveling wave attenuation with distance [231]
    tw_atten = np.exp(-0.002 * fault_location_frac * line_len_km)

    # ── Voltage signal ────────────────────────────────────────────────────────
    v = np.ones(N)   # pre-fault: normalized 1.0 p.u.

    if fault_type == 0:  # Normal — small load fluctuation
        v += 0.005 * np.sin(2 * np.pi * 50 * t) + 0.003 * rng.standard_normal(N)

    elif fault_type == 1:  # Pole-to-Ground
        # Voltage drops but not to zero (grounding impedance) [18]
        v_drop = tw_atten * 0.85 / (1 + R_fault / 200)
        v[mask] = 1.0 - v_drop * (1 - np.exp(-dt[mask] / 2e-3))

    elif fault_type == 2:  # Pole-to-Pole — voltage collapses to zero [18]
        v_drop = tw_atten / (1 + R_fault / 500)
        v[mask] = np.maximum(0, 1.0 - v_drop * dt[mask] / 3e-3)

    elif fault_type == 3:  # Double-Pole-to-Ground — asymmetric [231]
        ratio = 1.0 / (1 + R_fault / 300)
        v[mask] = (1.0 - ratio * (1 - np.exp(-dt[mask] / 1.5e-3)) *
                   (1 + 0.1 * np.sin(2 * np.pi * 150 * dt[mask])))

    elif fault_type == 4:  # Commutation fault — AC-induced [390]
        # LCC commutation failure: voltage dip + recovery attempt
        v[mask] = (1.0 - 0.6 * np.exp(-dt[mask] / 3e-3) *
                   np.abs(np.sin(2 * np.pi * 50 * dt[mask])))

    elif fault_type == 5:  # Lightning transient — fast spike [32]
        # High-frequency transient, decays quickly
        spike_dur = min(int(0.5e-3 * fs), np.sum(mask))
        v[mask]   = 1.0
        idx_start = np.argmax(mask)
        v[idx_start:idx_start + spike_dur] += (
            1.5 * tw_atten * np.exp(-np.linspace(0, 10, spike_dur)) *
            np.sin(2 * np.pi * 5000 * np.linspace(0, 0.5e-3, spike_dur))
        )

    # ── Current signal ────────────────────────────────────────────────────────
    i = np.zeros(N)
    i[~mask] = 1.0   # nominal load current (p.u.)

    if fault_type == 0:
        i = np.ones(N) + 0.02 * np.sin(2 * np.pi * 100 * t)

    elif fault_type in [1, 2, 3]:
        i_stage1 = _damped_oscillation(dt[mask], V_DC * tw_atten,
                                        R_eff, L_eff, C_eff)
        I_peak   = np.max(np.abs(i_stage1)) if len(i_stage1) > 0 else 1.0
        t2_mask  = (dt >= 3e-3) & mask
        t3_mask  = (dt >= 6e-3) & mask
        i[mask]  += i_stage1 / (V_DC / Z0)          # normalize
        if fault_type == 1:
            I_ac = 1.2 / (1 + R_fault / 100)
            i[t3_mask] += _ac_infeed(dt[t3_mask] - 6e-3, I_ac) * tw_atten
        if fault_type == 3:
            i[mask] *= (1 + 0.2 * rng.standard_normal(np.sum(mask)) * 0.1)

    elif fault_type == 4:
        i[mask] = (1.0 + 0.5 * np.exp(-dt[mask] / 4e-3) *
                   np.sin(2 * np.pi * 50 * dt[mask]) * (-1))

    elif fault_type == 5:
        i_spike  = np.zeros(N)
        idx_start = np.argmax(mask)
        spike_dur = min(int(0.5e-3 * fs), N - idx_start)
        i_spike[idx_start:idx_start + spike_dur] = (
            2.0 * tw_atten * np.exp(-np.linspace(0, 8, spike_dur)) *
            np.sin(2 * np.pi * 3000 * np.linspace(0, 0.5e-3, spike_dur))
        )
        i = np.ones(N) + i_spike

    # Clip to physically reasonable range
    i = np.clip(i, -10, 15)
    v = np.clip(v, -0.5, 2.0)

    # ── Add AWGN noise [244] ──────────────────────────────────────────────────
    def add_noise(signal, snr_db):
        sig_power  = np.mean(signal**2)
        noise_power= sig_power / (10 ** (snr_db / 10))
        noise      = rng.standard_normal(len(signal)) * np.sqrt(noise_power)
        return signal + noise

    v = add_noise(v, snr_db)
    i = add_noise(i, snr_db)

    return t, v.astype(np.float32), i.astype(np.float32), fault_type


# ── Dataset generation ────────────────────────────────────────────────────────

def build_dataset(n_per_class=300, fs=F_S_DEFAULT, snr_db=40.0,
                  r_fault_range=(0, 300), seed=42):
    """
    Build balanced dataset of all fault types.

    Sensitivity parameters are uniformly sampled within given ranges.
    """
    rng    = np.random.default_rng(seed)
    X_v, X_i, Y = [], [], []

    for ft in range(6):
        for k in range(n_per_class):
            R_f  = float(rng.uniform(*r_fault_range))
            loc  = float(rng.uniform(0.05, 0.95))
            snr  = float(rng.uniform(snr_db - 5, snr_db + 5))
            _, v, i, label = generate_fault_signal(
                fault_type=ft, R_fault=R_f, fault_location_frac=loc,
                fs=fs, snr_db=snr, seed=int(rng.integers(0, 1e6))
            )
            X_v.append(v)
            X_i.append(i)
            Y.append(label)

    X_v = np.array(X_v)
    X_i = np.array(X_i)
    X   = np.stack([X_v, X_i], axis=-1)   # shape (N, T, 2)
    Y   = np.array(Y)
    return X, Y


# ── DWT Feature Extraction [37, 219] ─────────────────────────────────────────

def dwt_features(signal, wavelet='db4', level=5):
    """
    Extract DWT energy features from a 1-D signal.
    Daubechies db4 chosen per [37, 219, 358].
    Returns a fixed-length (level+1) normalized energy-ratio vector.

    The decomposition level is capped at pywt's maximum level that avoids
    boundary-effect-dominated coefficients (pywt.dwt_max_level). At low
    sampling frequencies the 20 ms observation window contains too few
    samples for a full level-5 decomposition (e.g. N=20 at fs=1 kHz), and
    forcing level=5 there produces coefficients dominated by edge
    artefacts rather than signal content (PyWavelets itself raises this as
    a UserWarning). Levels beyond what the signal length supports are
    zero-padded so every sample still yields the same feature dimension.
    """
    max_lvl   = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    use_level = max(1, min(level, max_lvl))
    coeffs    = pywt.wavedec(signal, wavelet, level=use_level)
    energies  = np.array([np.sum(c**2) for c in coeffs])
    total     = np.sum(energies) + 1e-12
    energies  = energies / total
    # pad to level+1 bands so the flat feature vector length never changes
    if len(energies) < level + 1:
        energies = np.pad(energies, (0, level + 1 - len(energies)))
    return energies


def extract_dwt_features(X, wavelet='db4', level=5):
    """Apply DWT feature extraction to dataset (voltage + current channels)."""
    feats = []
    for sample in X:
        v_feat = dwt_features(sample[:, 0], wavelet, level)
        i_feat = dwt_features(sample[:, 1], wavelet, level)
        feats.append(np.concatenate([v_feat, i_feat]))
    return np.array(feats)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Generating example signals for all fault types...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#2E86AB', '#E84855', '#3BB273', '#F7B731',
              '#9B59B6', '#E67E22']

    for ft in range(6):
        t, v, i, label = generate_fault_signal(
            fault_type=ft, R_fault=50.0, fault_location_frac=0.3,
            fs=F_S_DEFAULT, snr_db=35.0, seed=ft * 7
        )
        ax = axes[ft]
        ax2 = ax.twinx()
        ax.plot(t * 1e3, v, color=colors[ft], lw=1.5, label='Voltage (p.u.)')
        ax2.plot(t * 1e3, i, color=colors[ft], lw=1.5, ls='--',
                 alpha=0.7, label='Current (p.u.)')
        ax.axvline(5.0, color='gray', ls=':', lw=1, alpha=0.7)
        ax.set_title(FAULT_TYPES[ft], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_ylabel('Voltage (p.u.)', fontsize=9, color=colors[ft])
        ax2.set_ylabel('Current (p.u.)', fontsize=9, color=colors[ft],
                       alpha=0.7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('HVDC Fault Signal Types — Cigré B4 Parameters\n'
                 '(fs = 50 kHz, R_fault = 50 Ω, SNR = 35 dB)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/fig_fault_signals.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print("Saved fig_fault_signals.png")

    # Build and inspect dataset
    print("\nBuilding dataset (300 samples/class × 6 classes)...")
    X, Y = build_dataset(n_per_class=300, seed=42)
    print(f"  Dataset shape : {X.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")
