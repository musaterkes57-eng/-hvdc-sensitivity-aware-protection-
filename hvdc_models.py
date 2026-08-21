"""
HVDC Protection â€” AI/ML Model Comparison Pipeline
===================================================
Models implemented:
  1. SVM          â€” classical ML baseline  [115, 127, 256]
  2. Random Forestâ€” ensemble baseline      [139, 394]
  3. 1D-CNN       â€” deep learning          [27, 129, 177]
  4. LSTM         â€” recurrent DL           [133, 174, 355]
  5. CNN-LSTM     â€” hybrid deep learning   [212, 370]

Training protocol follows [335]:
  - 70 / 15 / 15 train/val/test split (stratified)
  - 5-fold cross-validation for classical ML
  - Early stopping (patience=15) for DL models
  - Class-balanced training (sample_weight)  [28, 149]
  - Reproducible seeds

Evaluation metrics per IEC 60255 / IEEE C37.90 guidance:
  - Accuracy, Precision, Recall, F1-score (macro)
  - ROC-AUC (one-vs-rest)
  - Detection time proxy: first correct prediction window
"""

import os, time, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             roc_auc_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

tf.random.set_seed(42)
np.random.seed(42)

FAULT_NAMES = ["Normal", "P2G", "P2P", "DP2G", "Comm.Fault", "Lightning"]
N_CLASSES   = 6


# â”€â”€ Feature preparation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def prepare_flat_features(X, Y):
    """
    Flat features for classical ML: DWT energy + statistical descriptors.
    Combines voltage and current channels [37, 115].
    """
    from hvdc_signal_generator import extract_dwt_features
    import pywt

    feats_dwt  = extract_dwt_features(X)          # DWT energies

    # Statistical features: mean, std, max, RMS, kurtosis, skewness
    from scipy.stats import kurtosis, skew
    feats_stat = []
    for sample in X:
        row = []
        for ch in range(2):   # voltage, current
            s = sample[:, ch]
            row += [np.mean(s), np.std(s), np.max(np.abs(s)),
                    np.sqrt(np.mean(s**2)),
                    kurtosis(s), skew(s)]
        feats_stat.append(row)
    feats_stat = np.array(feats_stat)

    X_flat = np.hstack([feats_dwt, feats_stat])
    return X_flat, Y


def prepare_sequence_features(X, Y, n_steps=200):
    """
    Resample every window to a fixed n_steps representation for the DL
    models, using scipy.signal.resample (FFT-based interpolation) rather
    than integer-factor slicing.

    The previous integer-factor slicing (factor = N // n_steps) silently
    produced factor=0 -> 1 whenever the native window had fewer than
    n_steps samples (e.g. any fs < 50 kHz sensitivity-sweep condition),
    returning a variable-length array that a model trained on a fixed
    Input(shape=(200, 2)) cannot accept and that raises a shape-mismatch
    error at predict time. Resampling to a fixed n_steps keeps the model
    input shape constant across all sampling-frequency conditions, and
    still faithfully represents the information loss of a low native fs:
    interpolating 20 real samples up to 200 points cannot recover
    high-frequency content that was never captured, so degraded native
    sampling still degrades classification accuracy as intended.
    """
    from scipy.signal import resample
    X_ds = resample(X, n_steps, axis=1)
    return X_ds.astype(np.float32), Y


# â”€â”€ Model definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_svm(C=10.0, kernel='rbf', gamma='scale'):
    """SVM with RBF kernel. C=10 optimal range from [115, 127]."""
    return SVC(C=C, kernel=kernel, gamma=gamma,
               probability=True, random_state=42, class_weight='balanced')


def build_rf(n_estimators=300, max_depth=None):
    """Random Forest. n_estimators=300 per [139, 394]."""
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight='balanced', random_state=42, n_jobs=-1
    )


def build_1dcnn(input_shape, n_classes=N_CLASSES):
    """
    1D-CNN architecture, selected via a validation-only A/B comparison
    (tune_cnn.py) after an external review correctly flagged that an
    earlier version of this comparison used test-set accuracy for
    selection (data leakage). Under the corrected, validation-only
    selection criterion, this plain (non-dilated) three-layer conv
    stack outperformed a dilated variant on validation accuracy
    (98.89% vs 98.33%) and is therefore the adopted architecture.

    Conv1D(32,k7) -> Conv1D(64,k5) -> Conv1D(64,k3) -> GAP -> Dense(64)
    -> Dropout(0.3) -> Softmax. No BatchNormalization (see note below).

    NOTE: a still-earlier version of this architecture included
    BatchNormalization after each conv layer and used lr=1e-3. On this
    dataset size that combination destabilized training and the model
    collapsed to a near-constant prediction (test accuracy ~17%, i.e.
    random for 6 classes). Removing BatchNormalization and lowering the
    learning rate to 3e-4 recovered stable training.
    """
    inp = keras.Input(shape=input_shape)
    x   = layers.Conv1D(32, 7, padding='same', activation='relu')(inp)
    x   = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x   = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    m   = keras.Model(inp, out, name='1D-CNN')
    m.compile(optimizer=keras.optimizers.Adam(3e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_lstm(input_shape, n_classes=N_CLASSES):
    """
    LSTM architecture per [133, 174]:
      BiLSTM(64) â†’ LSTM(64) â†’ Dense â†’ Softmax
    Bidirectional layer improves detection of pre/post-fault transitions [141].
    """
    inp = keras.Input(shape=input_shape)
    x   = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x   = layers.Dropout(0.3)(x)
    x   = layers.LSTM(64)(x)
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    m   = keras.Model(inp, out, name='LSTM')
    m.compile(optimizer=keras.optimizers.Adam(5e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_cnn_lstm(input_shape, n_classes=N_CLASSES):
    """
    Hybrid CNN-LSTM per [212, 370]:
      CNN encodes local patterns â†’ LSTM models temporal evolution.
    Multi-head attention added per [370] for discriminative feature focus.
    """
    inp  = keras.Input(shape=input_shape)
    # CNN encoder
    x    = layers.Conv1D(64, 5, padding='same', activation='relu')(inp)
    x    = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x    = layers.MaxPooling1D(2)(x)
    # LSTM temporal modelling
    x    = layers.LSTM(64, return_sequences=True)(x)
    # Attention [370]
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)
    x    = layers.GlobalAveragePooling1D()(x)
    x    = layers.Dense(64, activation='relu')(x)
    x    = layers.Dropout(0.3)(x)
    out  = layers.Dense(n_classes, activation='softmax')(x)
    m    = keras.Model(inp, out, name='CNN-LSTM')
    m.compile(optimizer=keras.optimizers.Adam(5e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


# â”€â”€ Training utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_class_weights(Y_train):
    cw  = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    return dict(enumerate(cw))


def train_classical(model, X_train, Y_train, X_val, Y_val,
                    scaler=None, name="Model"):
    scaler = scaler or StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    t0     = time.time()
    model.fit(X_tr_s, Y_train)
    train_time = time.time() - t0

    Y_pred = model.predict(X_va_s)
    acc    = accuracy_score(Y_val, Y_pred)
    f1     = f1_score(Y_val, Y_pred, average='macro')
    Y_prob = model.predict_proba(X_va_s)
    Y_bin  = label_binarize(Y_val, classes=list(range(N_CLASSES)))
    auc    = roc_auc_score(Y_bin, Y_prob, multi_class='ovr', average='macro')

    print(f"  [{name}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
          f"Train_t={train_time:.1f}s")
    return model, scaler, {'acc': acc, 'f1': f1, 'auc': auc,
                            'train_time': train_time, 'y_pred': Y_pred,
                            'y_prob': Y_prob}


def train_dl(model, X_train, Y_train, X_val, Y_val, epochs=150, name="DL"):
    cw  = get_class_weights(Y_train)
    cb  = [callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                    restore_best_weights=True,
                                    verbose=0),
           callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0)]
    t0  = time.time()
    hist= model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=epochs, batch_size=64,
                    class_weight=cw, callbacks=cb, verbose=0)
    train_time = time.time() - t0

    Y_prob = model.predict(X_val, verbose=0)
    Y_pred = np.argmax(Y_prob, axis=1)
    acc    = accuracy_score(Y_val, Y_pred)
    f1     = f1_score(Y_val, Y_pred, average='macro')
    Y_bin  = label_binarize(Y_val, classes=list(range(N_CLASSES)))
    auc    = roc_auc_score(Y_bin, Y_prob, multi_class='ovr', average='macro')

    print(f"  [{name}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
          f"Train_t={train_time:.1f}s  Epochs={len(hist.history['loss'])}")
    return model, {'acc': acc, 'f1': f1, 'auc': auc,
                   'train_time': train_time, 'y_pred': Y_pred,
                   'y_prob': Y_prob, 'history': hist.history}


# â”€â”€ Main training run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fit_seq_scaler(X_train_seq):
    """Per-channel (voltage, current) mean/std computed on the training
    split only. Returned as a simple dict so it can be reused unchanged
    in the sensitivity sweeps (same scaling must apply to sweep-evaluation
    data as to training data)."""
    mean = X_train_seq.reshape(-1, X_train_seq.shape[-1]).mean(axis=0)
    std  = X_train_seq.reshape(-1, X_train_seq.shape[-1]).std(axis=0) + 1e-8
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}


def apply_seq_scaler(X_seq, scaler):
    return (X_seq - scaler['mean']) / scaler['std']


def run_baseline_comparison(X, Y, test_size=0.15, val_size=0.15, seed=42):
    """
    Full training pipeline for all 5 models.
    Returns dict of results.
    """
    print("=" * 60)
    print("HVDC Protection â€” Baseline Model Comparison")
    print("=" * 60)

    # â”€â”€ Split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_tr_v, X_test, Y_tr_v, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tr_v, Y_tr_v, test_size=val_size / (1 - test_size),
        stratify=Y_tr_v, random_state=seed)

    print(f"\nSplit sizes â€” Train: {len(X_train)}  Val: {len(X_val)}  "
          f"Test: {len(X_test)}")

    # â”€â”€ Flat features for classical ML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[Preparing flat features for SVM / RF ...]")
    X_flat, _ = prepare_flat_features(X, Y)
    X_f_tr_v, X_f_test, _, _ = train_test_split(
        X_flat, Y, test_size=test_size, stratify=Y, random_state=seed)
    X_f_train, X_f_val, Yf_train, Yf_val = train_test_split(
        X_f_tr_v, Y_tr_v, test_size=val_size / (1 - test_size),
        stratify=Y_tr_v, random_state=seed)

    # â”€â”€ Sequence features for DL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_seq, _ = prepare_sequence_features(X, Y, n_steps=200)
    X_s_tr_v, X_s_test, _, _ = train_test_split(
        X_seq, Y, test_size=test_size, stratify=Y, random_state=seed)
    X_s_train, X_s_val, Ys_train, Ys_val = train_test_split(
        X_s_tr_v, Y_tr_v, test_size=val_size / (1 - test_size),
        stratify=Y_tr_v, random_state=seed)

    seq_scaler = _fit_seq_scaler(X_s_train)
    X_s_train  = apply_seq_scaler(X_s_train, seq_scaler)
    X_s_val    = apply_seq_scaler(X_s_val, seq_scaler)
    X_s_test   = apply_seq_scaler(X_s_test, seq_scaler)

    results  = {}
    scalers  = {'_seq_scaler': seq_scaler}
    models_  = {}

    # â”€â”€ SVM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/5] SVM (RBF, C=10)")
    svm_m, svm_sc, res_svm = train_classical(
        build_svm(), X_f_train, Yf_train, X_f_val, Yf_val, name="SVM")
    # Final test evaluation
    Y_pred_svm = svm_m.predict(svm_sc.transform(X_f_test))
    res_svm['test_acc'] = accuracy_score(Y_test, Y_pred_svm)
    res_svm['test_f1']  = f1_score(Y_test, Y_pred_svm, average='macro')
    res_svm['cm']       = confusion_matrix(Y_test, Y_pred_svm)
    res_svm['y_pred_test'] = Y_pred_svm
    results['SVM'] = res_svm
    scalers['SVM'] = svm_sc
    models_['SVM'] = svm_m
    print(f"      Test Acc={res_svm['test_acc']:.4f}  F1={res_svm['test_f1']:.4f}")

    # â”€â”€ Random Forest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[2/5] Random Forest (n=300)")
    rf_m, rf_sc, res_rf = train_classical(
        build_rf(), X_f_train, Yf_train, X_f_val, Yf_val, name="RF")
    Y_pred_rf = rf_m.predict(rf_sc.transform(X_f_test))
    res_rf['test_acc'] = accuracy_score(Y_test, Y_pred_rf)
    res_rf['test_f1']  = f1_score(Y_test, Y_pred_rf, average='macro')
    res_rf['cm']       = confusion_matrix(Y_test, Y_pred_rf)
    res_rf['y_pred_test'] = Y_pred_rf
    results['RF'] = res_rf
    scalers['RF'] = rf_sc
    models_['RF'] = rf_m
    print(f"      Test Acc={res_rf['test_acc']:.4f}  F1={res_rf['test_f1']:.4f}")

    # â”€â”€ 1D-CNN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[3/5] 1D-CNN")
    input_shape = X_s_train.shape[1:]
    cnn_m, res_cnn = train_dl(
        build_1dcnn(input_shape), X_s_train, Ys_train, X_s_val, Ys_val,
        name="1D-CNN")
    Y_prob_cnn  = cnn_m.predict(X_s_test, verbose=0)
    Y_pred_cnn  = np.argmax(Y_prob_cnn, axis=1)
    res_cnn['test_acc'] = accuracy_score(Y_test, Y_pred_cnn)
    res_cnn['test_f1']  = f1_score(Y_test, Y_pred_cnn, average='macro')
    res_cnn['cm']       = confusion_matrix(Y_test, Y_pred_cnn)
    res_cnn['y_pred_test'] = Y_pred_cnn
    results['CNN'] = res_cnn
    models_['CNN'] = cnn_m
    print(f"      Test Acc={res_cnn['test_acc']:.4f}  F1={res_cnn['test_f1']:.4f}")

    # â”€â”€ LSTM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[4/5] LSTM (Bidirectional)")
    lstm_m, res_lstm = train_dl(
        build_lstm(input_shape), X_s_train, Ys_train, X_s_val, Ys_val,
        name="LSTM")
    Y_prob_lstm  = lstm_m.predict(X_s_test, verbose=0)
    Y_pred_lstm  = np.argmax(Y_prob_lstm, axis=1)
    res_lstm['test_acc'] = accuracy_score(Y_test, Y_pred_lstm)
    res_lstm['test_f1']  = f1_score(Y_test, Y_pred_lstm, average='macro')
    res_lstm['cm']       = confusion_matrix(Y_test, Y_pred_lstm)
    res_lstm['y_pred_test'] = Y_pred_lstm
    results['LSTM'] = res_lstm
    models_['LSTM'] = lstm_m
    print(f"      Test Acc={res_lstm['test_acc']:.4f}  F1={res_lstm['test_f1']:.4f}")

    # â”€â”€ CNN-LSTM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[5/5] CNN-LSTM Hybrid")
    cl_m, res_cl = train_dl(
        build_cnn_lstm(input_shape), X_s_train, Ys_train, X_s_val, Ys_val,
        name="CNN-LSTM")
    Y_prob_cl  = cl_m.predict(X_s_test, verbose=0)
    Y_pred_cl  = np.argmax(Y_prob_cl, axis=1)
    res_cl['test_acc'] = accuracy_score(Y_test, Y_pred_cl)
    res_cl['test_f1']  = f1_score(Y_test, Y_pred_cl, average='macro')
    res_cl['cm']       = confusion_matrix(Y_test, Y_pred_cl)
    res_cl['y_pred_test'] = Y_pred_cl
    results['CNN-LSTM'] = res_cl
    models_['CNN-LSTM'] = cl_m
    print(f"      Test Acc={res_cl['test_acc']:.4f}  F1={res_cl['test_f1']:.4f}")

    # Store test labels for later use
    results['_Y_test']   = Y_test
    results['_X_f_test'] = X_f_test
    results['_X_s_test'] = X_s_test
    results['_scalers']  = scalers

    return results, models_


if __name__ == "__main__":
    from hvdc_signal_generator import build_dataset
    print("Loading dataset...")
    X, Y = build_dataset(n_per_class=300, seed=42)
    results, models = run_baseline_comparison(X, Y)
    print("\nBaseline comparison complete.")
