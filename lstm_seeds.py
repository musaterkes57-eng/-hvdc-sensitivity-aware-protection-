import sys, warnings, json, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hvdc_signal_generator import build_dataset
from hvdc_models import (prepare_sequence_features, _fit_seq_scaler, apply_seq_scaler,
                          get_class_weights, N_CLASSES, build_lstm)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

X, Y = build_dataset(n_per_class=200, seed=42, snr_db=32.0, r_fault_range=(0, 500))
X_tr_v, X_test, Y_tr_v, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_tr_v, Y_tr_v, test_size=0.15/0.85, stratify=Y_tr_v, random_state=42)
X_s_train, Y_train = prepare_sequence_features(X_train, Y_train)
X_s_val, Y_val     = prepare_sequence_features(X_val, Y_val)
X_s_test, Y_test_  = prepare_sequence_features(X_test, Y_test)
sc = _fit_seq_scaler(X_s_train)
X_s_train = apply_seq_scaler(X_s_train, sc)
X_s_val   = apply_seq_scaler(X_s_val, sc)
X_s_test  = apply_seq_scaler(X_s_test, sc)
input_shape = X_s_train.shape[1:]
cw = get_class_weights(Y_train)

accs, f1s, lightning_recall = [], [], []
for seed in [1, 2, 3]:
    tf.random.set_seed(seed); np.random.seed(seed)
    m = build_lstm(input_shape)
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
          callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0)]
    m.fit(X_s_train, Y_train, validation_data=(X_s_val, Y_val),
          epochs=150, batch_size=64, class_weight=cw, callbacks=cb, verbose=0)
    yp = np.argmax(m.predict(X_s_test, verbose=0), axis=1)
    acc = accuracy_score(Y_test_, yp); f1 = f1_score(Y_test_, yp, average='macro')
    cm = confusion_matrix(Y_test_, yp, normalize='true')
    accs.append(acc); f1s.append(f1); lightning_recall.append(cm[5,5])
    print(f"seed={seed}: acc={acc:.4f} f1={f1:.4f} lightning_recall={cm[5,5]:.2f}")

print(f"\nMean acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"Mean f1 ={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print(f"Lightning recall values: {lightning_recall}")

OUTPUT_FILE = ROOT / "lstm_seed_check.json"
json.dump({'accs': accs, 'f1s': f1s, 'lightning_recall': lightning_recall},
          open(OUTPUT_FILE, 'w'))
print(f"Saved {OUTPUT_FILE}")
