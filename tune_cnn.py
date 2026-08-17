import sys, warnings, json, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/hvdc_fixed')

from hvdc_signal_generator import build_dataset
from hvdc_models import (prepare_sequence_features, _fit_seq_scaler, apply_seq_scaler,
                          get_class_weights, N_CLASSES)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
tf.random.set_seed(42); np.random.seed(42)

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

def evalmodel(m, tag):
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
          callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0)]
    hist = m.fit(X_s_train, Y_train, validation_data=(X_s_val, Y_val),
                 epochs=150, batch_size=64, class_weight=cw, callbacks=cb, verbose=0)
    yp = np.argmax(m.predict(X_s_test, verbose=0), axis=1)
    acc = accuracy_score(Y_test_, yp); f1 = f1_score(Y_test_, yp, average='macro')
    print(f"{tag}: test_acc={acc:.4f} f1={f1:.4f} epochs={len(hist.history['loss'])}")
    print("  per-class recall:", (confusion_matrix(Y_test_, yp, normalize='true').diagonal()).round(2))
    return acc

# Config A: lower LR, no BN, GAP -> wider dense
inp = keras.Input(shape=input_shape)
x = layers.Conv1D(32, 7, padding='same', activation='relu')(inp)
x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(N_CLASSES, activation='softmax')(x)
mA = keras.Model(inp, out)
mA.compile(optimizer=keras.optimizers.Adam(3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
evalmodel(mA, "A: no-BN, LR=3e-4, plain conv stack")

# Config B: keep dilation but lower LR + more patience + no BN
inp = keras.Input(shape=input_shape)
x = layers.Conv1D(64, 7, padding='same', activation='relu')(inp)
x = layers.Conv1D(64, 5, padding='same', dilation_rate=2, activation='relu')(x)
x = layers.Conv1D(128, 3, padding='same', dilation_rate=4, activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(N_CLASSES, activation='softmax')(x)
mB = keras.Model(inp, out)
mB.compile(optimizer=keras.optimizers.Adam(3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
evalmodel(mB, "B: dilated, no-BN, LR=3e-4")
