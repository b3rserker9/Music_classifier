import os
import re
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models
import librosa

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# ================================
# Config (modifica qui se serve)
# ================================
DATASET_PATH = "C:/Users/aliyo/Documents/GitHub/archive"
AUDIO_DIR = os.path.join(DATASET_PATH, 'DEAM_audio', 'MEMD_audio')
DYN_BASE = os.path.join(DATASET_PATH, 'DEAM_Annotations', 'annotations',
                        'annotations per each rater', 'dynamic (per second annotations)')
VAL_DIR = os.path.join(DYN_BASE, "valence")
ARO_DIR = os.path.join(DYN_BASE, "arousal")

SR = 44100
SEGMENT_LENGTH = 5  # seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE_ES = 8
PATIENCE_RLR = 4
RANDOM_STATE = 42

CHECKPOINT_PATH = "best_model_va_best.keras"
FINAL_MODEL_PATH = "best_model_va_final.keras"
XMEAN_PATH = "x_mean.npy"
XSTD_PATH = "x_std.npy"
SCALER_PATH = "label_scaler.pkl"
META_PATH = "va_training_meta.json"

PLOT_TRAINING = True

# =====================================
# Metriche/Loss: Concordance Correlation
# =====================================

def ccc(y_true, y_pred, eps=1e-8):
    x = y_true
    y = y_pred
    x_mu = tf.reduce_mean(x, axis=0)
    y_mu = tf.reduce_mean(y, axis=0)
    x_var = tf.reduce_mean((x - x_mu) ** 2, axis=0)
    y_var = tf.reduce_mean((y - y_mu) ** 2, axis=0)
    cov = tf.reduce_mean((x - x_mu) * (y - y_mu), axis=0)
    ccc_v = (2.0 * cov) / (x_var + y_var + (x_mu - y_mu) ** 2 + eps)
    return tf.reduce_mean(ccc_v)


def ccc_loss(y_true, y_pred):
    return 1.0 - ccc(y_true, y_pred)


# =====================================
# Utils caricamento e feature
# =====================================

def load_dynamic_file(path, label_type="valence"):
    """Carica annotazioni dinamiche DEAM per un brano.
    Restituisce DataFrame con colonne [time, f"{label_type}_mean"]."""
    df = pd.read_csv(path)
    sample_cols = [c for c in df.columns if c.startswith("sample_")]
    df = df[sample_cols].T
    times = [int(re.search(r"(\d+)ms", idx).group(1)) / 1000 for idx in df.index]
    mean_vals = df.mean(axis=1).values
    return pd.DataFrame({"time": times, f"{label_type}_mean": mean_vals})


def segment_audio(y_full, sr, segment_length):
    seg_samples = segment_length * sr
    segments = [
        y_full[i:i + seg_samples]
        for i in range(0, len(y_full), seg_samples)
        if len(y_full[i:i + seg_samples]) == seg_samples
    ]
    return segments


def stack_mel_with_deltas(mel_db):
    """Crea canali [mel, delta, delta2]"""
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)
    return np.stack([mel_db, delta, delta2], axis=-1)  # (mels, frames, 3)


def weighted_mean(values):
    """Media pesata con finestra di Hann per dare più peso al centro segmento."""
    if len(values) == 0:
        return np.nan
    w = np.hanning(len(values))
    if np.all(w == 0):
        return float(np.mean(values))
    return float(np.average(values, weights=w))


# =====================================
# Preprocessing dataset
# =====================================

def prepare_dataset():
    X_by_song, y_by_song = {}, {}

    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".mp3")])
    logging.info(f"Trovati {len(audio_files)} file audio.")

    for idx, file in enumerate(audio_files):
        try:
            song_id = int(os.path.splitext(file)[0])
        except Exception:
            logging.warning(f"Nome file non numerico: {file}, skip.")
            continue

        audio_path = os.path.join(AUDIO_DIR, file)
        val_path = os.path.join(VAL_DIR, f"{song_id}.csv")
        aro_path = os.path.join(ARO_DIR, f"{song_id}.csv")

        if not os.path.exists(val_path) or not os.path.exists(aro_path):
            logging.warning(f"Annotazioni mancanti per {song_id}, skip.")
            continue

        try:
            y_full, _ = librosa.load(audio_path, sr=SR, mono=True)
            df_val = load_dynamic_file(val_path, "valence")
            df_aro = load_dynamic_file(aro_path, "arousal")
            df_dyn = pd.merge(df_val, df_aro, on="time")

            segments = segment_audio(y_full, SR, SEGMENT_LENGTH)

            feats, labels = [], []
            for seg_idx, segment in enumerate(segments):
                seg_start = seg_idx * SEGMENT_LENGTH
                seg_end = seg_start + SEGMENT_LENGTH
                seg_slice = df_dyn[(df_dyn["time"] >= seg_start) & (df_dyn["time"] < seg_end)]
                if seg_slice.empty:
                    continue

                val = weighted_mean(seg_slice["valence_mean"].values)
                aro = weighted_mean(seg_slice["arousal_mean"].values)
                if np.isnan(val) or np.isnan(aro):
                    continue
                labels.append([val, aro])

                mel = librosa.feature.melspectrogram(y=segment, sr=SR, n_mels=N_MELS,
                                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_stack = stack_mel_with_deltas(mel_db)  # (mels, frames, 3)
                feats.append(mel_stack)

            if feats:
                X_by_song[song_id] = feats
                y_by_song[song_id] = labels

        except Exception as e:
            logging.error(f"Errore su {song_id}: {e}")

    logging.info("✅ Caricamento dinamico completo.")
    return X_by_song, y_by_song


def flatten_by_ids(ids, X_dict, Y_dict):
    X, Y = [], []
    for sid in ids:
        X.extend(X_dict[sid])
        Y.extend(Y_dict[sid])
    return np.array(X), np.array(Y)


# =====================================
# Modello: CRNN (Conv → BiGRU)
# =====================================

def build_crnn_model(n_mels, n_frames, n_ch=3):
    inputs = layers.Input(shape=(n_mels, n_frames, n_ch))

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # (mels/2, frames/2)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # (mels/4, frames/4)

    # mantieni il tempo come primo asse per la RNN
    # current shape: (batch, mels/4, frames/4, 64)
    x = layers.Permute((2, 1, 3))(x)  # (batch, frames/4, mels/4, 64)
    # concat freq e canali in features
    x = layers.Reshape((-1, (n_mels // 4) * 64))(x)  # (batch, frames_red, feat_dim)

    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='sigmoid')(x)  # valence, arousal in [0,1]

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=lambda yt, yp: 0.5 * tf.keras.losses.MAE(yt, yp) + 0.5 * ccc_loss(yt, yp),
        metrics=['mae', ccc]
    )
    return model


# =====================================
# Training + valutazione
# =====================================

def np_ccc(x, y, eps=1e-8):
    x_mu, y_mu = x.mean(axis=0), y.mean(axis=0)
    vx, vy = x.var(axis=0), y.var(axis=0)
    cov = ((x - x_mu) * (y - y_mu)).mean(axis=0)
    c = (2 * cov) / (vx + vy + (x_mu - y_mu) ** 2 + eps)
    return c.mean()


def main():
    start = time.time()

    # 1) Prepara dataset
    X_by_song, y_by_song = prepare_dataset()

    song_ids = list(X_by_song.keys())
    if len(song_ids) < 3:
        raise RuntimeError("Troppi pochi brani con feature: servono almeno 3 per train/val/test.")

    train_ids, temp_ids = train_test_split(song_ids, test_size=0.3, random_state=RANDOM_STATE)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=RANDOM_STATE)

    X_train, y_train = flatten_by_ids(train_ids, X_by_song, y_by_song)
    X_val, y_val = flatten_by_ids(val_ids, X_by_song, y_by_song)
    X_test, y_test = flatten_by_ids(test_ids, X_by_song, y_by_song)

    # shape: (segments, n_mels, n_frames, n_ch)
    _, n_mels, n_frames, n_ch = X_train.shape
    logging.info(f"Input shape: mels={n_mels}, frames={n_frames}, ch={n_ch}")

    # 2) Normalizzazione input (per-canale)
    eps = 1e-6
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + eps
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # 3) Normalizzazione labels [0,1]
    scaler = MinMaxScaler((0, 1))
    y_train_n = scaler.fit_transform(y_train.reshape(-1, 2)).reshape(y_train.shape)
    y_val_n = scaler.transform(y_val.reshape(-1, 2)).reshape(y_val.shape)
    y_test_n = scaler.transform(y_test.reshape(-1, 2)).reshape(y_test.shape)

    # 4) Modello
    model = build_crnn_model(n_mels, n_frames, n_ch)
    model.summary(print_fn=lambda x: logging.info(x))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_ES, min_delta=1e-4,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=PATIENCE_RLR, factor=0.5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train_n,
        validation_data=(X_val, y_val_n),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1
    )

    # 5) Salvataggi per inference
    np.save(XMEAN_PATH, X_mean)
    np.save(XSTD_PATH, X_std)
    joblib.dump(scaler, SCALER_PATH)
    meta = {
        "sr": SR,
        "segment_length": SEGMENT_LENGTH,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "input_shape": [int(n_mels), int(n_frames), int(n_ch)],
        "model": "CRNN(Conv2D→BiGRU)",
        "loss": "0.5*MAE + 0.5*(1-CCC)",
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    # 6) Carica best checkpoint e valuta
    best_model = tf.keras.models.load_model(
        CHECKPOINT_PATH,
        custom_objects={"ccc": ccc, "ccc_loss": ccc_loss}
    )
    best_model.save(FINAL_MODEL_PATH)

    y_pred = best_model.predict(X_test)

    # MAE su scala normalizzata
    mae_val = float(np.mean(np.abs(y_pred[:, 0] - y_test_n[:, 0])))
    mae_aro = float(np.mean(np.abs(y_pred[:, 1] - y_test_n[:, 1])))
    logging.info(f"Test MAE (norm) - Valence: {mae_val:.4f}, Arousal: {mae_aro:.4f}")

    # CCC normalizzato
    ccc_norm = float(np_ccc(y_test_n, y_pred))
    logging.info(f"Test CCC (norm): {ccc_norm:.4f}")

    # su scala originale
    pred_orig = scaler.inverse_transform(y_pred)
    true_orig = scaler.inverse_transform(y_test_n)

    rmse_val = float(np.sqrt(np.mean((pred_orig[:, 0] - true_orig[:, 0]) ** 2)))
    rmse_aro = float(np.sqrt(np.mean((pred_orig[:, 1] - true_orig[:, 1]) ** 2)))
    logging.info(f"Test RMSE (orig) - Valence: {rmse_val:.4f}, Arousal: {rmse_aro:.4f}")

    ccc_orig = float(np_ccc(true_orig, pred_orig))
    logging.info(f"Test CCC (orig): {ccc_orig:.4f}")

    # 7) Plot training
    if PLOT_TRAINING:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Val MAE')
        plt.title('Training vs Validation MAE')
        plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()
        plt.tight_layout()
        plt.show()

    elapsed = time.time() - start
    logging.info(f"✅ Finito in {elapsed/60:.1f} min. Migliore salvato in: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    # Evita errore OMP su alcune installazioni
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    main()
