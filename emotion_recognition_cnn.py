from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import numpy as np
import os
import re
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# =========================================================
# Dataset path
# =========================================================
dataset_path = "C:/Users/aliyo/Documents/GitHub/archive"
audio_dir = os.path.join(dataset_path, 'DEAM_audio', 'MEMD_audio')
dyn_base = os.path.join(dataset_path, 'DEAM_Annotations', 'annotations',
                        'annotations per each rater', 'dynamic (per second annotations)')
val_dir = os.path.join(dyn_base, "valence")
aro_dir = os.path.join(dyn_base, "arousal")

# =========================================================
# Funzione caricamento annotazioni dinamiche
# =========================================================
def load_dynamic_file(path, label_type="valence"):
    df = pd.read_csv(path)
    sample_cols = [c for c in df.columns if c.startswith("sample_")]
    df = df[sample_cols].T
    times = [int(re.search(r"(\d+)ms", idx).group(1)) / 1000 for idx in df.index]
    mean_vals = df.mean(axis=1).values
    return pd.DataFrame({"time": times, f"{label_type}_mean": mean_vals})

# =========================================================
# Preprocessing audio + labels
# =========================================================
sr = 44100
segment_length = 5
segment_samples = segment_length * sr
X_by_song, y_by_song = {}, {}
audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".mp3")])

for idx, file in enumerate(audio_files):
    song_id = int(os.path.splitext(file)[0])
    audio_path = os.path.join(audio_dir, file)
    val_path = os.path.join(val_dir, f"{song_id}.csv")
    aro_path = os.path.join(aro_dir, f"{song_id}.csv")

    if not os.path.exists(val_path) or not os.path.exists(aro_path):
        logging.warning(f"Missing annotations for {song_id}, skipping.")
        continue

    try:
        y_full, _ = librosa.load(audio_path, sr=sr, mono=True)
        df_val = load_dynamic_file(val_path, "valence")
        df_aro = load_dynamic_file(aro_path, "arousal")
        df_dyn = pd.merge(df_val, df_aro, on="time")

        segments = [
            y_full[i:i + segment_samples]
            for i in range(0, len(y_full), segment_samples)
            if len(y_full[i:i + segment_samples]) == segment_samples
        ]

        mel_specs, labels = [], []
        for seg_idx, segment in enumerate(segments):
            seg_start = seg_idx * segment_length
            seg_end = seg_start + segment_length
            seg_slice = df_dyn[(df_dyn["time"] >= seg_start) & (df_dyn["time"] < seg_end)]
            if seg_slice.empty:
                continue
            val, aro = seg_slice["valence_mean"].mean(), seg_slice["arousal_mean"].mean()
            labels.append([val, aro])

            mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128,
                                                 n_fft=2048, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_specs.append(mel_db)

        if mel_specs:
            X_by_song[song_id] = mel_specs
            y_by_song[song_id] = labels

    except Exception as e:
        logging.error(f"Error processing {song_id}: {e}")

logging.info("✅ Dynamic data loading complete.")

# =========================================================
# Train/Val/Test split
# =========================================================
song_ids = list(X_by_song.keys())
train_ids, temp_ids = train_test_split(song_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

def flatten_by_ids(ids, X_dict, y_dict):
    X, y = [], []
    for sid in ids:
        X.extend(X_dict[sid]); y.extend(y_dict[sid])
    return np.array(X), np.array(y)

X_train, y_train = flatten_by_ids(train_ids, X_by_song, y_by_song)
X_val, y_val = flatten_by_ids(val_ids, X_by_song, y_by_song)
X_test, y_test = flatten_by_ids(test_ids, X_by_song, y_by_song)

# =========================================================
# Normalizzazione input
# =========================================================
eps = 1e-6
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + eps
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
X_train, X_val, X_test = np.expand_dims(X_train, -1), np.expand_dims(X_val, -1), np.expand_dims(X_test, -1)

# =========================================================
# Normalizzazione labels
# =========================================================
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0, 1))
y_train_n = scaler.fit_transform(y_train.reshape(-1, 2)).reshape(y_train.shape)
y_val_n = scaler.transform(y_val.reshape(-1, 2)).reshape(y_val.shape)
y_test_n = scaler.transform(y_test.reshape(-1, 2)).reshape(y_test.shape)

# =========================================================
# Modello CNN
# =========================================================
_, n_mels, n_frames, _ = X_train.shape
model = models.Sequential([
    layers.Input(shape=(n_mels, n_frames, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# =========================================================
# Training con early stopping "stabile"
# =========================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, min_delta=1e-4,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_dynamic_early_stop.keras", monitor="val_loss",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=4, factor=0.5, verbose=1
    )
]

history = model.fit(
    X_train, y_train_n,
    validation_data=(X_val, y_val_n),
    epochs=50, batch_size=32,
    callbacks=callbacks
)

# =========================================================
# Valutazione
# =========================================================
best_model = tf.keras.models.load_model("best_model_dynamic.keras")
y_pred = best_model.predict(X_test)

mae_valence = np.mean(np.abs(y_pred[:, 0] - y_test_n[:, 0]))
mae_arousal = np.mean(np.abs(y_pred[:, 1] - y_test_n[:, 1]))
print(f"Test MAE - Valence: {mae_valence:.4f}, Arousal: {mae_arousal:.4f}")

pred_orig = scaler.inverse_transform(y_pred)
true_orig = scaler.inverse_transform(y_test_n)
rmse_valence = np.sqrt(np.mean((pred_orig[:, 0] - true_orig[:, 0]) ** 2))
rmse_arousal = np.sqrt(np.mean((pred_orig[:, 1] - true_orig[:, 1]) ** 2))
print(f"Test RMSE (original scale) - Valence: {rmse_valence:.4f}, Arousal: {rmse_arousal:.4f}")

# =========================================================
# Curve di training
# =========================================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss (MAE)')
plt.plot(history.history['val_loss'], label='Val Loss (MAE)')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Training vs Validation MAE')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()
plt.tight_layout()
plt.show()
