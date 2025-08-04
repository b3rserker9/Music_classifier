from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
import time
import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

dataset_path = kagglehub.dataset_download('imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music')
print(f"Path to dataset file: {dataset_path}")
print('Data source import complete.')

"""**Defining Dataset Root Path**"""
# Define the dataset's root path

# List all files and subdirectories within the dataset directory
files = os.listdir(dataset_path)
print("Files and subdirectories in the dataset:")
print(files)
"""**Accessing Individual Files**"""
# Define the paths for "DEAM_audio/MEMD_audio" and the static annotations CSV
audio_dir = os.path.join(dataset_path, 'DEAM_audio', 'MEMD_audio')
static_csv = os.path.join(dataset_path, 'DEAM_Annotations', 'annotations',
                          'annotations averaged per song', 'song_level',
                          'static_annotations_averaged_songs_1_2000.csv')

# Print to verify the paths
print("Audio Directory Path:", audio_dir)
print("Static CSV Path:", static_csv)

# Check if the paths exist
if os.path.exists(audio_dir):
    print("Audio directory exists.")
else:
    print("Audio directory does not exist!")

if os.path.exists(static_csv):
    print("Static CSV file exists.")
else:
    print("Static CSV file does not exist!")

"""**Verifying Successful Setup**"""

song_id = 10  # song ID

# Load audio
audio_path = os.path.join(audio_dir, f"{song_id}.mp3")
y, sr = librosa.load(audio_path, sr=44100, mono=True)
print(f"Audio loaded: {len(y)} samples at {sr} Hz")

"""## Converting Raw Audio to Mel-Spectrograms with librosa

Convert the raw audio frequencies to a mel-spectrogram for song_id = 10
"""

# Convert to mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
print(f"Mel-spectrogram shape: {mel_spec_db.shape}")

# Load static annotations
df = pd.read_csv(static_csv)
label = df[df['song_id'] == song_id][[' valence_mean', ' arousal_mean']].values[0]
print(f"Valence: {label[0]}, Arousal: {label[1]}")

# Segment into 5-second chunks (for consistency with CNN)
segment_length = 5  # seconds
segment_samples = segment_length * sr
segments = [y[i:i + segment_samples] for i in range(0, len(y), segment_samples)
            if len(y[i:i + segment_samples]) == segment_samples]

mel_specs = []
for segment in segments:
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_specs.append(mel_spec_db)

mel_specs = np.array(mel_specs)
print(f"Number of 5-second segments: {len(mel_specs)}, Shape of each: {mel_specs[0].shape}")

"""Instead of using raw frequencies, mel-spectrograms map them onto the mel scale. Feelings in music show up in patterns of sound, like speed and tone. For example:

1. Fast and lively sounds (high arousal/valence) often feel energetic and happy.

2. Slow and deep sounds (low arousal/valence) feel calm or sad.


Mel-spectrograms do a great job of capturing these patterns.
"""

# 1. Visualize full mel-spectrogram
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel-Spectrogram for Song ID {song_id} (Full)')
plt.tight_layout()

# 2. Visualize first segment's mel-spectrogram
plt.subplot(1, 2, 2)
librosa.display.specshow(mel_specs[0], sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel-Spectrogram for Song ID {song_id} (First 5s Segment)')
plt.tight_layout()

# 3. Plot valence-arousal on 2D plane
plt.figure(figsize=(6, 6))
plt.scatter(label[0], label[1], color='red', s=100, label=f'Song {song_id}')
plt.xlabel('Valence Mean (1-9)')
plt.ylabel('Arousal Mean (1-9)')
plt.title(f'Valence-Arousal Plane for Song ID {song_id}')
plt.xlim(1, 9)
plt.ylim(1, 9)
plt.grid(True)
plt.legend()
plt.show()

"""**Extend the conversion to full dataset**"""

# Process all songs
X_by_song = {}
y_by_song = {}

for idx, song_id in enumerate(df['song_id'].values):
    audio_path = os.path.join(audio_dir, f"{song_id}.mp3")
    start_time = time.time()

    try:
        y_full, sr = librosa.load(audio_path, sr=44100, mono=True)
        segment_samples = 5 * sr

        segments = [
            y_full[i:i + segment_samples]
            for i in range(0, len(y_full), segment_samples)
            if len(y_full[i:i + segment_samples]) == segment_samples
        ]

        label = df[df['song_id'] == song_id][[' valence_mean', ' arousal_mean']].values[0]

        mel_specs = []
        for segment in segments:
            mel_spec = librosa.feature.melspectrogram(
                y=segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec_db)

        X_by_song[song_id] = mel_specs
        y_by_song[song_id] = [label] * len(mel_specs)

        elapsed = time.time() - start_time
        logging.info(f"[{idx + 1}/{len(df)}] Processed '{song_id}' with {len(segments)} segments in {elapsed:.2f}s")

    except Exception as e:
        logging.error(f"Error processing '{song_id}': {e}")

logging.info("✅ Data loading and mel-spectrogram extraction complete.")

"""Output: X_by_song and y_by_song are dictionaries where each key is a song_id, and values are lists of mel-spectrograms and corresponding raw label arrays.

## Data Splitting

Instead of splitting the song at the segment level, I decided to split at the song level to make sure all segments of a song stay in the same set.



*   Reason: If the split is done at the segment level, 5-second segments from the same song might end up in different sets (train, val, test). This could introduce data leakage because segments from the same song are highly correlated

*   Benefit: Reduces the risk of leakage
"""

# Get list of song IDs
song_ids = list(X_by_song.keys())

# Split songs into train, validation, and test sets
train_ids, temp_ids = train_test_split(song_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Flatten into segment-level lists
X_train, y_train = [], []
for sid in train_ids:
    X_train.extend(X_by_song[sid])
    y_train.extend(y_by_song[sid])

X_val, y_val = [], []
for sid in val_ids:
    X_val.extend(X_by_song[sid])
    y_val.extend(y_by_song[sid])

X_test, y_test = [], []
for sid in test_ids:
    X_test.extend(X_by_song[sid])
    y_test.extend(y_by_song[sid])

# Convert to numpy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Expand dimensions for CNN or LSTM
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"Train segments: {X_train.shape}, Val segments: {X_val.shape}, Test segments: {X_test.shape}")
print(f"Train labels: {y_train.shape}, Val labels: {y_val.shape}, Test labels: {y_test.shape}")

"""*   Training: 70% of the data
*   Validation: 15% of the data
*   Test: 15% of the data

## Data Normalization

Normalization with MinMaxScaler (only for labels). The mel-spectrograms in X_train, X_val, X_test remain as-is with power_to_db normalization.
"""

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit scaler on training labels only (flatten to 2D for scaler)
y_train_2d = y_train.reshape(-1, 2)  # Shape: (n_train_segments, 2)
scaler.fit(y_train_2d)  # Compute min and max from training set only

# Transform all sets
y_train_normalized = scaler.transform(y_train_2d).reshape(y_train.shape)
y_val_normalized = scaler.transform(y_val.reshape(-1, 2)).reshape(y_val.shape)
y_test_normalized = scaler.transform(y_test.reshape(-1, 2)).reshape(y_test.shape)

print(f"Training label min: {scaler.data_min_}, max: {scaler.data_max_}")
print(f"Normalized training label example: {y_train_normalized[0]}")
print(f"Normalized validation label example: {y_val_normalized[0]}")
print(f"Normalized test label example: {y_test_normalized[0]}")

np.savez_compressed('deam_dataset_processed.npz',
                    X_train=X_train,
                    y_train=y_train_normalized,
                    X_val=X_val,
                    y_val=y_val_normalized,
                    X_test=X_test,
                    y_test=y_test_normalized)
print("Dataset saved to 'deam_dataset_processed.npz'")

"""# CNN Model

## Define CNN model
"""
# Define CNN model with Input layer
model = models.Sequential([
    layers.Input(shape=(128, 431, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='linear')
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

"""## Train the model"""

# Train model
history = model.fit(X_train, y_train_normalized,
                    validation_data=(X_val, y_val_normalized),
                    epochs=50, batch_size=32,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10),
                        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                    ])

"""The CNN model is trained to learn the mapping from the mel-spectrogram inputs to the two output variables (valence_mean and arousal_mean)

## Evaluate the model
"""

# Evaluate model
y_pred = model.predict(X_test)
mae_valence = np.mean(np.abs(y_pred[:, 0] - y_test_normalized[:, 0]))
mae_arousal = np.mean(np.abs(y_pred[:, 1] - y_test_normalized[:, 1]))
print(f"Test MAE - Valence: {mae_valence:.4f}, Arousal: {mae_arousal:.4f}")

"""**Additional Metrics**"""

# Additional metrics
mse_valence = np.mean((y_pred[:, 0] - y_test_normalized[:, 0]) ** 2)
mse_arousal = np.mean((y_pred[:, 1] - y_test_normalized[:, 1]) ** 2)
rmse_valence = np.sqrt(mse_valence)
rmse_arousal = np.sqrt(mse_arousal)
print(f"Test MSE - Valence: {mse_valence:.4f}, Arousal: {mse_arousal:.4f}")
print(f"Test RMSE - Valence: {rmse_valence:.4f}, Arousal: {rmse_arousal:.4f}")

"""**MSE**: The squared errors (0.0277 for valence, 0.0383 for arousal) indicate the average squared deviation. The higher MSE for arousal (0.0383) compared to valence (0.0277) suggests larger individual errors for arousal predictions.

**RMSE**: Root Mean Squared Error measures the average magnitude of the errors between predicted values and actual values, with errors squared before averaging and then taking the square root. Since the original labels are on a 1-9 scale, and the normalization range is effectively 8 units (9 - 1), I can scale the errors back to the original scale for better intuition:

* Valence RMSE (original) = 0.1664 × 8 = 1.3312
* Arousal RMSE (original) = 0.1957 × 8 = 1.5656

# Plot Training Model #
"""

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

"""**Evaluate song id=47**"""

# Evaluate song_id=47 specifically
specific_song_id = 47
if specific_song_id in test_ids:
    idx_start = sum(len(X_by_song[sid]) for sid in test_ids[:test_ids.index(specific_song_id)])
    idx_end = idx_start + len(X_by_song[specific_song_id])

    y_pred_song47 = y_pred[idx_start:idx_end]
    y_true_song47 = y_test_normalized[idx_start:idx_end]

    mae_valence = np.mean(np.abs(y_pred_song47[:, 0] - y_true_song47[:, 0]))
    mae_arousal = np.mean(np.abs(y_pred_song47[:, 1] - y_true_song47[:, 1]))
    rmse_valence = np.sqrt(np.mean((y_pred_song47[:, 0] - y_true_song47[:, 0]) ** 2))
    rmse_arousal = np.sqrt(np.mean((y_pred_song47[:, 1] - y_true_song47[:, 1]) ** 2))
    print(f"\nTest MAE for Song ID {specific_song_id} - Valence: {mae_valence:.4f}, Arousal: {mae_arousal:.4f}")
    print(f"Test RMSE for Song ID {specific_song_id} - Valence: {rmse_valence:.4f}, Arousal: {rmse_arousal:.4f}")

    pred_valence_mean = np.mean(y_pred_song47[:, 0])
    pred_arousal_mean = np.mean(y_pred_song47[:, 1])
    true_valence_mean = np.mean(y_true_song47[:, 0])
    true_arousal_mean = np.mean(y_true_song47[:, 1])
    print(f"Predicted Valence Mean: {pred_valence_mean:.3f}, True Valence Mean: {true_valence_mean:.3f}")
    print(f"Predicted Arousal Mean: {pred_arousal_mean:.3f}, True Arousal Mean: {true_arousal_mean:.3f}")

    pred_original = scaler.inverse_transform([np.mean(y_pred_song47, axis=0)])[0]
    true_original = scaler.inverse_transform([np.mean(y_true_song47, axis=0)])[0]
    print(f"Predicted (Original): Valence {pred_original[0]:.1f}, Arousal {pred_original[1]:.1f}")
    print(f"True (Original): Valence {true_original[0]:.1f}, Arousal {true_original[1]:.1f}")
else:
    print(f"Song ID {specific_song_id} not in test set.")

# Visualize predictions vs actual
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test_normalized[:, 0], y_pred[:, 0], alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Valence (Normalized)')
plt.ylabel('Predicted Valence (Normalized)')
plt.title('Valence: Actual vs Predicted')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test_normalized[:, 1], y_pred[:, 1], alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Arousal (Normalized)')
plt.ylabel('Predicted Arousal (Normalized)')
plt.title('Arousal: Actual vs Predicted')
plt.legend()

plt.tight_layout()
plt.show()

"""**Mean Absolute Error (MAE)**

    1. Valence MAE: 0.0538 × 8 ≈ 0.43 (original scale)
    2. Arousal MAE: 0.0478 × 8 ≈ 0.38 (original scale)

The average absolute error per segment is about 0.4 units on the 1-9 scale, which is quite low and indicates good segment-level accuracy for song_id=47. This is well below the overall test RMSE (0.1664 for valence, 0.1957 for arousal), suggesting song_id=47 is easier to predict than the average test song.

**Predicted VS True Means**

**Normalized Differences**

    1. Valence: 0.466 (predicted) - 0.412 (actual) = 0.054
    2. Arousal: 0.460 (predicted) - 0.413 (actual) = 0.047

**Original Scale Differences**

    1. Valence: 4.8 (predicted) - 4.4 (actual) = 0.4
    2. Arousal: 4.7 (predicted) - 4.4 (actual) = 0.3


The model overpredicts both valence and arousal by about 0.3-0.4 units on the original scale. This consistent bias suggests the model might be slightly skewed toward higher values, possibly due to the training data distribution or feature representation.

Conclusion: The low MAE/RMSE (0.0538, 0.0478) and small error in the original scale (~0.4) show the model predicts this song’s emotional content decently. This is a positive sign for the CNN’s capability on certain songs
"""
