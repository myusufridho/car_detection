import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# --- 1. Persiapan dan Pemuatan Data ---
print("--- Memuat dan Mempersiapkan Dataset ---")

data_dir = 'data_preparation/hood'

if not os.path.isdir(data_dir):
    print(f"Error: Direktori '{data_dir}' tidak ditemukan.")
    print("Pastikan struktur folder Anda sudah benar.")
    exit()

BATCH_SIZE = 32
IMG_SIZE = (256, 256)

train_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=42,
    validation_split=0.2,
    subset='training'
)

val_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

class_names = train_ds.class_names
print("Nama Kelas:", class_names)
print(f"Jumlah gambar training: {len(train_ds) * BATCH_SIZE}")
print(f"Jumlah gambar validasi: {len(val_ds) * BATCH_SIZE}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. Arsitektur Model CNN ---
print("\n--- Membangun Model CNN ---")

model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    Rescaling(1./255),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 3. Melatih Model dengan Early Stopping ---
print("\n--- Memulai Pelatihan Model dengan Early Stopping ---")

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

EPOCHS = 50

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# --- 4. Menyimpan Model ---
print("\n--- Menyimpan Model ---")

model_save_path = 'hood.keras'
model.save(model_save_path)
print(f"Model berhasil disimpan di file: {model_save_path}")

# --- 5. Evaluasi dan Visualisasi Hasil ---
print("\n--- Melakukan Evaluasi Model ---")

# Mendapatkan label dan prediksi untuk set validasi
val_labels = np.concatenate([y for x, y in val_ds], axis=0)
val_images = np.concatenate([x for x, y in val_ds], axis=0)

val_pred_raw = model.predict(val_images)
val_pred = (val_pred_raw > 0.5).astype(int)

# Menampilkan Classification Report
print("\nClassification Report:")
print(classification_report(val_labels, val_pred, target_names=class_names))

# Menampilkan Confusion Matrix
conf_matrix = confusion_matrix(val_labels, val_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prediksi')
plt.ylabel('Label Sebenarnya')
plt.title('Confusion Matrix')
plt.show()

# Plot akurasi dan loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Training dan Validasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Loss Training dan Validasi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nPelatihan dan evaluasi selesai.")
