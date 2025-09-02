# evaluate_cnn.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "data_preparation/dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# === Data generator (hanya rescale, tanpa augmentasi) ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Load best model ===
model = tf.keras.models.load_model("cnn_best_model.h5")

# === Evaluate ===
loss, acc = model.evaluate(val_generator, verbose=0)
print(f"✅ Accuracy: {acc*100:.2f}%")

# === Predictions ===
y_pred = model.predict(val_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# === Classification Report (Precision, Recall, F1) ===
report = classification_report(y_true, y_pred_classes, target_names=class_labels, digits=4)
print("📊 Classification Report:")
print(report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("📈 Confusion Matrix disimpan di confusion_matrix.png")
