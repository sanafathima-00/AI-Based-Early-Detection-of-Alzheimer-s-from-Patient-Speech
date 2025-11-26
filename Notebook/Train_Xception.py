# Notebook/Train_Xception.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATASET_BASE = r"D:\Programming\Alzheimers Detection\dataset_spectrograms"
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
TEST_DIR  = os.path.join(DATASET_BASE, "test")
MODEL_PATH = r"D:\Programming\Alzheimers Detection\model.hdf5"
LOG_CSV = r"D:\Programming\Alzheimers Detection\training_log.csv"

# --------------------------------------------------
# Data generators
# --------------------------------------------------
batch_size = 4      # Reduced for stability + less RAM
img_size = (250, 250)
epochs = 40         # Increased training

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("\nDataset Loaded Successfully!")
print("Train count:", train_gen.samples)
print("Val count:", val_gen.samples)

# --------------------------------------------------
# Build model (Stage-1 only)
# --------------------------------------------------
base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)

# Freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
preds = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.inputs, outputs=preds)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Callbacks
# --------------------------------------------------
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduceLR = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger(LOG_CSV, append=True)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
print("\n=== TRAINING (Stage-1 only, 40 epochs) ===\n")

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[checkpoint, reduceLR, earlystop, csv_logger],
    verbose=1
)

model.save(MODEL_PATH)
print("\nTraining complete. Model saved:", MODEL_PATH)

# --------------------------------------------------
# Plot curves
# --------------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss")
plt.show()
