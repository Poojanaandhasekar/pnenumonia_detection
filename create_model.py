import tensorflow as tf
from tensorflow.keras import layers, models

# Config
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
).cache().prefetch(tf.data.AUTOTUNE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
).cache().prefetch(tf.data.AUTOTUNE)

# Define CNN model for binary classification
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save the model
model.save("models/densenet_model_v3.keras")
