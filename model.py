import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

MobileNetV2 = keras.applications.MobileNetV2
layers = keras.layers
models = keras.models

# Image loader with error handling
def _load_image(path, img_size=(224, 224)):
    img = tf.io.read_file(path)
    try:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
    except Exception as e:
        tf.print("⚠️ Could not load:", path, "Error:", e)
        img = tf.zeros([img_size[0], img_size[1], 3])  # fallback black image
    return img

# Data augmentation
def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

class GarbagePercentagePredictor:
    def __init__(self, model_path="models/garbage_model.keras",
                 image_dir="data/raw/images", labels_csv="data/labels.csv"):
        self.model_path = model_path
        self.image_dir = image_dir
        self.labels_csv = labels_csv
        self.model = None

    # Build MobileNetV2 regression model
    def create_model(self, img_size=(224, 224)):
        base = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(img_size[0], img_size[1], 3))
        base.trainable = False

        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="linear")(x)

        model = models.Model(inputs=base.input, outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="mse", metrics=["mae"])
        self.model = model
        return model

    # Build dataset pipeline
    def _build_dataset(self, df, img_size=(224, 224), augment=False,
                       batch_size=32, shuffle=True):
        paths = [os.path.join(self.image_dir, fn) for fn in df["filename"].tolist()]
        labels = df["percentage"].astype("float32").values

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _map(path, y):
            img = _load_image(path, img_size)
            if augment:
                img = _augment(img)
            return img, y

        if shuffle:
            ds = ds.shuffle(buffer_size=min(1000, len(paths)))
        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # Training pipeline
    def train_model(self, epochs=50, img_size=(224, 224), batch_size=32):
        df = pd.read_csv(self.labels_csv)
        df = df.dropna()
        df = df[df["filename"].astype(str).str.len() > 0]

        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        train_ds = self._build_dataset(train_df, img_size, augment=True,
                                       batch_size=batch_size, shuffle=True)
        val_ds = self._build_dataset(val_df, img_size, augment=False,
                                     batch_size=batch_size, shuffle=False)

        if self.model is None:
            self.create_model(img_size)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                             monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                               monitor="val_mae",
                                               save_best_only=True,
                                               mode="min")
        ]

        # Initial training
        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                       callbacks=callbacks, verbose=1)

        # Fine-tuning
        self.model.layers[0].trainable = True
        for layer in self.model.layers[0].layers[:-30]:
            layer.trainable = False

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                           loss="mse", metrics=["mae"])
        self.model.fit(train_ds, validation_data=val_ds,
                       epochs=max(5, epochs // 3), callbacks=callbacks, verbose=1)

        # Save model in .keras format
        self.save_model()

    # Save model in .keras format
    def save_model(self):
        if self.model is not None:
            self.model.save(self.model_path, save_format="keras")
            print(f"✅ Model saved as {self.model_path}")
        else:
            print("⚠️ No model to save!")

    # Load model from .keras file
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ Model file not found at {self.model_path}")

    # Preprocess single image
    def _preprocess_single(self, path, img_size=(224, 224)):
        img = _load_image(path, img_size)
        return tf.expand_dims(img, 0)

    # Prediction with test-time augmentation
    def predict_percent_and_confidence(self, path, img_size=(224, 224), tta_runs=8):
        if self.model is None:
            self.load_model()
        preds = []
        base = self._preprocess_single(path, img_size)[0]
        for _ in range(tta_runs):
            x = base
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.random_brightness(x, 0.1)
            x = tf.image.random_contrast(x, 0.95, 1.05)
            x = tf.expand_dims(x, 0)
            y = self.model.predict(x, verbose=0)[0][0]
            preds.append(float(y))

        preds = np.array(preds)
        mean_pct = float(np.clip(preds.mean(), 0, 100))
        std = float(preds.std())
        confidence = float(max(0.0, 100.0 - std * 10.0))
        return mean_pct, confidence
