"""
Training script - GPU optimized for NVIDIA RTX 3070 Ti.
Trains ASL and ISL models with mixed precision and OneDeviceStrategy.
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM, Masking, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Let TensorFlow grow GPU memory usage on demand instead of reserving all VRAM up front.
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        # Safe to ignore when runtime has already initialized GPU context.
        pass

# Use available CPU cores for TensorFlow ops/input pipeline support.
_cores = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(_cores)
tf.config.threading.set_inter_op_parallelism_threads(max(2, _cores // 2))

# GPU Strategy
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")


def parse_args():
    """Parse CLI args to control which dataset models are trained."""
    parser = argparse.ArgumentParser(description="Train sign-language models")
    parser.add_argument(
        "--dataset",
        choices=["rsl", "asl", "isl", "all"],
        default="rsl",
        help="Dataset to train. Defaults to rsl to avoid retraining ASL/ISL unintentionally.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for training/inference datasets.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs before early stopping.",
    )
    return parser.parse_args()


class EpochPerfLogger(Callback):
    """Logs per-epoch wall time and current GPU memory usage for quick perf visibility."""

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._start
        msg = f"[Perf] epoch={epoch + 1} time={elapsed:.2f}s"
        try:
            mem = tf.config.experimental.get_memory_info('GPU:0')
            current_gb = mem['current'] / (1024 ** 3)
            peak_gb = mem['peak'] / (1024 ** 3)
            msg += f" gpu_mem_current={current_gb:.2f}GB gpu_mem_peak={peak_gb:.2f}GB"
        except Exception:
            msg += " gpu_mem=unavailable"
        print(msg)


def load_npy_with_fallback(base_dir, filename, candidates):
    """Load an .npy file from canonical path first, then legacy paths.

    Skips empty/corrupted files to avoid training crashes from partial outputs.
    """
    for rel_dir in candidates:
        path = os.path.join(base_dir, rel_dir, filename)
        if os.path.exists(path):
            try:
                arr = np.load(path)
                if arr.size == 0:
                    print(f"Skipping empty {filename}: {path}")
                    continue
                print(f"Loading {filename} from: {path}")
                return arr
            except Exception as e:
                print(f"Skipping unreadable {filename}: {path} ({e})")
                continue
    raise FileNotFoundError(
        f"Could not find {filename}. Checked: "
        + ", ".join([os.path.join(base_dir, d, filename) for d in candidates])
    )

def build_model(num_classes, input_dim=63, include_top5=False):
    """
    Build the Sequential model for sign language recognition.
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    with strategy.scope():
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax', dtype='float32')  # Ensure output is float32
        ])
        
        metrics = ['accuracy']
        if include_top5:
            metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy'))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
        
        return model


def build_sequence_model(input_shape=(30, 63), num_classes=50):
    """Builds an LSTM network for dynamic sequence recognition."""
    with strategy.scope():
        model = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.0),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dense(num_classes, activation='softmax', dtype='float32')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


def prepare_asl_split(asl_train_features, asl_train_labels, asl_test_features, asl_test_labels):
    """Use external ASL test set only if it is sufficiently large; else build stratified split.

    A tiny external test set can produce misleading metrics. This fallback creates a
    robust split from the large ASL training pool.
    """
    num_classes = len(np.unique(asl_train_labels))
    min_reliable_test_samples = num_classes * 50  # target >= 50 samples/class

    if len(asl_test_labels) >= min_reliable_test_samples:
        print(
            f"Using provided ASL test set ({len(asl_test_labels)} samples)."
        )
        return asl_train_features, asl_train_labels, asl_test_features, asl_test_labels

    print(
        f"ASL test set too small ({len(asl_test_labels)} samples). "
        f"Creating stratified split from ASL train set for reliable evaluation."
    )
    X_train_asl, X_test_asl, y_train_asl, y_test_asl = train_test_split(
        asl_train_features,
        asl_train_labels,
        test_size=0.1,
        random_state=42,
        stratify=asl_train_labels,
    )
    print(
        f"ASL stratified split created: train={len(y_train_asl)}, test={len(y_test_asl)}"
    )
    return X_train_asl, y_train_asl, X_test_asl, y_test_asl

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_path, dataset_name):
    """
    Train the model and evaluate with classification report and confusion matrix.
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_path: Path to save the model
        dataset_name: Name for printing
    """
    # Callbacks
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')
    
    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # Will stop early
        batch_size=512,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Load best model
    model.load_weights(model_path)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y_test, y_pred_classes, zero_division=0))
    
    print(f"\n{dataset_name} Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
if __name__ == "__main__":
    args = parse_args()
    selected = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs

    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Canonical first, then legacy locations from earlier preprocess versions.
    search_dirs = [
        os.path.join('data', 'processed'),
        os.path.join('data', 'raw', 'processed'),
        os.path.join('data', 'raw', 'ASL', 'processed'),
    ]
    
    if selected in ("asl", "all"):
        # ASL Training and Testing
        asl_train_features = load_npy_with_fallback(base_dir, 'asl_train_landmarks.npy', search_dirs)
        asl_train_labels = load_npy_with_fallback(base_dir, 'asl_train_labels.npy', search_dirs)
        asl_test_features = load_npy_with_fallback(base_dir, 'asl_test_landmarks.npy', search_dirs)
        asl_test_labels = load_npy_with_fallback(base_dir, 'asl_test_labels.npy', search_dirs)

        X_train_asl, y_train_asl, X_test_asl, y_test_asl = prepare_asl_split(
            asl_train_features,
            asl_train_labels,
            asl_test_features,
            asl_test_labels,
        )

        num_classes_asl = len(np.unique(asl_train_labels))
        model_asl = build_model(num_classes_asl)

        model_asl_path = os.path.join(models_dir, 'model_asl.keras')
        train_and_evaluate(model_asl, X_train_asl, y_train_asl, X_test_asl, y_test_asl, model_asl_path, "ASL")

    if selected in ("isl", "all"):
        # ISL Training and Testing (80/20 split)
        isl_features = load_npy_with_fallback(base_dir, 'isl_landmarks.npy', search_dirs)
        isl_labels = load_npy_with_fallback(base_dir, 'isl_labels.npy', search_dirs)

        X_train_isl, X_test_isl, y_train_isl, y_test_isl = train_test_split(
            isl_features, isl_labels, test_size=0.2, random_state=42, stratify=isl_labels
        )

        num_classes_isl = len(np.unique(isl_labels))
        model_isl = build_model(num_classes_isl)

        model_isl_path = os.path.join(models_dir, 'model_isl.keras')
        train_and_evaluate(model_isl, X_train_isl, y_train_isl, X_test_isl, y_test_isl, model_isl_path, "ISL")

    if selected in ("rsl", "all"):
        # RSL Dynamic Training and Testing
        print("\nLoading dynamic RSL data...")
        X_rsl = load_npy_with_fallback(base_dir, 'rsl_dynamic_landmarks.npy', search_dirs)
        y_rsl = load_npy_with_fallback(base_dir, 'rsl_dynamic_labels.npy', search_dirs)
        rsl_classes = load_npy_with_fallback(base_dir, 'rsl_dynamic_classes.npy', search_dirs)

        X_train_rsl, X_test_rsl, y_train_rsl, y_test_rsl = train_test_split(
            X_rsl, y_rsl, test_size=0.2, random_state=42, stratify=y_rsl
        )

        train_ds_rsl = (
            tf.data.Dataset.from_tensor_slices((X_train_rsl, y_train_rsl))
            .cache()
            .shuffle(buffer_size=len(y_train_rsl), reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds_rsl = (
            tf.data.Dataset.from_tensor_slices((X_test_rsl, y_test_rsl))
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Use class count from saved classes array — avoids off-by-one when
        # some label indices have no samples (len(np.unique) < max_label + 1).
        num_classes_rsl = len(rsl_classes)
        input_shape_rsl = (X_rsl.shape[1], X_rsl.shape[2])  # Should be (30, 63)

        model_rsl = build_sequence_model(input_shape=input_shape_rsl, num_classes=num_classes_rsl)
        model_rsl_path = os.path.join(models_dir, 'model_rsl_dynamic.keras')

        early_stop_rsl = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
        checkpoint_rsl = ModelCheckpoint(model_rsl_path, save_best_only=True, monitor='val_accuracy', mode='max')
        reduce_lr_rsl = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
        perf_logger_rsl = EpochPerfLogger()

        model_rsl.fit(
            train_ds_rsl,
            validation_data=test_ds_rsl,
            epochs=epochs,
            callbacks=[early_stop_rsl, checkpoint_rsl, reduce_lr_rsl, perf_logger_rsl],
            verbose=1,
        )

        model_rsl.load_weights(model_rsl_path)
        y_pred_rsl = model_rsl.predict(test_ds_rsl)
        y_pred_rsl_classes = np.argmax(y_pred_rsl, axis=1)

        eval_loss, eval_acc = model_rsl.evaluate(test_ds_rsl, verbose=0)
        print(f"\nRSL Dynamic Eval Loss: {eval_loss:.4f}")
        print(f"RSL Dynamic Eval Accuracy: {eval_acc:.4f}")
        print("\nRSL Classification Report:")
        print(classification_report(y_test_rsl, y_pred_rsl_classes, zero_division=0))
