"""
Training script - GPU optimized for NVIDIA RTX 3070 Ti.
Trains ASL and ISL models with mixed precision and OneDeviceStrategy.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Use available CPU cores for TensorFlow ops/input pipeline support.
_cores = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(_cores)
tf.config.threading.set_inter_op_parallelism_threads(max(2, _cores // 2))

# GPU Strategy
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")


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

def build_model(num_classes):
    """
    Build the Sequential model for sign language recognition.
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    with strategy.scope():
        model = Sequential([
            Input(shape=(63,)),
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
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Canonical first, then legacy locations from earlier preprocess versions.
    search_dirs = [
        os.path.join('data', 'processed'),
        os.path.join('data', 'raw', 'processed'),
        os.path.join('data', 'raw', 'ASL', 'processed'),
    ]
    
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
