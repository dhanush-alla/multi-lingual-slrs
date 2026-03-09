"""
Script to convert images -> landmarks using MediaPipe.
Processes ASL and ISL datasets for Phase 1.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from tqdm import tqdm
import concurrent.futures

# Download or provide path to hand landmarker model
MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
# For now, assume it's downloaded to the project dir
model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')

# If not exists, download
if not os.path.exists(model_path):
    import urllib.request
    urllib.request.urlretrieve(MODEL_PATH, model_path)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)


def is_valid_npy(path, min_samples=1):
    """Return True when a .npy file exists and can be loaded with enough samples."""
    if not os.path.exists(path):
        return False
    try:
        arr = np.load(path)
        return arr.size >= min_samples
    except Exception:
        return False


def first_valid_path(paths, min_samples=1):
    """Return first valid .npy path from candidates, else None."""
    for p in paths:
        if is_valid_npy(p, min_samples=min_samples):
            return p
    return None

def extract_hand_landmarks(image_path):
    """
    Extract 21 3D hand landmarks from an image using MediaPipe HandLandmarker.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Flattened 63-feature array (21 landmarks * 3 coords) normalized relative to wrist,
        or None if no hand detected or an error occurred
    """
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create MP Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect
            result = landmarker.detect(mp_image)
    except Exception as e:
        # Catch any MediaPipe/runtime errors and return None
        print(f"Warning: failed to process {image_path}: {e}")
        return None
    
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]
        
        # Extract coordinates
        coords = []
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        
        coords = np.array(coords)
        
        # Normalize relative to wrist (first landmark)
        wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
        coords[::3] -= wrist_x  # x coords
        coords[1::3] -= wrist_y  # y coords
        coords[2::3] -= wrist_z  # z coords
        
        return coords
    else:
        # no landmarks detected
        return None

def _process_image(task):
    """Worker for parallel processing.

    Args:
        task: tuple (img_path, label, cls)
    Returns:
        (landmarks_array, label) or None if should be skipped
    """
    img_path, label, cls = task
    landmarks = extract_hand_landmarks(img_path)
    if landmarks is None:
        if cls.lower() == 'nothing':
            return (np.zeros(21 * 3, dtype=np.float32), label)
        else:
            # skip this one
            return None
    return (landmarks, label)


def preprocess_asl_test_flat(test_dir, output_prefix, class_to_idx):
    """Process flat ASL test folder where labels are encoded in filenames.

    Expected format: <label>_test.jpg, e.g. A_test.jpg, nothing_test.jpg.
    """
    landmarks_list = []
    labels_list = []

    img_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tasks = []
    for img_file in img_files:
        label_token = img_file.split('_')[0]
        if label_token not in class_to_idx:
            print(f"Skipped image {img_file} - unknown label token '{label_token}'")
            continue
        tasks.append((os.path.join(test_dir, img_file), class_to_idx[label_token], label_token))

    print(f"Processing {output_prefix} flat test dataset from {test_dir}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_process_image, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="ASL test", unit="img"):
            res = f.result()
            if res is not None:
                landmarks_list.append(res[0])
                labels_list.append(res[1])

    landmarks_array = np.array(landmarks_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int32)

    processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    np.save(os.path.join(processed_dir, f'{output_prefix}_landmarks.npy'), landmarks_array)
    np.save(os.path.join(processed_dir, f'{output_prefix}_labels.npy'), labels_array)

    print(f"Saved {len(landmarks_array)} samples for {output_prefix}")
    print(f"Landmarks shape: {landmarks_array.shape}, Labels shape: {labels_array.shape}")


def preprocess_dataset(data_dir, output_prefix):
    """
    Process a dataset directory and save landmarks and labels.
    
    Args:
        data_dir: Directory containing class folders
        output_prefix: Prefix for output files (e.g., 'asl_train')
    """
    landmarks_list = []
    labels_list = []
    
    # Get class folders (A-Z)
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"Processing {output_prefix} dataset from {data_dir}")
    
    # Assemble tasks for parallel executor
    tasks = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        label = class_to_idx[cls]
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in img_files:
            tasks.append((os.path.join(cls_dir, img_file), label, cls))
    
    # Use all CPU cores to process images
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_process_image, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Images", unit="img"):
            res = f.result()
            if res is not None:
                landmarks_list.append(res[0])
                labels_list.append(res[1])
    
    # Convert to numpy arrays
    landmarks_array = np.array(landmarks_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int32)
    
    # Save to a single canonical processed directory used by training.
    processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, f'{output_prefix}_landmarks.npy'), landmarks_array)
    np.save(os.path.join(processed_dir, f'{output_prefix}_labels.npy'), labels_array)
    
    print(f"Saved {len(landmarks_array)} samples for {output_prefix}")
    print(f"Landmarks shape: {landmarks_array.shape}, Labels shape: {labels_array.shape}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    canonical_processed_dir = os.path.join(base_dir, 'data', 'processed')
    legacy_asl_processed_dir = os.path.join(base_dir, 'data', 'raw', 'ASL', 'processed')
    legacy_raw_processed_dir = os.path.join(base_dir, 'data', 'raw', 'processed')
    
    # ASL Training (nested folder structure)
    asl_train_dir = os.path.join(base_dir, 'data', 'raw', 'ASL', 'asl_alphabet_train', 'asl_alphabet_train')
    asl_classes = sorted([d for d in os.listdir(asl_train_dir) if os.path.isdir(os.path.join(asl_train_dir, d))])
    asl_class_to_idx = {cls: idx for idx, cls in enumerate(asl_classes)}

    asl_train_landmarks = os.path.join(canonical_processed_dir, 'asl_train_landmarks.npy')
    asl_train_legacy = [
        os.path.join(legacy_asl_processed_dir, 'asl_train_landmarks.npy'),
        os.path.join(legacy_raw_processed_dir, 'asl_train_landmarks.npy'),
    ]
    if not is_valid_npy(asl_train_landmarks) and not first_valid_path(asl_train_legacy, min_samples=1):
        preprocess_dataset(asl_train_dir, 'asl_train')
    else:
        print("ASL training data already exists, skipping.")
    
    # ASL Testing (flat files inside subfolder)
    asl_test_dir = os.path.join(base_dir, 'data', 'raw', 'ASL', 'asl_alphabet_test', 'asl_alphabet_test')
    asl_test_landmarks = os.path.join(canonical_processed_dir, 'asl_test_landmarks.npy')
    # Require at least 20 samples for ASL test to avoid tiny/partial test files.
    if not is_valid_npy(asl_test_landmarks, min_samples=20):
        preprocess_asl_test_flat(asl_test_dir, 'asl_test', asl_class_to_idx)
    else:
        print("ASL test data already exists, skipping.")
    
    # ISL Training (full dataset, split later)
    isl_dir = os.path.join(base_dir, 'data', 'raw', 'ISL', 'isl_alphabet_images')
    isl_landmarks = os.path.join(canonical_processed_dir, 'isl_landmarks.npy')
    isl_legacy = [
        os.path.join(legacy_raw_processed_dir, 'isl_landmarks.npy'),
    ]
    if not is_valid_npy(isl_landmarks) and not first_valid_path(isl_legacy, min_samples=1):
        preprocess_dataset(isl_dir, 'isl')
    else:
        print("ISL data already exists, skipping.")
