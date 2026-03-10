"""
Script to convert images / video landmarks -> .npy arrays using MediaPipe.
Processes ASL and ISL static-image datasets, and the RSL Slovo dynamic dataset.
"""

import os

# Prevent BLAS/OMP thread oversubscription that can trigger RAM exhaustion
# when combined with multiprocessing workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from tqdm import tqdm
import concurrent.futures

MAX_WORKERS = min(6, os.cpu_count() or 1)

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
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1,
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


def read_image_unicode_safe(image_path):
    """Read image robustly on Windows paths containing Unicode folder names."""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
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
            image = read_image_unicode_safe(image_path)
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

    # Fallback detector path for cropped/static images where task API misses.
    try:
        image = read_image_unicode_safe(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
        ) as hands:
            res = hands.process(image_rgb)

        if not res.multi_hand_landmarks:
            return None

        coords = []
        for lm in res.multi_hand_landmarks[0].landmark:
            coords.extend([lm.x, lm.y, lm.z])

        coords = np.array(coords, dtype=np.float32)
        wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
        coords[::3] -= wrist_x
        coords[1::3] -= wrist_y
        coords[2::3] -= wrist_z
        return coords
    except Exception:
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


def _frame_to_63(frame):
    """Convert a single Slovo JSON frame dict to a 63-float wrist-relative array.

    Returns a zero array if no hand landmarks are present in the frame.
    """
    hand_data = frame.get('hand 1') or frame.get('hand 2')
    if not hand_data or len(hand_data) < 21:
        return np.zeros(63, dtype=np.float32)

    coords = np.array(
        [[lm['x'], lm['y'], lm['z']] for lm in hand_data], dtype=np.float32
    ).flatten()
    wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
    coords[0::3] -= wrist_x
    coords[1::3] -= wrist_y
    coords[2::3] -= wrist_z
    return coords


def _build_sequence(feature_frames, num_frames=30, num_features=63):
    """Return a (num_frames, num_features) float32 array from a list of frame arrays.

    - len >= num_frames : take the middle num_frames (temporal centre-crop)
    - len <  num_frames : zero-pad at the end
    - all zeros         : return None (no hand ever detected)
    """
    if not feature_frames:
        return None

    arr = np.array(feature_frames, dtype=np.float32)
    if not np.any(arr):          # every frame was zero — skip
        return None

    n = len(arr)
    if n >= num_frames:
        start = (n - num_frames) // 2
        return arr[start : start + num_frames]
    else:
        pad = np.zeros((num_frames - n, num_features), dtype=np.float32)
        return np.vstack([arr, pad])


def _augment_sequence(seq):
    """Return 3 augmented copies of a (T, 63) wrist-relative landmark sequence.

    Augmentations:
      1. Horizontal flip — negates all x coordinates (mirrors the gesture).
      2. Gaussian noise  — small perturbation (σ=0.005) to all coordinates.
      3. Time shift      — rolls the sequence by a random ±3-frame offset.
    """
    rng = np.random.default_rng()
    aug = []

    # 1. Horizontal flip: x is at indices 0, 3, 6, … in the 63-d vector
    flipped = seq.copy()
    flipped[:, 0::3] = -flipped[:, 0::3]
    aug.append(flipped)

    # 2. Gaussian noise
    noisy = (seq + rng.normal(0.0, 0.005, seq.shape)).astype(np.float32)
    aug.append(noisy)

    # 3. Random time shift (wrap-around is harmless — edges are zero-padded)
    shift = int(rng.integers(-3, 4))
    aug.append(np.roll(seq, shift, axis=0).astype(np.float32))

    return aug


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
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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


def preprocess_dataset_with_mapping(data_dir, output_prefix, class_to_idx):
    """Process a dataset using a provided class-to-index mapping.

    This is used for RSL so train and test share identical label IDs.
    """
    landmarks_list = []
    labels_list = []

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    tasks = []

    for cls in classes:
        if cls not in class_to_idx:
            print(f"Skipping unknown class folder in {output_prefix}: {cls}")
            continue
        cls_dir = os.path.join(data_dir, cls)
        label = class_to_idx[cls]
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in img_files:
            tasks.append((os.path.join(cls_dir, img_file), label, cls))

    print(f"Processing {output_prefix} dataset from {data_dir}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_process_image, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=output_prefix, unit="img"):
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


def process_rsl_dynamic_data(rsl_dir):
    """Stream slovo_mediapipe.json + annotations.csv and build Top-50 dynamic .npy arrays.

    Outputs:
      data/processed/rsl_dynamic_landmarks.npy  shape (N, 30, 63)  float32
      data/processed/rsl_dynamic_labels.npy     shape (N,)         int32
      data/processed/rsl_dynamic_classes.npy    shape (50,)        str   (class names in label order)
    """
    try:
        import ijson
    except ImportError:
        raise ImportError(
            "ijson is required for RSL dynamic preprocessing. "
            "Install with: pip install ijson"
        )
    import csv as _csv
    from collections import Counter

    json_path = os.path.join(rsl_dir, 'slovo_mediapipe.json')
    csv_path  = os.path.join(rsl_dir, 'annotations.csv')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"slovo_mediapipe.json not found at: {json_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"annotations.csv not found at: {csv_path}")

    # ------------------------------------------------------------------
    # Step 1: Parse annotations.csv → Top-50 labels + per-video metadata
    # ------------------------------------------------------------------
    print("Parsing annotations.csv ...")
    label_counts = Counter()
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as fh:
        reader = _csv.DictReader(fh, delimiter='\t')
        for row in reader:
            label_counts[row['text']] += 1
            rows.append(row)

    TOP_N = 20
    # Exclude 'no_event' (background/non-sign noise), then take the most frequent real signs.
    top50_labels = [
        label for label, _ in label_counts.most_common(TOP_N + 10)
        if label != 'no_event'
    ][:TOP_N]
    top50_set    = set(top50_labels)
    label_to_idx = {label: idx for idx, label in enumerate(top50_labels)}
    print(f"Top-{TOP_N} labels selected (excluding no_event). Most common: {top50_labels[:5]}")

    # attachment_id -> (label_idx, begin_frame, end_frame)
    video_meta = {}
    for row in rows:
        text = row['text']
        if text not in top50_set:
            continue
        vid_id = row['attachment_id']
        begin  = int(row['begin'])
        end    = int(row['end'])
        video_meta[vid_id] = (label_to_idx[text], begin, end)

    print(f"Videos to process: {len(video_meta)} (Top-50 classes)")

    # ------------------------------------------------------------------
    # Step 2: Stream JSON and extract 30-frame sequences
    # ------------------------------------------------------------------
    NUM_FRAMES   = 30
    NUM_FEATURES = 63  # 21 landmarks × 3 coords

    landmarks_list = []
    labels_list    = []
    skipped        = 0

    print("Streaming slovo_mediapipe.json (may take a few minutes) ...")
    with open(json_path, 'rb') as fh:
        for vid_id, frame_list in tqdm(
            ijson.kvitems(fh, ''), desc="Videos", unit="vid"
        ):
            if vid_id not in video_meta:
                continue

            label_idx, begin, end = video_meta[vid_id]

            # Slice to the annotated gesture window
            gesture_frames = frame_list[begin : end + 1]

            # Convert each frame to a 63-d feature vector
            feature_frames = [_frame_to_63(f) for f in gesture_frames]

            seq = _build_sequence(feature_frames, NUM_FRAMES, NUM_FEATURES)
            if seq is None:
                skipped += 1
                continue

            landmarks_list.append(seq)
            labels_list.append(label_idx)

    print(f"Extracted {len(landmarks_list)} sequences, skipped {skipped}")

    # Augment each sequence with 3 variants (4× total data).
    aug_landmarks: list = []
    aug_labels:    list = []
    for seq, lbl in zip(landmarks_list, labels_list):
        aug_landmarks.append(seq)
        aug_labels.append(lbl)
        for aug_seq in _augment_sequence(seq):
            aug_landmarks.append(aug_seq)
            aug_labels.append(lbl)
    print(f"After augmentation: {len(aug_landmarks)} sequences ({len(aug_landmarks) // max(len(landmarks_list),1)}x)")

    landmarks_array = np.array(aug_landmarks, dtype=np.float32)
    labels_array    = np.array(aug_labels,    dtype=np.int32)

    processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    np.save(os.path.join(processed_dir, 'rsl_dynamic_landmarks.npy'), landmarks_array)
    np.save(os.path.join(processed_dir, 'rsl_dynamic_labels.npy'),    labels_array)
    np.save(os.path.join(processed_dir, 'rsl_dynamic_classes.npy'),   np.array(top50_labels))

    print(f"Saved rsl_dynamic_landmarks.npy : {landmarks_array.shape}")
    print(f"Saved rsl_dynamic_labels.npy    : {labels_array.shape}")
    print(f"Saved rsl_dynamic_classes.npy   : {len(top50_labels)} classes")

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

    # RSL Dynamic (Slovo Top-50 LSTM dataset)
    rsl_dir = os.path.join(base_dir, 'data', 'raw', 'RSL')
    rsl_dynamic_landmarks = os.path.join(canonical_processed_dir, 'rsl_dynamic_landmarks.npy')
    rsl_dynamic_labels    = os.path.join(canonical_processed_dir, 'rsl_dynamic_labels.npy')

    # Always regenerate RSL dynamic data to pick up pipeline changes.
    process_rsl_dynamic_data(rsl_dir)
