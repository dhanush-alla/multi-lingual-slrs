"""
THE LIVE APP - Real-time sign language recognition using OpenCV and MediaPipe.
60 FPS interface with temporal smoothing, sentence building, and translation/TTS.
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter, deque
import time
import threading
import deep_translator
import pyttsx3
import os
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions


RSL_DISPLAY_MAP = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'YO',
    'Ж': 'ZH', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'KH', 'Ц': 'TS', 'Ч': 'CH', 'Ш': 'SH', 'Щ': 'SCH',
    'Ъ': 'HARD', 'Ы': 'YI', 'Ь': 'SOFT', 'Э': 'E', 'Ю': 'YU', 'Я': 'YA',
}


def label_for_display(label, current_model):
    """Return a screen-friendly token (OpenCV fonts do not support Cyrillic well)."""
    if label is None:
        return "None"
    if current_model == 'rsl':
        return RSL_DISPLAY_MAP.get(label, label)
    return label


def open_camera():
    """Try common camera indices and return first working capture."""
    backends = [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for idx in [0, 1, 2]:
        for backend in backends:
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FPS, 60)
                ok, _ = cap.read()
                if ok:
                    backend_name = "default" if backend is None else str(backend)
                    print(f"Using camera index {idx} (backend={backend_name})")
                    return cap
            cap.release()
    return None

def load_models():
    """
    Load trained models for ASL, ISL, and RSL.
    
    Returns:
        Dictionary of loaded models {language: model}
    """
    models = {}
    models['asl'] = tf.keras.models.load_model('models/model_asl.keras')
    models['isl'] = tf.keras.models.load_model('models/model_isl.keras')
    rsl_model_path = 'models/model_rsl_dynamic.keras'
    if os.path.exists(rsl_model_path):
        models['rsl'] = tf.keras.models.load_model(rsl_model_path)
    else:
        print("RSL model not found at models/model_rsl_dynamic.keras; RSL mode disabled.")
    return models


def extract_coords_from_result(hand_landmarks):
    """Convert MediaPipe landmarks to wrist-normalized 63D vector."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    coords = np.array(coords, dtype=np.float32)
    wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
    coords[::3] -= wrist_x
    coords[1::3] -= wrist_y
    coords[2::3] -= wrist_z
    return coords

def main():
    """
    Main live recognition loop using webcam.
    """
    # Initialize
    cap = open_camera()
    if cap is None:
        print("Could not open webcam (tried indices 0, 1, 2).")
        return
    
    model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    hands = HandLandmarker.create_from_options(options)
    
    models = load_models()
    
    # Classes
    asl_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['del', 'nothing', 'space']
    isl_classes = [chr(i) for i in range(ord('a'), ord('z')+1)]
    classes = {'asl': asl_classes, 'isl': isl_classes}

    rsl_classes_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'rsl_dynamic_classes.npy')
    if 'rsl' in models and os.path.exists(rsl_classes_path):
        classes['rsl'] = list(np.load(rsl_classes_path, allow_pickle=True))
    
    # State
    current_model = 'asl'
    prediction_buffer = deque(maxlen=10)
    prob_buffer = deque(maxlen=8)
    rsl_frame_buffer = deque(maxlen=30)  # Rolling 30-frame window for dynamic RSL LSTM inference
    last_prediction = None
    last_change_time = time.time()
    last_committed_token = None
    token_released_since_commit = True
    hold_threshold = 1.5  # seconds
    sentence = []
    word_buffer = []
    confidence_thresholds = {
        'asl': 0.8,
        'isl': 0.8,
        'rsl': 0.45,
    }
    
    # Translation and TTS
    translator = deep_translator.GoogleTranslator(source='en', target='te')  # Telugu; change to 'hi' for Hindi
    tts_thread = None

    def _speak(text):
        """Speak text using PowerShell SpeechSynthesizer — fully thread-safe on Windows."""
        if not text:
            return
        import subprocess
        # Escape single quotes for PowerShell
        safe = text.replace("'", "''")
        ps_cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"([System.Speech.Synthesis.SpeechSynthesizer]::new()).Speak('{safe}')"
        )
        t = threading.Thread(
            target=lambda: subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True
            ),
            daemon=True
        )
        t.start()
        return t
    
    print("Starting live recognition. Press '1' for ASL, '2' for ISL, '3' for RSL, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            results = hands.detect(mp_image)
        except Exception:
            results = None
        
        prediction = None
        confidence = 0.0
        if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
            coords = extract_coords_from_result(results.hand_landmarks[0])
        else:
            coords = None

        if current_model == 'rsl':
            # Dynamic sequence inference: push every frame (or zeros) into the
            # rolling 30-frame buffer then feed the full sequence to the LSTM.
            rsl_frame_buffer.append(
                coords if coords is not None else np.zeros(63, dtype=np.float32)
            )
            if len(rsl_frame_buffer) == 30:
                seq = np.array(list(rsl_frame_buffer), dtype=np.float32)
                pred = models['rsl'].predict(seq[np.newaxis], verbose=0)[0]
                prob_buffer.append(pred)
                avg_prob = np.mean(np.array(prob_buffer), axis=0)
                pred_class = int(np.argmax(avg_prob))
                confidence = float(np.max(avg_prob))
                if confidence > confidence_thresholds.get('rsl', 0.45) and pred_class < len(classes['rsl']):
                    prediction = classes['rsl'][pred_class]
            if coords is None:
                prob_buffer.clear()
        else:
            if coords is not None:
                pred = models[current_model].predict(np.expand_dims(coords, 0), verbose=0)[0]
                prob_buffer.append(pred)
                avg_prob = np.mean(np.array(prob_buffer), axis=0)
                pred_class = int(np.argmax(avg_prob))
                confidence = float(np.max(avg_prob))
                threshold = confidence_thresholds.get(current_model, 0.8)
                if confidence > threshold and pred_class < len(classes[current_model]):
                    prediction = classes[current_model][pred_class]
            else:
                prob_buffer.clear()
        
        # Temporal smoothing
        prediction_buffer.append(prediction)
        if prediction_buffer:
            smoothed = Counter(prediction_buffer).most_common(1)[0][0]
        else:
            smoothed = None
        
        # Sentence engine
        current_time = time.time()
        if smoothed in [None, 'nothing']:
            # Re-arm same-token acceptance after user removes gesture.
            token_released_since_commit = True

        if smoothed != last_prediction:
            last_prediction = smoothed
            last_change_time = current_time
        elif smoothed is not None and (current_time - last_change_time) > hold_threshold:
            # Block repeated appending of the same token unless gesture was released.
            if smoothed == last_committed_token and not token_released_since_commit:
                pass
            else:
                if smoothed == 'space':
                    if word_buffer:
                        word = ''.join(word_buffer).upper() if current_model == 'asl' else ''.join(word_buffer)
                        sentence.append(word)
                        word_buffer = []
                        # Translate and speak
                        try:
                            translated = translator.translate(word)
                            _speak(translated)
                        except Exception as e:
                            print(f"Translation/TTS error: {e}")
                elif smoothed == 'del':
                    if word_buffer:
                        word_buffer.pop()
                elif smoothed not in ['nothing']:
                    word_buffer.append(smoothed)

                last_committed_token = smoothed
                token_released_since_commit = False
        
        # Display
        display_current = label_for_display(smoothed, current_model)
        display_word = ''.join([label_for_display(ch, current_model) for ch in word_buffer])

        cv2.putText(frame, f'Model: {current_model.upper()} (1/2/3) | p=play | c=clear | q=quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f'Current: {display_current}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f'Conf: {confidence:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
        cv2.putText(frame, f'Word: {display_word}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f'Sentence: {" ".join(sentence)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        cv2.imshow('Sign Language Translator', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_model = 'asl'
            word_buffer = []
            prediction_buffer.clear()
            last_prediction = None
            last_committed_token = None
            token_released_since_commit = True
            last_change_time = time.time()
            prob_buffer.clear()
            rsl_frame_buffer.clear()
        elif key == ord('2'):
            current_model = 'isl'
            word_buffer = []
            prediction_buffer.clear()
            last_prediction = None
            last_committed_token = None
            token_released_since_commit = True
            last_change_time = time.time()
            prob_buffer.clear()
            rsl_frame_buffer.clear()
            prob_buffer.clear()
        elif key == ord('3'):
            if 'rsl' in models and 'rsl' in classes:
                current_model = 'rsl'
                word_buffer = []
                prediction_buffer.clear()
                last_prediction = None
                last_committed_token = None
                token_released_since_commit = True
                last_change_time = time.time()
                prob_buffer.clear()
                rsl_frame_buffer.clear()
            else:
                print("RSL model/classes unavailable. Ensure model_rsl_dynamic.keras and rsl_dynamic_classes.npy exist.")
        elif key == ord('p'):
            full_sentence = ' '.join(sentence)
            if word_buffer:
                full_sentence = (full_sentence + ' ' + ''.join(word_buffer)).strip()
            if full_sentence:
                _speak(full_sentence)
        elif key == ord('c'):
            sentence = []
            word_buffer = []
            prediction_buffer.clear()
            last_prediction = None
            last_committed_token = None
            token_released_since_commit = True
            last_change_time = time.time()
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
