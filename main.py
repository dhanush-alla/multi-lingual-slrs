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
    Load trained models for ASL and ISL.
    
    Returns:
        Dictionary of loaded models {language: model}
    """
    models = {}
    models['asl'] = tf.keras.models.load_model('models/model_asl.keras')
    models['isl'] = tf.keras.models.load_model('models/model_isl.keras')
    return models

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
        num_hands=1,
    )
    hands = HandLandmarker.create_from_options(options)
    
    models = load_models()
    
    # Classes
    asl_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['space', 'nothing', 'del']
    isl_classes = [chr(i) for i in range(ord('a'), ord('z')+1)]
    classes = {'asl': asl_classes, 'isl': isl_classes}
    
    # State
    current_model = 'asl'
    prediction_buffer = deque(maxlen=10)
    last_prediction = None
    last_change_time = time.time()
    last_committed_token = None
    token_released_since_commit = True
    hold_threshold = 1.5  # seconds
    sentence = []
    word_buffer = []
    
    # Translation and TTS
    translator = deep_translator.GoogleTranslator(source='en', target='te')  # Telugu; change to 'hi' for Hindi
    try:
        tts_engine = pyttsx3.init()
    except Exception as e:
        print(f"TTS init failed: {e}. Continuing without voice output.")
        tts_engine = None
    tts_thread = None
    
    print("Starting live recognition. Press '1' for ASL, '2' for ISL, 'q' to quit.")
    
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
        if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
            landmarks = results.hand_landmarks[0]
            
            # Extract coordinates
            coords = []
            for lm in landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)
            
            # Normalize relative to wrist
            wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
            coords[::3] -= wrist_x
            coords[1::3] -= wrist_y
            coords[2::3] -= wrist_z
            
            # Predict
            pred = models[current_model].predict(np.expand_dims(coords, 0), verbose=0)
            pred_class = np.argmax(pred)
            confidence = np.max(pred)
            
            if confidence > 0.8:  # Confidence threshold
                prediction = classes[current_model][pred_class]
        
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
                            if tts_engine is not None and tts_thread and tts_thread.is_alive():
                                tts_thread.join()
                            if tts_engine is not None:
                                tts_thread = threading.Thread(target=lambda: (tts_engine.say(translated), tts_engine.runAndWait()))
                                tts_thread.start()
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
        cv2.putText(frame, f'Model: {current_model.upper()} (1/2 to switch)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f'Current: {smoothed or "None"}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f'Word: {"".join(word_buffer)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f'Sentence: {" ".join(sentence)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        cv2.imshow('Sign Language Translator', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_model = 'asl'
            word_buffer = []  # Reset on switch
            prediction_buffer.clear()
            last_prediction = None
            last_committed_token = None
            token_released_since_commit = True
            last_change_time = time.time()
        elif key == ord('2'):
            current_model = 'isl'
            word_buffer = []  # Reset on switch
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
    if tts_thread and tts_thread.is_alive():
        tts_thread.join()

if __name__ == "__main__":
    main()
