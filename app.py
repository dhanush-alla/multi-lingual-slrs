# -*- coding: utf-8 -*-
"""
Flask-SocketIO backend for the Sign Language Translator web app.
Handles MediaPipe inference and model prediction; browser handles camera & UI.
"""

import os
import base64
import time
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import deep_translator

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'slp_secret_key'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models…")
_models = {}
_models['asl'] = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'model_asl.keras'))
_models['isl'] = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'model_isl.keras'))
_rsl_path = os.path.join(BASE_DIR, 'models', 'model_rsl_dynamic.keras')
if os.path.exists(_rsl_path):
    _models['rsl'] = tf.keras.models.load_model(_rsl_path)
    print("RSL model loaded.")
else:
    print("WARNING: RSL model not found — RSL mode disabled.")
print("Models ready.")

# ── Classes ───────────────────────────────────────────────────────────────────
_asl_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['del', 'nothing', 'space']
_isl_classes = [chr(i) for i in range(ord('a'), ord('z') + 1)]
CLASSES = {'asl': _asl_classes, 'isl': _isl_classes}

_rsl_classes_path = os.path.join(BASE_DIR, 'data', 'processed', 'rsl_dynamic_classes.npy')
if 'rsl' in _models and os.path.exists(_rsl_classes_path):
    CLASSES['rsl'] = list(np.load(_rsl_classes_path, allow_pickle=True))

# ── MediaPipe HandLandmarker (shared, single-user local use) ──────────────────
_task_path = os.path.join(BASE_DIR, 'hand_landmarker.task')
_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_task_path),
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1,
)
hand_landmarker = HandLandmarker.create_from_options(_options)

# ── Constants ─────────────────────────────────────────────────────────────────
RSL_DISPLAY_MAP = {
    'А': 'A',  'Б': 'B',   'В': 'V',    'Г': 'G',   'Д': 'D',    'Е': 'E',
    'Ё': 'YO', 'Ж': 'ZH',  'З': 'Z',    'И': 'I',   'Й': 'Y',    'К': 'K',
    'Л': 'L',  'М': 'M',   'Н': 'N',    'О': 'O',   'П': 'P',    'Р': 'R',
    'С': 'S',  'Т': 'T',   'У': 'U',    'Ф': 'F',   'Х': 'KH',   'Ц': 'TS',
    'Ч': 'CH', 'Ш': 'SH',  'Щ': 'SCH',  'Ъ': 'HARD','Ы': 'YI',   'Ь': 'SOFT',
    'Э': 'E',  'Ю': 'YU',  'Я': 'YA',
}

CONFIDENCE_THRESHOLDS = {'asl': 0.8, 'isl': 0.8, 'rsl': 0.45}
HOLD_THRESHOLD = 1.5  # seconds a gesture must be held before committing

# ── Per-client state ───────────────────────────────────────────────────────────
client_states: dict = {}


def _make_state() -> dict:
    return {
        'mode': 'asl',
        'prediction_buffer': deque(maxlen=10),
        'prob_buffer': deque(maxlen=8),
        'rsl_frame_buffer': deque(maxlen=30),
        'last_prediction': None,
        'last_change_time': time.time(),
        'last_committed_token': None,
        'token_released': True,
        'word_buffer': [],
        'sentence': [],
    }


def _reset_buffers(state: dict) -> None:
    state['prediction_buffer'].clear()
    state['prob_buffer'].clear()
    state['rsl_frame_buffer'].clear()
    state['last_prediction'] = None
    state['last_committed_token'] = None
    state['token_released'] = True
    state['last_change_time'] = time.time()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _extract_coords(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe landmark list to wrist-normalised 63D float32 vector."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    coords = np.array(coords, dtype=np.float32)
    wrist_x, wrist_y, wrist_z = float(coords[0]), float(coords[1]), float(coords[2])
    coords[::3]  -= wrist_x
    coords[1::3] -= wrist_y
    coords[2::3] -= wrist_z
    return coords


def _display(label: str | None, mode: str) -> str:
    if not label:
        return ''
    if mode == 'rsl':
        return RSL_DISPLAY_MAP.get(label, label)
    return label


def _build_response(state: dict, confidence: float = 0.0) -> dict:
    mode = state['mode']
    pb   = state['prediction_buffer']
    smoothed = Counter(pb).most_common(1)[0][0] if pb else None
    word  = ''.join(_display(ch, mode) for ch in state['word_buffer'])
    sent  = ' '.join(state['sentence'])
    return {
        'prediction': _display(smoothed, mode) or '',
        'word': word,
        'sentence': sent,
        'confidence': round(confidence, 2),
    }


# ── SocketIO events ────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    from flask import request
    client_states[request.sid] = _make_state()


@socketio.on('disconnect')
def on_disconnect():
    from flask import request
    client_states.pop(request.sid, None)


@socketio.on('process_frame')
def process_frame(data):
    from flask import request
    state = client_states.get(request.sid)
    if state is None:
        return

    # ── Decode base64 JPEG ─────────────────────────────────────────────────
    try:
        raw   = data['frame'].split(',', 1)[-1]          # strip data:image/jpeg;base64,
        arr   = np.frombuffer(base64.b64decode(raw), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception:
        emit('result', _build_response(state))
        return

    # ── MediaPipe ──────────────────────────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        results = hand_landmarker.detect(mp_img)
    except Exception:
        results = None

    coords = None
    if results and results.hand_landmarks:
        coords = _extract_coords(results.hand_landmarks[0])

    # ── Inference ──────────────────────────────────────────────────────────
    mode       = state['mode']
    prediction = None
    confidence = 0.0

    if mode == 'rsl':
        state['rsl_frame_buffer'].append(
            coords if coords is not None else np.zeros(63, dtype=np.float32)
        )
        if len(state['rsl_frame_buffer']) == 30:
            seq  = np.array(list(state['rsl_frame_buffer']), dtype=np.float32)
            pred = _models['rsl'].predict(seq[np.newaxis], verbose=0)[0]
            state['prob_buffer'].append(pred)
            avg  = np.mean(np.array(state['prob_buffer']), axis=0)
            idx  = int(np.argmax(avg))
            confidence = float(np.max(avg))
            if confidence > CONFIDENCE_THRESHOLDS['rsl'] and idx < len(CLASSES['rsl']):
                prediction = CLASSES['rsl'][idx]
        if coords is None:
            state['prob_buffer'].clear()
    else:
        if coords is not None:
            pred = _models[mode].predict(np.expand_dims(coords, 0), verbose=0)[0]
            state['prob_buffer'].append(pred)
            avg  = np.mean(np.array(state['prob_buffer']), axis=0)
            idx  = int(np.argmax(avg))
            confidence = float(np.max(avg))
            if confidence > CONFIDENCE_THRESHOLDS.get(mode, 0.8) and idx < len(CLASSES[mode]):
                prediction = CLASSES[mode][idx]
        else:
            state['prob_buffer'].clear()

    # ── Temporal smoothing ─────────────────────────────────────────────────
    state['prediction_buffer'].append(prediction)
    pb       = state['prediction_buffer']
    smoothed = Counter(pb).most_common(1)[0][0] if pb else None

    # ── Sentence engine ────────────────────────────────────────────────────
    now = time.time()

    if smoothed in (None, 'nothing'):
        state['token_released'] = True

    if smoothed != state['last_prediction']:
        state['last_prediction'] = smoothed
        state['last_change_time'] = now
    elif smoothed is not None and (now - state['last_change_time']) > HOLD_THRESHOLD:
        if smoothed == state['last_committed_token'] and not state['token_released']:
            pass  # same token, wait for release
        else:
            if smoothed == 'space':
                if state['word_buffer']:
                    word = ''.join(state['word_buffer'])
                    if mode == 'asl':
                        word = word.upper()
                    state['sentence'].append(word)
                    state['word_buffer'] = []
            elif smoothed == 'del':
                if state['word_buffer']:
                    state['word_buffer'].pop()
            elif smoothed not in ('nothing',):
                state['word_buffer'].append(smoothed)

            state['last_committed_token'] = smoothed
            state['token_released'] = False

    emit('result', _build_response(state, confidence))


@socketio.on('change_mode')
def change_mode(data):
    from flask import request
    state = client_states.get(request.sid)
    if state is None:
        return
    mode = data.get('mode', 'asl')
    if mode not in CLASSES:
        return
    state['mode'] = mode
    state['word_buffer'] = []
    state['sentence'] = []
    _reset_buffers(state)
    emit('result', _build_response(state))


@socketio.on('handle_action')
def handle_action(data):
    from flask import request
    state = client_states.get(request.sid)
    if state is None:
        return
    action = data.get('action', '')
    mode   = state['mode']

    if action == 'space':
        if state['word_buffer']:
            word = ''.join(state['word_buffer'])
            if mode == 'asl':
                word = word.upper()
            state['sentence'].append(word)
            state['word_buffer'] = []

    elif action == 'backspace':
        if state['word_buffer']:
            state['word_buffer'].pop()
        elif state['sentence']:
            # Pull last word back into word_buffer for editing
            state['word_buffer'] = list(state['sentence'].pop())

    elif action == 'clear':
        state['sentence'] = []
        state['word_buffer'] = []
        _reset_buffers(state)

    elif action in ('translate_te', 'translate_hi'):
        target_lang = 'te' if action == 'translate_te' else 'hi'
        full = ' '.join(state['sentence'])
        if state['word_buffer']:
            cur = ''.join(state['word_buffer'])
            if mode == 'asl':
                cur = cur.upper()
            full = (full + ' ' + cur).strip()
        if full:
            try:
                translated = deep_translator.GoogleTranslator(
                    source='auto', target=target_lang
                ).translate(full)
            except Exception as e:
                translated = f'[Translation error: {e}]'
            emit('translation', {'text': translated, 'lang': target_lang})
            return

    elif action == 'speak':
        full = ' '.join(state['sentence'])
        if state['word_buffer']:
            cur = ''.join(state['word_buffer'])
            if mode == 'asl':
                cur = cur.upper()
            full = (full + ' ' + cur).strip()
        if full:
            import subprocess, threading
            safe = full.replace("'", "''")
            ps_cmd = (
                f"Add-Type -AssemblyName System.Speech; "
                f"([System.Speech.Synthesis.SpeechSynthesizer]::new()).Speak('{safe}')"
            )
            threading.Thread(
                target=lambda: subprocess.run(
                    ['powershell', '-NoProfile', '-Command', ps_cmd],
                    capture_output=True
                ), daemon=True
            ).start()
        return

    emit('result', _build_response(state))


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("Server starting at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
