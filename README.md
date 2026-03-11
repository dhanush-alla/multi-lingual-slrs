# Sign Language Recognition Project

Multi-language sign language recognition system using neural networks, MediaPipe landmarks, and GPU optimization.

## Project Overview

This project recognizes sign language across two alphabet sets:
- **ASL** (American Sign Language)
- **ISL** (Indian Sign Language)  

## Project Structure

```
SignLanguageProject_Pro/
├── data/
│   ├── raw/              # Original datasets
│   │   ├── ASL/          # Kaggle ASL Alphabet
│   │   └── ISL/          # Kaggle ISL Alphabet
│   └── processed/        # Extracted landmarks
│
├── models/               # Saved model weights
│
├── preprocess.py         # Image -> landmarks conversion
├── train.py              # GPU-optimized training
├── main.py               # Live recognition app
├── check_npy.py          # Quick NPY inspection utility
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Preprocessing

```bash
python preprocess.py
```

### Training

```bash
python train.py
```

### Live Recognition

```bash
python main.py
```

### Implementation

<div align="center">
  <figure>
    <img width="802" height="640" alt="Universal Translator Live HUD Dashboard" src="https://github.com/user-attachments/assets/f8332e2a-29e3-437f-adc3-d16af9026e54" />
    <figcaption><b>Figure 1:</b> The MLSLRS Live HUD Dashboard.</figcaption>
  </figure>
</div>

<div align="center">
  <figure>
    <img width="1920" height="1080" alt="Web Browser Deployment with Flask and WebSockets" src="https://github.com/user-attachments/assets/8f5ea520-efb5-41e5-ab5e-bda1ffb678da" />
    <figcaption><b>Figure 2:</b> The web-migrated deployment running in the browser. Real-time video frames are streamed via Socket.IO to the Flask backend for ML inference, wrapped in the updated glassmorphic UI.</figcaption>
  </figure>
</div>



## Performance

- **Framework**: TensorFlow 2.15+
- **Input**: Hand landmarks (MediaPipe)
- **Output**: Sign prediction with confidence scores

## Configuration

Edit the following files to customize:
- `train.py` - Model architecture and training parameters
- `preprocess.py` - Dataset paths and preprocessing options
- `main.py` - Live app settings and confidence thresholds

## Notes

- Landmarks are extracted using MediaPipe Hand Landmarker
- Temporal smoothing reduces jitter in sequences
- Models saved in Keras 2026 format (.keras)

## License

Custom Project

## Author

Dhanush Alla
