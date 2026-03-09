# Sign Language Recognition Project

Multi-language sign language recognition system using neural networks, MediaPipe landmarks, and GPU optimization.

## 📋 Project Overview

This project recognizes sign language across two alphabets:
- **ASL** (American Sign Language)
- **ISL** (Indian Sign Language)  

## 🏗️ Project Structure

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

## 🚀 Quick Start

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

## 📊 Performance

- **Framework**: TensorFlow 2.15+
- **Input**: Hand landmarks (MediaPipe)
- **Output**: Sign prediction with confidence scores

## 🔧 Configuration

Edit the following files to customize:
- `train.py` - Model architecture and training parameters
- `preprocess.py` - Dataset paths and preprocessing options
- `main.py` - Live app settings and confidence thresholds

## 📝 Notes

- Landmarks are extracted using MediaPipe Hand Landmarker
- Temporal smoothing reduces jitter in sequences
- Models saved in Keras 2026 format (.keras)

## 📄 License

Custom Project

## 👤 Author

Sign Language Project Team
