# Audio Recognition Project 🎵

A machine learning system for classifying audio recordings as cat or dog sounds using feature extraction and Random Forest classification.

## Overview

This project implements a complete audio classification pipeline that processes `.wav` audio files, extracts acoustic features, and trains a Random Forest classifier to distinguish between cat and dog sounds with **91.04% accuracy**.

## Features

### Audio Processing
- Audio loading using `librosa`
- Noise reduction with `noisereduce`
- Silence trimming
- Amplitude normalization

### Feature Extraction
- **58 features** per audio file:
  - 13 MFCC coefficients (mean, std, min, max)
  - Spectral centroid (mean, std, min, max)
  - Spectral rolloff (mean, std)
- Automatic removal of highly correlated features (threshold > 0.9)

### Model Training
- Random Forest Classifier
- 100 estimators
- Standard scaling preprocessing
- Reproducible results (fixed random state)

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 91.04% |
| **Cat F1-Score** | 0.93 |
| **Dog F1-Score** | 0.88 |


## Technologies

- Python
- Librosa
- Scikit-Learn
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Noisereduce
- IPython
- KaggleHub

## Repository Structure

```text
audio-recognition-project/
│
├── train/
│   ├── cat/          # Training cat sounds
│   └── dog/          # Training dog sounds
│
├── test/
│   ├── cats/         # Test cat sounds
│   └── test/         # Test dog sounds
│
└── AUDIO_RECOGNITION_PROJECT.ipynb
```

## Implementation Notes
- Feature Reduction: Automatically drops features with correlation > 0.9
- Scaler: StandardScaler applied to all features
- Reproducibility: Fixed random_state=42 in RandomForest

## Future Improvements
- Implement deep learning approaches (CNN)
- Add more audio features
- Explore data augmentation techniques
- Optimize hyperparameters through grid search

