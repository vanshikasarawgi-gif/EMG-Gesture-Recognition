## EMG-Based Hand Gesture Recognition using Machine Learning

### 📌 Overview
This project implements an end-to-end Electromyography (EMG) signal processing and classification pipeline to recognize hand gestures using machine learning.
Surface EMG signals are processed, meaningful time-domain features are extracted, and classifiers are trained to predict different hand gestures.

The project demonstrates applications relevant to prosthetics, rehabilitation engineering, and human–computer interaction.

### 📂 Dataset

Dataset: GRABMyo (PhysioNet)

Signals: Surface EMG

Channels: 32 EMG electrodes

Gestures: 17 hand gestures

Trials: 7 trials per gesture

Sampling: High-resolution EMG recordings

Each recording corresponds to one gesture trial.

### 🛠️ Methodology
#### 1. Data Loading

EMG data loaded using the WFDB library

.dat and .hea files read as physiological signal records

#### 2. Signal Windowing

Raw EMG signals divided into fixed-length windows

Window size: 200 samples

Each window represents a short segment of muscle activity

#### 3. Feature Extraction

For each window and each channel, the following time-domain features are extracted:

Root Mean Square (RMS)

Mean Absolute Value (MAV)

Waveform Length

Zero Crossings

📐 Total features per window:
32 channels × 4 features = 128 features

#### 4. Labeling

Each EMG window is labeled based on the gesture it belongs to

Gesture names are mapped to numerical class labels

#### 5. Data Preprocessing

Train–test split applied

Feature scaling performed using StandardScaler

Scaling parameters learned only from training data to avoid data leakage

#### 6. Model Training

Two machine learning models were trained:

Support Vector Machine (SVM)

Random Forest Classifier

#### 7. Model Evaluation

Accuracy score

Confusion matrix

Cross-validation for robustness

📊 Best Test Accuracy: ~94%

#### 8. Visualization

Raw EMG signal visualization

Feature distribution analysis

PCA (Principal Component Analysis) for 2D visualization of high-dimensional EMG features

### 📈 Results

High classification accuracy across multiple gestures

SVM showed slightly better generalization than Random Forest

Some overlap observed between similar gestures (expected in EMG data)

### 🚀 Future Improvements

Deep learning models (CNN / LSTM)

Subject-independent testing

Real-time gesture recognition

Feature selection and optimization

### 🧪 Technologies Used

Python

NumPy, Matplotlib

WFDB

Scikit-learn

Jupyter Notebook

### 🎯 Relevance

This project is relevant to:

Biomedical Engineering

Biosignal Processing

Machine Learning in Healthcare

Prosthetics & Assistive Technology

### 📎 Notes

This project uses offline EMG data only

No hardware or electrodes are required to run the code
