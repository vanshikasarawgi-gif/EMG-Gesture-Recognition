## EMG-Based Hand Gesture Recognition using Machine Learning

### 📌 Overview

This project presents an **end-to-end EMG (Electromyography) gesture recognition pipeline** using classical machine learning. Surface EMG signals are processed, time-domain features are extracted, and multiple classifiers are trained and compared to recognize hand gestures.

The work is relevant to **prosthetics, rehabilitation engineering, and human–computer interaction**.

---

### 📂 Dataset

* **Dataset:** GRABMyo (PhysioNet)
* **Subset Used:** Session 1, Participant 1
* **Signals:** Surface EMG
* **Channels:** 32 EMG electrodes
* **Gestures:** 17 hand gestures
* **Trials:** 7 trials per gesture

> A focused subset of the dataset is used to enable controlled experimentation and fair model comparison.

---

### 🛠️ Methodology

#### 1. Data Loading

* EMG data loaded using the **WFDB** library
* `.dat` and `.hea` files read as physiological signal records

#### 2. Signal Windowing

* Raw EMG signals divided into fixed-length windows
* **Window size:** 200 samples
* Each window represents a short segment of muscle activity

#### 3. Feature Extraction

For each window and each channel, the following **time-domain features** are extracted:

* Root Mean Square (RMS)
* Mean Absolute Value (MAV)
* Waveform Length
* Zero Crossings

**Total features per window:**
32 channels × 4 features = **128 features**

#### 4. Labeling

* Each EMG window is labeled based on the gesture performed
* Gesture names mapped to numerical class labels

#### 5. Data Preprocessing

* Train–test split applied
* Feature scaling using **StandardScaler**
* Scaling parameters learned only from training data to avoid data leakage

#### 6. Model Training

The following machine learning models were implemented and compared:

* Logistic Regression (baseline)
* k-Nearest Neighbors (kNN)
* Support Vector Machine (RBF kernel)
* Random Forest Classifier

#### 7. Model Evaluation

* Test accuracy
* Confusion matrix and classification report (Logistic Regression)
* 5-fold cross-validation for robustness

#### 8. Dimensionality Reduction & Visualization

* **PCA (Principal Component Analysis)** used for 2D visualization
* PCA applied only for visualization, not for classification

---

### 📈 Results

| Model                  | Accuracy |
| ---------------------- | -------- |
| Logistic Regression    | 94.73%   |
| k-Nearest Neighbors    | 77.92%   |
| Support Vector Machine | 94.15%   |
| Random Forest          | 91.35%   |

* Logistic Regression and SVM showed the best performance
* kNN struggled due to high-dimensional feature space
* Random Forest performed competitively but slightly below linear models

---

### 🔁 Cross-Validation (Logistic Regression)

* **Mean Accuracy:** ~93.5%
* **Low standard deviation**, indicating stable and reliable performance

---

### 📉 PCA Analysis

* PCA reduced the feature space to **2 principal components**
* **Total variance explained:** ~60.7%
* Visualization shows partial overlap between similar gestures, which is expected in EMG data

---

### 🚀 Future Improvements

* Deep learning models (CNN / LSTM) on raw EMG signals
* Training on multiple subjects and sessions
* Subject-independent evaluation
* Feature selection and hyperparameter optimization
* Real-time gesture recognition

---

### 🧪 Technologies Used

* Python
* NumPy, Pandas
* WFDB
* Scikit-learn
* Matplotlib, Seaborn
* Jupyter Notebook

---

### 🎯 Relevance

This project is relevant to:

* Biomedical Engineering
* Biosignal Processing
* Machine Learning in Healthcare
* Prosthetics & Assistive Technology

---

### 📎 Notes

* This project uses **offline EMG data only**
* No hardware or electrodes are required to run the code
