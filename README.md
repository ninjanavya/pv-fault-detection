# ☀️ Solar PV Fault Detection & Classification using ANN

> **Author:** Navya Khandelwal (RA2411026030021)  
> **Subject:** Artificial Neural Networks and Machine Learning  
> **Demo App:** `streamlit run app.py`

---

## 🤖 Machine Learning Approach

> ### ✅ This project uses **Supervised Machine Learning**
>
> | Property | Detail |
> |----------|--------|
> | **ML Type** | Supervised Learning |
> | **Task** | Multi-class Classification (5 classes) |
> | **Primary Model** | Artificial Neural Network (ANN) |
> | **Also Benchmarked** | Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost |
> | **Labels source** | Ground-truth fault labels (`f_nv`) from the dataset |
> | **Why Supervised?** | We have labelled training data — the model learns the mapping from sensor readings → fault class by seeing thousands of labelled examples |

**Supervised learning** means the model is trained on **input–output pairs**:
- **Input (X):** 6 sensor readings (`irr`, `pvt`, `idc1`, `idc2`, `vdc1`, `vdc2`)
- **Output (y):** Known fault label (0 = Normal, 1 = Short-Circuit, … 4 = Shadowing)

The model learns this mapping during training, then **predicts the fault class for new, unseen sensor readings**.

---

## 📌 Overview

Solar PV systems develop faults — short circuits, open circuits, shading and degradation — that silently reduce power output. This project trains an **Artificial Neural Network (ANN)** to classify **5 operating conditions** from **6 real sensor readings**, benchmarks it against classical ML models, and ships a **live Streamlit web demo**.

---

## 🎯 Fault Classes

| Label | Condition | Description |
|-------|-----------|-------------|
| 0 | **Normal** | Panel operating correctly |
| 1 | **Short-Circuit** | Electrical short in a string or module |
| 2 | **Degradation** | Panel efficiency gradually declining |
| 3 | **Open-Circuit** | Broken or disconnected string |
| 4 | **Shadowing** | Partial or full shading on panels |

---

## 📥 Inputs (6 Sensor Features)

| Feature | Description |
|---------|-------------|
| `irr` | Solar Irradiance (W/m²) |
| `pvt` | PV Panel Temperature (°C) |
| `idc1` | DC Current — String 1 (A) |
| `idc2` | DC Current — String 2 (A) |
| `vdc1` | DC Voltage — String 1 (V) |
| `vdc2` | DC Voltage — String 2 (V) |

---

## 🔬 Methodology

```
1,373,798 raw rows  (dataset_elec.mat + dataset_amb.mat)
        │
        ▼
  Balance to 5,000 rows per class  →  25,000 total
        │
        ▼
  Stratified 80/20 split  →  20,000 train / 5,000 test
        │
        ▼
  StandardScaler  (fit on TRAIN only — no data leakage)
        │
        ▼
  ANN: 6 → Dense(32, ReLU) → Dense(16, ReLU) → Dense(5, Softmax)
        │
        ▼
  Fault Class Prediction
```

**ANN Hyperparameters:** Adam optimizer · Sparse Categorical Cross-Entropy · Batch 64 · 20 Epochs · 837 total parameters

---

## 📊 Results

### ✅ Test Accuracy: **98.40%** (4,920 / 5,000 correct)

### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 98.2% | 94.9% | 96.5% |
| Short-Circuit | 100.0% | 99.5% | 99.8% |
| Degradation | 99.4% | 99.4% | 99.4% |
| Open-Circuit | 99.7% | 100.0% | 99.9% |
| Shadowing | 94.8% | 98.2% | 96.5% |

> 68 of 80 total errors are **Normal ↔ Shadowing** — mild shading looks like low irradiance.

### Confusion Matrix

| | Normal | Short-Circuit | Degradation | Open-Circuit | Shadowing |
|---|---|---|---|---|---|
| **Normal** | 949 | 0 | 0 | 0 | 51 |
| **Short-Circuit** | 0 | 995 | 5 | 0 | 0 |
| **Degradation** | 0 | 0 | 994 | 3 | 3 |
| **Open-Circuit** | 0 | 0 | 0 | 1000 | 0 |
| **Shadowing** | 17 | 0 | 1 | 0 | 982 |

---

## 🏆 Model Comparison

| Model | Accuracy | Macro F1 | Train Time |
|-------|----------|----------|------------|
| Logistic Regression | 94.22% | 94.15% | ~0.5 s |
| Decision Tree | 99.00% | 99.00% | ~0.2 s |
| Random Forest | 99.36% | 99.36% | ~1.3 s |
| SVM | 97.92% | 97.92% | ~1.4 s |
| XGBoost | 99.52% | 99.52% | ~1.5 s |
| **ANN (this work)** | **98.44%** | **98.44%** | ~21 s |

The ANN is competitive with only **837 parameters**, produces class probabilities, and is easily extendable.

### Network Size Experiment

| Hidden Layers | Parameters | Val Accuracy | Test Accuracy |
|---------------|------------|--------------|---------------|
| 8, 4 | 117 | 96.45% | 96.00% |
| 16, 8 | 293 | 98.15% | 97.84% |
| **32, 16 ✅** | **837** | **98.60%** | **98.44%** |
| 64, 32 | 2,693 | 98.85% | 98.40% |

---

## 🚀 How to Run

### Requirements
- Python 3.11 or 3.12 (TensorFlow does not support 3.13+ yet)
- Dataset files: `dataset_elec.mat` and `dataset_amb.mat` (place in project root)

### Setup & Run

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates solar_pv_ann.keras, solar_pv_scaler.pkl, samples.csv)
python main.py

# 4. Launch the demo app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 File Structure

```
solar/
├── app.py                  # Streamlit demo app (3-tab UI)
├── app_v2.py               # Streamlit demo v2
├── main.py                 # Training, evaluation & experiments
├── predict.py              # Quick single-prediction script
├── solar_pv_ann.keras      # Trained ANN model
├── solar_pv_scaler.pkl     # Fitted StandardScaler
├── samples.csv             # 15 example rows (3 per class)
├── requirements.txt        # All Python dependencies
├── results.txt             # Full output of the last training run
├── run_app.bat             # One-click launcher (Windows)
└── README.md               # This file
```

> **Note:** `dataset_elec.mat` and `dataset_amb.mat` are excluded from this repo (each ~10–18 MB). Add them to the project root before running `main.py`.

---

## 🌐 Demo App Features

- **Predict tab** — Load a sample by class or enter your own sensor values → get a colour-coded fault prediction with confidence %
- **Model Performance tab** — Test accuracy, per-class metrics, and Blues-gradient confusion matrix
- **How It Works tab** — Full pipeline overview and network architecture

---

## ⚠️ Limitations

- Test set is balanced; a real plant is predominantly Normal
- Single dataset — not validated on real plant data
- Time-series structure not used (i.i.d. assumption)

## 🔭 Future Work

- Real-time IoT sensor monitoring
- LSTM / 1D-CNN with time-based splits
- Class weights or SMOTE for imbalanced evaluation
- SHAP explainability
- TensorFlow Lite deployment on edge devices
