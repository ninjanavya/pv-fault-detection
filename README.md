# ☀️ PV Fault Detection and Classification using an ANN

A machine learning approach for intelligent photovoltaic (PV) system monitoring.
The project classifies **five operating conditions** of a solar PV system from **six sensor readings** using an Artificial Neural Network, benchmarks it against classical ML models, and includes a **Streamlit demo app**.

**Author:** Navya Khandelwal (RA2411026030021) · Subjects: Artificial Neural Networks and Machine Learning

<!-- After adding docs/app_screenshot.png, remove the comment markers on the next line -->
<!-- ![Demo app](docs/app_screenshot.png) -->

## Problem
Solar PV systems develop faults (short circuit, open circuit, shading, degradation) that lower output. The raw measurements do not tell us directly which fault is occurring, so a supervised model learns the relationship from labelled data.

| Label | Condition |
|---|---|
| 0 | Normal |
| 1 | Short-Circuit |
| 2 | Degradation |
| 3 | Open-Circuit |
| 4 | Shadowing |

**Inputs (6):** `irr` (irradiance), `pvt` (PV temperature), `idc1`, `idc2` (DC currents), `vdc1`, `vdc2` (DC voltages) · **Target:** `f_nv` (fault class)

## Method
1. Merge `dataset_elec.mat` and `dataset_amb.mat` → 1,373,798 rows, 7 attributes.
2. Severe class imbalance (Normal ≈ 1.16 M rows; faults 1 to 3 only 5,999 to 10,371) → balance to **5,000 rows per class** (25,000 total, 0 duplicates).
3. Stratified 80/20 split (`random_state=42`): 16,000 train · 4,000 validation · 5,000 test.
4. `StandardScaler` **fit on the training set only** (no leakage; 0 test rows appear in train).
5. ANN: `6 → Dense 32 (ReLU) → Dense 16 (ReLU) → Dense 5 (softmax)` = **837 parameters**. Adam, sparse categorical cross-entropy, batch size 64, 20 epochs.

![Class distribution](docs/class_distribution.png)

## Results
**ANN test accuracy: 98.44 %** (4,922 / 5,000 unseen samples, macro F1 0.984).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.985 | 0.947 | 0.966 |
| Short-Circuit | 0.998 | 0.994 | 0.996 |
| Degradation | 0.994 | 0.997 | 0.996 |
| Open-Circuit | 0.999 | 1.000 | 1.000 |
| Shadowing | 0.947 | 0.984 | 0.965 |

![Confusion matrix](docs/confusion_matrix.png)

Most errors (67 of 78) are **Normal vs Shadowing**: mild partial shading looks like slightly lower irradiance.

### Comparison with classical ML
![Model comparison](docs/model_comparison.png)

| Model | Accuracy | Macro F1 | Train time |
|---|---|---|---|
| Logistic Regression | 94.22 % | 94.15 % | 0.5 s |
| Decision Tree | 99.00 % | 99.00 % | 0.2 s |
| Random Forest | 99.36 % | 99.36 % | 1.3 s |
| SVM | 97.92 % | 97.92 % | 1.4 s |
| XGBoost | 99.52 % | 99.52 % | 1.5 s |
| **ANN (this work)** | **98.44 %** | **98.44 %** | 20.8 s |

Tree ensembles are slightly more accurate on this tabular data and train much faster. The ANN is a competitive baseline with only 837 parameters, outputs class probabilities, and can be extended to deeper or sequence models.

### Network size experiment
| Hidden layers | Parameters | Val. accuracy | Test accuracy |
|---|---|---|---|
| 8, 4 | 117 | 96.45 % | 96.00 % |
| 16, 8 | 293 | 98.15 % | 97.84 % |
| **32, 16 (chosen)** | **837** | **98.60 %** | **98.44 %** |
| 64, 32 | 2,693 | 98.85 % | 98.40 % |

A larger network gave no test gain with 3× the parameters, so 32/16 was kept.

### Feature importance (Random Forest)
`vdc2` 0.300 · `idc2` 0.208 · `irr` 0.189 · `idc1` 0.157 · `vdc1` 0.077 · `pvt` 0.069. Adding engineered features (string power, current ratio, voltage difference) raised Random Forest accuracy from 99.36 % to 99.62 %.

All printed results of the final run are in [`results.txt`](results.txt). The full report is in [`docs/Project_Report.docx`](docs/Project_Report.docx).

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (use: source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt   # use Python 3.11 or 3.12 (TensorFlow does not support newer versions yet)
python main.py                    # trains, evaluates, runs experiments, writes results.txt
streamlit run app_v2.py           # demo app
```
Place `dataset_elec.mat` and `dataset_amb.mat` in the project folder (source: **[add dataset name and link]**).

## Files
| File | Purpose |
|---|---|
| `main.py` | Data loading, balancing, training, evaluation and experiments |
| `app_v2.py` | Streamlit demo (predict a fault from sensor values, see model performance) |
| `solar_pv_ann.keras`, `solar_pv_scaler.pkl` | Trained model and fitted scaler |
| `samples.csv` | Example rows for the demo |
| `results.txt` | Printed results of the final run |
| `docs/` | Figures and the project report |

## Limitations
- The test set is balanced; a real plant is mostly Normal, so false alarms would matter more in practice.
- Undersampling discards most Normal data; a random split on time-series rows may be optimistic.
- Single dataset, not tested on real plant data.

## Future work
Real-time monitoring with IoT sensors · testing on real plant data and imbalanced test sets · class weights or SMOTE · LSTM / 1D-CNN with time-based splits · early fault warning · explainable AI (SHAP) · edge deployment with TensorFlow Lite.
