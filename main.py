import scipy.io
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

# NEW: fixed seeds so every run gives the same numbers
SEED = 42
tf.keras.utils.set_random_seed(SEED)

results = []                       # NEW: everything to report is collected here
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    results.append(s)

# ---------------- 1. LOAD DATA ----------------
data_elec = scipy.io.loadmat("dataset_elec.mat")
data_amb = scipy.io.loadmat("dataset_amb.mat")

idc1 = data_elec["idc1"].flatten()
idc2 = data_elec["idc2"].flatten()
vdc1 = data_elec["vdc1"].flatten()
vdc2 = data_elec["vdc2"].flatten()
irr = data_amb["irr"].flatten()
pvt = data_amb["pvt"].flatten()
f_nv = data_amb["f_nv"].flatten()

df = pd.DataFrame({"irr": irr, "pvt": pvt, "idc1": idc1, "idc2": idc2,
                   "vdc1": vdc1, "vdc2": vdc2, "f_nv": f_nv})

log("Shape:", df.shape)
log("Original class counts:\n", df["f_nv"].value_counts().sort_index())   # NEW: shows which classes have < 5000 rows

# class distribution plot
plt.figure(figsize=(8, 5))
df["f_nv"].value_counts().sort_index().plot(kind="bar")
plt.title("Solar PV Fault Class Distribution")
plt.xlabel("Fault Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------- 2. BALANCING ----------------
from sklearn.utils import resample

# NEW: sample WITHOUT replacement when the class has enough rows, so duplicates
# appear only for classes that really have fewer than 5000 rows.
parts = []
for c in range(5):
    cls_df = df[df["f_nv"] == c]
    parts.append(resample(cls_df, n_samples=5000,
                          replace=len(cls_df) < 5000, random_state=SEED))
balanced_df = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

log("Balanced shape:", balanced_df.shape)
log("Balanced class counts:\n", balanced_df["f_nv"].value_counts().sort_index())
log("Duplicate rows in balanced_df:", balanced_df.duplicated().sum())   # NEW

# ---------------- 3. SPLIT ----------------
from sklearn.model_selection import train_test_split

FEATURES = ["irr", "pvt", "idc1", "idc2", "vdc1", "vdc2"]
X = balanced_df[FEATURES]
y = balanced_df["f_nv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
log("Train / test shapes:", X_train.shape, X_test.shape)

# NEW: how many test rows also appear in train (leakage check)
leak = X_test.merge(X_train.drop_duplicates(), on=FEATURES, how="inner").shape[0]
log("Test rows that also appear in train:", leak)

X_train_raw, X_test_raw = X_train.copy(), X_test.copy()   # NEW: unscaled copies for later experiments

# ---------------- 4. SCALING ----------------
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # fit on train only
X_test = scaler.transform(X_test)
joblib.dump(scaler, "solar_pv_scaler.pkl")
log("Scaler saved")

# ---------------- 5. ANN ----------------
def build_ann(h1=32, h2=16):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(h1, activation="relu"),
        tf.keras.layers.Dense(h2, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

model = build_ann()
model.summary()

t0 = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_split=0.2, verbose=1)
ann_time = time.time() - t0

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
log("\nFINAL ANN test accuracy:", round(test_accuracy, 4), "| test loss:", round(test_loss, 4))
log("Final train acc:", round(history.history["accuracy"][-1], 4),
    "| final val acc:", round(history.history["val_accuracy"][-1], 4))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
class_names = ["Normal", "Short-Circuit", "Degradation", "Open-Circuit", "Shadowing"]
log("\nClassification report:\n", classification_report(y_test, y_pred, target_names=class_names, digits=3))
cm = confusion_matrix(y_test, y_pred)
log("Confusion matrix:\n", cm)

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Solar PV Fault Classification")
plt.xlabel("Predicted Class"); plt.ylabel("Actual Class")
plt.tight_layout(); plt.show()

model.save("solar_pv_ann.keras")
log("Model saved")

# NEW: demo examples for the Streamlit app (3 rows per class)
balanced_df.groupby("f_nv").sample(3, random_state=7)[FEATURES + ["f_nv"]].to_csv("samples.csv", index=False)

# ======================= NEW: EXPERIMENTS FOR THE REPORT =======================
# A) Classical ML comparison (same scaled split)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "SVM": SVC(),                       # delete this line if it is too slow
}
try:
    from xgboost import XGBClassifier   # pip install xgboost
    models["XGBoost"] = XGBClassifier(n_estimators=200, random_state=SEED)
except ImportError:
    log("xgboost not installed, skipping XGBoost")

rows = []
for name, m in models.items():
    t = time.time(); m.fit(X_train, y_train); dt = time.time() - t
    p = m.predict(X_test)
    rows.append([name, accuracy_score(y_test, p), f1_score(y_test, p, average="macro"), dt])
rows.append(["ANN (this work)", test_accuracy, f1_score(y_test, y_pred, average="macro"), ann_time])
comp = pd.DataFrame(rows, columns=["Model", "Accuracy", "Macro F1", "Train seconds"]).round(4)
log("\nMODEL COMPARISON:\n", comp.to_string(index=False))

# B) Feature importance (Random Forest)
imp = dict(zip(FEATURES, models["Random Forest"].feature_importances_.round(3)))
log("\nRandom Forest feature importance:", imp)

# C) Engineered features (Random Forest, before vs after)
def add_feats(d):
    d = d.copy()
    d["p1"] = d.vdc1 * d.idc1; d["p2"] = d.vdc2 * d.idc2
    d["i_ratio"] = d.idc1 / (d.idc2 + 1e-6); d["v_diff"] = d.vdc1 - d.vdc2
    return d
rf_a = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_train_raw, y_train)
rf_b = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(add_feats(X_train_raw), y_train)
log("RF accuracy before engineered features:", round(rf_a.score(X_test_raw, y_test), 4),
    "| after:", round(rf_b.score(add_feats(X_test_raw), y_test), 4))

# D) Network size experiment (ANN, 20 epochs each)
log("\nNETWORK SIZE EXPERIMENT:")
for h1, h2 in [(8, 4), (16, 8), (32, 16), (64, 32)]:
    tf.keras.utils.set_random_seed(SEED)
    m = build_ann(h1, h2)
    h = m.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
    log(f"hidden {h1},{h2} | params {m.count_params()} | val acc {h.history['val_accuracy'][-1]:.4f}"
        f" | test acc {m.evaluate(X_test, y_test, verbose=0)[1]:.4f}")

# Everything above is saved in one text file: just send me its contents
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print("\nDONE")