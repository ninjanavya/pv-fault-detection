import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── env vars before TensorFlow import ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("KERAS_HOME",   os.path.join(BASE_DIR, ".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".cache", "matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

# ── constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Normal", "Short-Circuit", "Degradation", "Open-Circuit", "Shadowing"]

FEATURES = ["irr", "pvt", "idc1", "idc2", "vdc1", "vdc2"]
FEATURE_LABELS = {
    "irr":  "irr  – Irradiance (W/m²)",
    "pvt":  "pvt  – PV Temperature (°C)",
    "idc1": "idc1 – DC Current 1 (A)",
    "idc2": "idc2 – DC Current 2 (A)",
    "vdc1": "vdc1 – DC Voltage 1 (V)",
    "vdc2": "vdc2 – DC Voltage 2 (V)",
}

DEFAULT_VALUES = {"irr": 955.724, "pvt": 46.090,
                  "idc1": 9.130,  "idc2": 8.798,
                  "vdc1": 259.089, "vdc2": 201.761}

# Colour hex, icon, description, action for each class
CLASS_META = {
    "Normal":        ("#1a7a4a", "✅",
                      "The panel is operating within expected parameters.",
                      "No action needed — continue routine monitoring."),
    "Short-Circuit": ("#c0392b", "⚡",
                      "A short circuit has been detected across a string or module.",
                      "Isolate the affected string immediately and inspect wiring."),
    "Degradation":   ("#d35400", "📉",
                      "Panel efficiency has gradually declined below normal levels.",
                      "Schedule an inspection; consider cleaning or replacement."),
    "Open-Circuit":  ("#6c3483", "🔌",
                      "An open-circuit break is preventing current from flowing.",
                      "Check connectors, fuses, and junction-box terminations."),
    "Shadowing":     ("#b7950b", "🌥️",
                      "Partial or full shading is reducing power output.",
                      "Remove obstructions or add bypass diodes if recurring."),
}

# ── hardcoded performance data ────────────────────────────────────────────────
PERF = {
    "Normal":        (98.2, 94.9, 96.5),
    "Short-Circuit": (100.0, 99.5, 99.8),
    "Degradation":   (99.4, 99.4, 99.4),
    "Open-Circuit":  (99.7, 100.0, 99.9),
    "Shadowing":     (94.8, 98.2, 96.5),
}
CM = np.array([
    [949,   0,  0,    0,  51],
    [  0, 995,  5,    0,   0],
    [  0,   0, 994,   3,   3],
    [  0,   0,  0, 1000,   0],
    [ 17,   0,  1,    0, 982],
])

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar PV Fault Detection",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar PV Fault Detection")
st.caption("ANN classifier: 6 sensor inputs → 32 → 16 → 5 conditions")

# ── load model & scaler (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = tf.keras.models.load_model("solar_pv_ann.keras")
    scaler = joblib.load("solar_pv_scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# ── load samples.csv if present ───────────────────────────────────────────────
SAMPLES_PATH = os.path.join(BASE_DIR, "samples.csv")
samples_df = pd.read_csv(SAMPLES_PATH) if os.path.exists(SAMPLES_PATH) else None

# ── session-state defaults ────────────────────────────────────────────────────
for f in FEATURES:
    st.session_state.setdefault(f, float(DEFAULT_VALUES[f]))
st.session_state.setdefault("loaded_class", None)
st.session_state.setdefault("result", None)   # stores last prediction dict

# ══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_perf, tab_how = st.tabs(
    ["🔍 Predict", "📊 Model Performance", "ℹ️ How It Works"]
)

#                   PREDICT TAB                        
#
with tab_predict:
    col_in, col_out = st.columns([1, 1], gap="large")

    # LEFT: inputs 
    with col_in:
        # Example loader (only when samples.csv exists)
        if samples_df is not None:
            st.subheader("Load Example")
            ex_col1, ex_col2 = st.columns([2, 1])
            sel_class = ex_col1.selectbox("Class", CLASS_NAMES, key="sel_class")
            class_idx = CLASS_NAMES.index(sel_class)
            rows = samples_df[samples_df["f_nv"] == class_idx]
            row_labels = [f"Row {i+1}" for i in range(len(rows))]
            sel_row = ex_col2.selectbox("Row", row_labels, key="sel_row")
            row_idx  = row_labels.index(sel_row)

            if st.button("📂 Load example", use_container_width=True):
                row = rows.iloc[row_idx]
                for f in FEATURES:
                    st.session_state[f] = float(row[f])
                st.session_state["loaded_class"] = sel_class
                st.session_state["result"] = None   # clear stale result
                st.rerun()

            st.divider()

        # Six sensor inputs
        st.subheader("Sensor Readings")
        r1c1, r1c2 = st.columns(2)
        irr  = r1c1.number_input(FEATURE_LABELS["irr"],  key="irr",  format="%.4f")
        pvt  = r1c2.number_input(FEATURE_LABELS["pvt"],  key="pvt",  format="%.4f")

        r2c1, r2c2 = st.columns(2)
        idc1 = r2c1.number_input(FEATURE_LABELS["idc1"], key="idc1", format="%.4f")
        idc2 = r2c2.number_input(FEATURE_LABELS["idc2"], key="idc2", format="%.4f")

        r3c1, r3c2 = st.columns(2)
        vdc1 = r3c1.number_input(FEATURE_LABELS["vdc1"], key="vdc1", format="%.4f")
        vdc2 = r3c2.number_input(FEATURE_LABELS["vdc2"], key="vdc2", format="%.4f")

        st.write("")
        pressed = st.button("⚡ Predict Fault", type="primary", use_container_width=True)

        if pressed:
            inp = pd.DataFrame([[irr, pvt, idc1, idc2, vdc1, vdc2]], columns=FEATURES)
            x   = scaler.transform(inp)
            probs = model.predict(x, verbose=0)[0]
            idx   = int(np.argmax(probs))
            st.session_state["result"] = {
                "probs":      probs,
                "pred_class": CLASS_NAMES[idx],
                "confidence": float(probs[idx]) * 100,
            }
            st.rerun()

    # RIGHT: output 
    with col_out:
        res = st.session_state["result"]

        if res is None:
            # Prompt before first prediction
            st.info(
                "👈  **Fill in sensor readings** (or load an example) "
                "and press **Predict Fault** to classify the panel condition."
            )
        else:
            pred  = res["pred_class"]
            conf  = res["confidence"]
            probs = res["probs"]
            color, icon, meaning, action = CLASS_META[pred]

            # ── large colour-coded result card ────────────────────────────────
            st.markdown(f"""
<div style="
    background:{color}22;
    border-left: 6px solid {color};
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
">
  <div style="font-size:2.4rem; font-weight:700; color:{color};">
      {icon} {pred}
  </div>
  <div style="font-size:1.15rem; color:#555; margin-top:4px;">
      Confidence: <strong>{conf:.2f}%</strong>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── what it means / suggested action ─────────────────────────────
            st.markdown(f"**What it means:** {meaning}")
            st.markdown(f"**Suggested action:** {action}")
            st.write("")

            # ── probability bar chart ─────────────────────────────────────────
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({"Probability": probs}, index=CLASS_NAMES)
            st.bar_chart(prob_df, color="#4a90d9")

            # ── match indicator when example was loaded ───────────────────────
            if st.session_state["loaded_class"]:
                true_cls = st.session_state["loaded_class"]
                if pred == true_cls:
                    st.success(f"✅ Prediction **matches** the true class: **{true_cls}**")
                else:
                    st.error(
                        f"❌ Prediction **does not match** the true class: "
                        f"**{true_cls}** (got **{pred}**)"
                    )

# ╔══════════════════════════════════════════════════════╗
# ║               MODEL PERFORMANCE TAB                  ║
# ╚══════════════════════════════════════════════════════╝
with tab_perf:
    # ── headline metric ───────────────────────────────────────────────────────
    st.subheader("Test Accuracy")
    st.metric("Overall Test Accuracy", "98.40%", "4,920 of 5,000 correct")
    st.write("")

    # ── per-class table ───────────────────────────────────────────────────────
    st.subheader("Per-Class Metrics (test set)")
    perf_df = pd.DataFrame(
        [(cls, p, r, f1) for cls, (p, r, f1) in PERF.items()],
        columns=["Class", "Precision (%)", "Recall (%)", "F1 (%)"],
    ).set_index("Class")
    st.dataframe(perf_df.style.format("{:.1f}"), width="stretch")
    st.write("")

    # ── confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(CM, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.index.name   = "Actual \\ Predicted"

    # Blues gradient: higher count -> darker blue
    max_val = CM.max()
    def blues_bg(val):
        intensity = int(220 - (val / max_val) * 180)
        return f"background-color: rgb({intensity},{intensity},255); color: {'white' if intensity < 130 else 'black'};"

    styled_cm = cm_df.style.map(blues_bg)   # applymap → map in pandas 2.x
    st.dataframe(styled_cm, width="stretch")
    st.caption(
        "**Note:** 68 of the 80 total errors are between Normal and Shadowing — "
        "both conditions can present with low irradiance readings."
    )

# ╔══════════════════════════════════════════════════════╗
# ║                HOW IT WORKS TAB                      ║
# ╚══════════════════════════════════════════════════════╝
with tab_how:
    st.subheader("Pipeline Overview")
    st.markdown("""
| Step | Detail |
|------|--------|
| **Raw data** | 1,373,798 rows from two MATLAB sensor files (`dataset_elec.mat` + `dataset_amb.mat`) |
| **Balancing** | Random oversampling/undersampling to **5,000 rows per class** (25,000 total) |
| **Split** | Stratified 80 / 20 train-test split → 20,000 train / 5,000 test |
| **Scaling** | `StandardScaler` fit on **training set only**, applied to both sets |
| **Model** | ANN with **837 trainable parameters** |
| **Output** | One of 5 fault classes via softmax |
""")

    st.subheader("Network Architecture")
    st.markdown("""
```
Input  (6 features)
  │
Dense(32, activation='relu')
  │
Dense(16, activation='relu')
  │
Dense(5,  activation='softmax')   ← 5 fault classes
```

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Cross-Entropy |
| Batch size | 64 |
| Epochs | 20 |
| Total parameters | 837 |
""")

    st.subheader("Fault Classes")
    for cls, (color, icon, meaning, action) in CLASS_META.items():
        st.markdown(
            f"**{icon} {cls}** — {meaning} *Action: {action}*"
        )
