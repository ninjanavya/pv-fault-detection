# Solar PV Fault Detection - demo app (v2)
# Run:  streamlit run app_v2.py
# Same folder needs: solar_pv_ann.keras, solar_pv_scaler.pkl, samples.csv (optional)
import os
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

CLASSES = ["Normal", "Short-Circuit", "Degradation", "Open-Circuit", "Shadowing"]
FEATURES = ["irr", "pvt", "idc1", "idc2", "vdc1", "vdc2"]
LABELS = {"irr": "Irradiance", "pvt": "PV temperature", "idc1": "DC current 1",
          "idc2": "DC current 2", "vdc1": "DC voltage 1", "vdc2": "DC voltage 2"}
INFO = {  # colour, icon, meaning, suggested action
    "Normal": ("#1b8a3a", "✅", "The array is operating as expected.", "No action needed."),
    "Short-Circuit": ("#c62828", "⚡", "Voltage collapses while current rises abnormally.",
                      "Isolate the string and inspect wiring and bypass diodes."),
    "Degradation": ("#e65100", "📉", "Gradual power loss, for example from ageing.",
                    "Schedule an inspection and compare with earlier output."),
    "Open-Circuit": ("#6a1b9a", "🔌", "Current drops to near zero while voltage stays high.",
                     "Check connectors, fuses and cable breaks."),
    "Shadowing": ("#b8860b", "🌥️", "Partial shading cuts the current of shaded cells.",
                  "Remove the obstruction or clean the panels."),
}

st.set_page_config(page_title="Solar PV Fault Detection", page_icon="☀️", layout="wide")

@st.cache_resource
def load():
    return (tf.keras.models.load_model("solar_pv_ann.keras"),
            joblib.load("solar_pv_scaler.pkl"))

model, scaler = load()
samples = pd.read_csv("samples.csv") if os.path.exists("samples.csv") else None

st.title("☀️ Solar PV Fault Detection")
st.caption("ANN classifier: 6 sensor inputs → 32 → 16 → 5 operating conditions")
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model performance", "ℹ️ How it works"])

with tab1:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Sensor readings")
        if samples is not None:
            a, b, c = st.columns([2, 1, 1])
            cls = a.selectbox("Example class", CLASSES)
            k = b.selectbox("Row", [1, 2, 3])
            c.write("")
            if c.button("Load"):
                row = samples[samples["f_nv"] == CLASSES.index(cls)].iloc[k - 1]
                for f in FEATURES:
                    st.session_state[f] = float(row[f])
                st.session_state["truth"] = cls
        cols = st.columns(2)
        vals = [cols[i % 2].number_input(LABELS[f], key=f, format="%.4f",
                value=float(samples[f].median()) if samples is not None else 0.0)
                for i, f in enumerate(FEATURES)]
        go = st.button("Predict", type="primary", use_container_width=True)
    with right:
        st.subheader("Result")
        if go:
            x = scaler.transform(pd.DataFrame([vals], columns=FEATURES))
            p = model.predict(x, verbose=0)[0]
            pred = CLASSES[int(p.argmax())]
            col, icon, meaning, action = INFO[pred]
            st.markdown(
                f"<div style='background:{col};color:white;padding:22px;border-radius:16px'>"
                f"<div style='font-size:40px;font-weight:700'>{icon} {pred}</div>"
                f"<div style='font-size:20px'>Confidence: {p.max()*100:.1f}%</div></div>",
                unsafe_allow_html=True)
            st.write(f"**What it means:** {meaning}")
            st.write(f"**Suggested action:** {action}")
            if "truth" in st.session_state:
                ok = st.session_state["truth"] == pred
                st.write(("✅ Matches" if ok else "❌ Differs from")
                         + f" the example's true class: **{st.session_state['truth']}**")
            st.bar_chart(pd.DataFrame({"Probability": p}, index=CLASSES))
        else:
            st.info("Load an example or type values, then press Predict.")

with tab2:
    st.subheader("Test-set results (5,000 unseen samples, 1,000 per class)")
    st.metric("Test accuracy", "98.40%", "4,920 of 5,000 correct")
    st.dataframe(pd.DataFrame({
        "Class": CLASSES, "Precision %": [98.2, 100, 99.4, 99.7, 94.8],
        "Recall %": [94.9, 99.5, 99.4, 100, 98.2], "F1 %": [96.5, 99.8, 99.4, 99.9, 96.5]}),
        hide_index=True, use_container_width=True)
    cm = pd.DataFrame([[949, 0, 0, 0, 51], [0, 995, 5, 0, 0], [0, 0, 994, 3, 3],
                       [0, 0, 0, 1000, 0], [17, 0, 1, 0, 982]], index=CLASSES, columns=CLASSES)
    st.write("Confusion matrix (rows: actual, columns: predicted)")
    try:
        st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)
    except Exception:
        st.dataframe(cm, use_container_width=True)
    st.caption("Most errors (68 of 80) are Normal vs Shadowing: mild shading looks like lower irradiance. "
               "Update these numbers from your final run.")

with tab3:
    st.markdown("""
**Pipeline:** raw data (1,373,798 rows) → 5,000 rows per class (25,000) → stratified 80/20 split
→ StandardScaler (fit on train only) → ANN (837 parameters) → fault class.

**Network:** 6 inputs → Dense 32 (ReLU) → Dense 16 (ReLU) → Dense 5 (softmax).
Trained with Adam, sparse categorical cross-entropy, batch size 64, 20 epochs.

**Note:** demo examples come from the balanced dataset, so some may have been seen in training.
The test accuracy above is the reliable number.
""")
