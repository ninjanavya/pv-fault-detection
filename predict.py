import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved model and scaler
model = load_model("solar_pv_ann.keras")
scaler = joblib.load("solar_pv_scaler.pkl")

# Class names
class_names = [
    "Normal",
    "Short-Circuit",
    "Degradation",
    "Open-Circuit",
    "Shadowing"
]

# Example solar PV sensor values
sample = np.array([[
    955.724,   # irradiance
    46.0895,   # PV temperature
    9.1298,    # DC current 1
    8.7981,    # DC current 2
    259.089,   # DC voltage 1
    201.761    # DC voltage 2
]])

# Scale the input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

predicted_class = np.argmax(prediction, axis=1)[0]
confidence = np.max(prediction) * 100

print("\nSolar PV Fault Prediction")
print("-------------------------")
print("Predicted Class:", class_names[predicted_class])
print("Confidence:", round(confidence, 2), "%")
