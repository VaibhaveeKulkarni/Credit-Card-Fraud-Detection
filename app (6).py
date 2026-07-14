import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load fraud_model.pkl and scaler.pkl ---
# The joblib library is used to load the pre-trained machine learning model and scaler.
# 'fraud_model.pkl' contains the Random Forest classifier.
# 'scaler.pkl' contains the StandardScaler fitted on the 'Amount' feature.
try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Make sure 'fraud_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop the app if essential files are missing

# --- 8. Add a clean interface with a title and project description ---
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
This application uses a pre-trained Random Forest model to predict whether a credit card transaction is fraudulent or genuine.

Please enter the transaction details in the input fields below to get a prediction.
""")

# --- 9. Include an educational disclaimer ---
st.markdown("""
**Disclaimer:** This is an educational demonstration. The model's predictions are based on the patterns it learned from the training data.
In a real-world scenario, fraud detection involves more complex systems, continuous monitoring, and human oversight. Always use this tool responsibly.
""")

st.subheader("Enter Transaction Details")

# Define the features that the model expects, in the correct order.
# These are the same features used during model training.
feature_columns = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# --- 3. Allow the user to enter all required transaction details ---
# Create input fields for all features, arranged in columns for better readability.
input_data = {}

# Use st.columns to arrange inputs in a grid (e.g., 3 columns per row).
num_cols = 3
chunks = [feature_columns[i:i + num_cols] for i in range(0, len(feature_columns), num_cols)]

for chunk in chunks:
    cols = st.columns(num_cols)
    for i, feature in enumerate(chunk):
        with cols[i]:
            # For 'Amount', provide a positive default and range.
            # For 'V' features, they are usually centered around 0, so a default of 0 is reasonable.
            if feature == 'Amount':
                input_data[feature] = st.number_input(f"Enter {feature}", value=50.0, min_value=0.0, format="%.2f", key=feature)
            else:
                input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, format="%.6f", key=feature)

# --- 5. Predict whether the transaction is Fraud or Genuine ---
# --- 6. Display the prediction clearly.---
# --- 7. Show the prediction confidence if available.---
# --- 10. Handle errors gracefully.---
if st.button("Predict Fraud"):
    try:
        # Convert the collected input data into a pandas DataFrame.
        # This DataFrame must have the same column names and order as the training data.
        input_df = pd.DataFrame([input_data])

        # --- 4. Scale the Amount field using the saved scaler.---
        # Apply the loaded StandardScaler to the 'Amount' column of the input data.
        # It's crucial to use the same scaler that was fitted during training.
        input_df['Amount'] = scaler.transform(input_df[['Amount']])

        # Make a prediction using the loaded model.
        # .predict() returns the predicted class (0 for genuine, 1 for fraud).
        prediction = model.predict(input_df)[0]

        # Get the prediction probabilities.
        # .predict_proba() returns the probabilities of the input belonging to each class.
        # [0][1] gets the probability of the positive class (fraud).
        prediction_proba = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"### 🚨 Fraudulent Transaction Detected! (Confidence: {prediction_proba:.2%})")
            st.markdown("**Action Recommended:** This transaction has a high probability of being fraudulent. Further investigation is advised.")
        else:
            st.success(f"### ✅ Genuine Transaction. (Confidence: {(1 - prediction_proba):.2%})")
            st.markdown("**Action Recommended:** This transaction appears legitimate.")

    except Exception as e:
        # Catch any potential errors during prediction and display them gracefully.
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input values are valid numbers.")
