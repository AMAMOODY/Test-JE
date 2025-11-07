import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np

# --- Configuration ---
MODEL_FILE = 'ebm_flexible_model.pkl'
MAPPINGS_FILE = 'feature_mappings.json'

# --- Load Model and Mappings ---
@st.cache_resource
def load_model():
    print("[INFO] Loading model...")
    model = joblib.load(MODEL_FILE)
    print("[INFO] Model loaded.")
    return model

@st.cache_data
def load_mappings():
    print("[INFO] Loading mappings...")
    with open(MAPPINGS_FILE, 'r') as f:
        mappings = json.load(f)
    print("[INFO] Mappings loaded.")
    return mappings

try:
    model = load_model()
    mappings = load_mappings()
    MODEL_FEATURES = model.feature_names_in_
    NOMINAL_FEATURES = list(mappings.keys())
except FileNotFoundError:
    st.error(f"Error: Model or mapping file not found. Make sure {MODEL_FILE} and {MAPPINGS_FILE} are in the repository.")
    st.stop()

# --- Build The User Interface ---
st.set_page_config(layout="centered")
st.title('Grade Predictor')

# Create a dictionary to hold user inputs
inputs = {}

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

# Iterate over all features the model needs
for i, feature in enumerate(MODEL_FEATURES):
    # Decide which column to put the input in
    target_col = col1 if i % 2 == 0 else col2
    
    if feature in NOMINAL_FEATURES:
        # This is a text/nominal feature
        options = list(mappings[feature].keys())
        inputs[feature] = target_col.selectbox(f"Select {feature}", options=options)
    else:
        # This is a numeric/continuous feature
        # --- [FIX] ---
        # Add step=1 and format="%d" to force integer input
        inputs[feature] = target_col.number_input(
            f"Enter {feature}", 
            value=None, 
            step=1, 
            format="%d"
        )
        # --- [END FIX] ---

# --- Prediction Logic ---
if st.button('Predict Grade', use_container_width=True, type="primary"):
    try:
        processed_data = {}
        for feature in MODEL_FEATURES:
            value = inputs.get(feature)
            
            # Check for empty numeric fields
            if value is None and feature not in NOMINAL_FEATURES:
                st.error(f"Error: Please provide a value for {feature}.")
                st.stop() # Stop execution

            if feature in NOMINAL_FEATURES:
                # Convert text ("Band A") to its ID (0)
                mapped_value = mappings[feature].get(str(value), -1)
                processed_data[feature] = [mapped_value]
            else:
                # Convert number to int
                processed_data[feature] = [int(value)]

        # Create the DataFrame for the model
        input_df = pd.DataFrame.from_dict(processed_data)
        input_df = input_df[MODEL_FEATURES]

        # Run prediction
        prediction = model.predict(input_df)
        predicted_grade = int(prediction[0])

        # Display the result
        st.success(f"## Predicted Grade: {predicted_grade}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
