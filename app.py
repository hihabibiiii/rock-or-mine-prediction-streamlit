import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("rock_mine_model.pkl", "rb"))

st.title("🪨 Rock vs Mine Prediction App")

st.write("Enter SONAR signal values (60 features):")

input_data = []

for i in range(60):
    value = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f")
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == "R":
        st.success("🪨 This is a ROCK")
    else:
        st.error("💣 This is a MINE")
