import streamlit as st
import joblib
import json

st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using a model"
         " built on the Palmer Penguins dataset. Use the form below to get started")

# Load the model, unique_penguin_mapping, and feature order
rfc = joblib.load('random_forest_penguin.joblib')

with open('output_penguin.json', 'r') as f:
    unique_penguin_mapping = json.load(f)

with open('feature_order.json', 'r') as f:
    feature_order = json.load(f)

# User inputs
island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgersen'])
sex = st.selectbox('Sex', options=["Female", "Male"])
bill_length = st.number_input('Bill Length (mm)', min_value=0)
bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
body_mass = st.number_input('Body Mass (g)', min_value=0)

# Create a dictionary to hold the input data
input_data = {feature: 0 for feature in feature_order}
input_data.update({
    f'island_{island}': 1,
    f'sex_{sex}': 1,
    'bill_length_mm': bill_length,
    'bill_depth_mm': bill_depth,
    'flipper_length_mm': flipper_length,
    'body_mass_g': body_mass
})

# Arrange the input data in the order expected by the model
ordered_input_data = [input_data[feature] for feature in feature_order]

# Make the prediction
new_prediction = rfc.predict([ordered_input_data])
prediction_species = unique_penguin_mapping[new_prediction[0]]

st.write("We predict your penguin is of the {} species".format(prediction_species))
