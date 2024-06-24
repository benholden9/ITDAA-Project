import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the best model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the features
with open('features.pkl', 'rb') as feature_file:
    features = pickle.load(feature_file)

# Function to get user input
def get_user_input():
    user_data = {}
    user_data['age'] = st.sidebar.number_input('age', 0, 120, 25)
    user_data['sex'] = st.sidebar.selectbox('sex', [0, 1])
    user_data['cp'] = st.sidebar.selectbox('cp', [0, 1, 2, 3])
    user_data['trestbps'] = st.sidebar.number_input('trestbps', 0, 200, 120)
    user_data['chol'] = st.sidebar.number_input('chol', 0, 600, 200)
    user_data['fbs'] = st.sidebar.selectbox('fbs', [0, 1])
    user_data['restecg'] = st.sidebar.selectbox('restecg', [0, 1, 2])
    user_data['thalach'] = st.sidebar.number_input('thalach', 0, 220, 150)
    user_data['exang'] = st.sidebar.selectbox('exang', [0, 1])
    user_data['oldpeak'] = st.sidebar.number_input('oldpeak', 0.0, 10.0, 1.0)
    user_data['slope'] = st.sidebar.selectbox('slope', [0, 1, 2])
    user_data['ca'] = st.sidebar.selectbox('ca', [0, 1, 2, 3, 4])
    user_data['thal'] = st.sidebar.selectbox('thal', [0, 1, 2, 3])

    features_df = pd.DataFrame(user_data, index=[0])
    return features_df

# Get user input
user_input = get_user_input()

# One-hot encode the categorical features
user_input_encoded = pd.get_dummies(user_input)

# Align the input features with the training features
user_input_aligned = user_input_encoded.reindex(columns=features, fill_value=0)

# Set the title
st.title("Heart Disease Prediction")

# Display the user input
st.subheader('User Input:')
st.write(user_input_aligned)

# Standardize the user input (apply the same scaling as used in training)
user_input_scaled = scaler.transform(user_input_aligned)

# Make predictions
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Display the prediction
st.subheader('Prediction:')
st.write('Heart Disease' if prediction[0] == 1 else 'No Heart Disease')

# Display the prediction probability
st.subheader('Prediction Probability:')
st.write(prediction_proba)
