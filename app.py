import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Function to load a file with error handling
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except ImportError as e:
        st.error(f"Import error: {str(e)}. Make sure all dependencies are installed.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {file_path}\n{str(e)}")
        return None

# Load the best model, scaler, and feature names
model = load_pickle('best_model.pkl')
scaler = load_pickle('scaler.pkl')
features = load_pickle('features.pkl')

if model is None or scaler is None or features is None:
    st.stop()  # Stop the script if any of the files couldn't be loaded

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
st.write(f"Probability of having heart disease: {prediction_proba[0][1]:.2f}")

# Add information about model performance
st.subheader('Model Information:')
st.write(f"Model: {type(model).__name__}")
