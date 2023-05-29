import streamlit as st
import pickle
import numpy as np
import torch
from torch.nn import Linear  # Import the model class
import joblib
from sklearn.preprocessing import StandardScaler

# Load the PyTorch model
pytorch_model = Linear(6, 1)  # Initialize your model
pytorch_model.load_state_dict(torch.load('final_model.pt'))
pytorch_model.eval()

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Input features
features = ['Shooting_Efficiency', '3P_Efficiency', 'FT_Efficiency', 'RB_Rate', 'AST_Rate', 'TO_Rate']

# Create input sliders for each feature
inputs = {feature: st.slider(feature, 0.00, 1.00) for feature in features}

if st.button('Predict'):
    feature_array = np.array([list(inputs.values())])

    # Scale the input features
    feature_array = scaler.transform(feature_array)

    # Transform the feature_array to a PyTorch tensor
    feature_tensor = torch.from_numpy(feature_array).float()

    # Perform prediction with PyTorch model
    prediction = pytorch_model(feature_tensor)

    st.write('The predicted PER is ', prediction.item())
