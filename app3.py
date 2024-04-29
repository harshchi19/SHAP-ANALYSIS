import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import shap
import matplotlib.pyplot as plt

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define global encoder and scaler objects
encoder = OneHotEncoder()
scaler = StandardScaler()

# Functions for data preparation
def prepare_data():
    # Simulating data
    np.random.seed(42)
    n_samples = 1000
    destination = np.random.choice(['Delhi', 'Mumbai', 'Goa', 'Kerala', 'Jaipur'], n_samples)
    accommodation = np.random.choice(['Hotel', 'Hostel', 'Guest House', 'Resort'], n_samples)
    travel_mode = np.random.choice(['Train', 'Plane', 'Bus', 'Car'], n_samples)
    duration = np.random.normal(7, 2, n_samples).astype(int)  # Mean duration 7 days
    attractions = np.random.poisson(5, n_samples)  # Mean 5 attractions

    budget = 2000 * duration + 1500 * attractions + np.random.normal(5000, 1500, n_samples)
    budget = budget * (1 + 0.1 * (destination == 'Goa'))  # Increase budget for Goa

    data = pd.DataFrame({
        'Destination': destination,
        'Accommodation': accommodation,
        'Travel Mode': travel_mode,
        'Duration': duration,
        'Attractions': attractions,
        'Budget': budget
    })
    
    return data

def encode_and_scale_data(data):
    global encoder, scaler  # Access global encoder and scaler objects
    # Encoding categorical data
    encoded_features = encoder.fit_transform(data[['Destination', 'Accommodation', 'Travel Mode']]).toarray()
    feature_names = encoder.get_feature_names_out(['Destination', 'Accommodation', 'Travel Mode'])
    encoded_data = pd.DataFrame(encoded_features, columns=feature_names)

    # Standardizing numerical data
    scaled_features = scaler.fit_transform(data[['Duration', 'Attractions']])
    scaled_data = pd.DataFrame(scaled_features, columns=['Scaled Duration', 'Scaled Attractions'])

    # Final dataset
    final_data = pd.concat([encoded_data, scaled_data, data['Budget']], axis=1)
    
    return final_data

# Function for model training
def train_model(final_data):
    # Splitting the dataset
    X = final_data.drop('Budget', axis=1)
    y = final_data['Budget']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Function to estimate budget based on user input
def estimate_budget(model, user_input):
    global encoder, scaler  # Access global encoder and scaler objects
    # Encode user input
    user_data = pd.DataFrame(user_input, index=[0])
    user_encoded = encoder.transform(user_data[['Destination', 'Accommodation', 'Travel Mode']]).toarray()
    user_features = np.concatenate([user_encoded, scaler.transform(user_data[['Duration', 'Attractions']])], axis=1)
    
    # Predict budget
    budget = model.predict(user_features)
    
    return budget

# Function to explain predictions using SHAP
def explain_shap(model, X_train):
    # Initialize the SHAP Explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    return shap_values

# Streamlit app
def main():
    st.title('Travel Budget Estimation')

    # Input form
    st.sidebar.header('Input Parameters')
    destination = st.sidebar.selectbox('Destination', ['Delhi', 'Mumbai', 'Goa', 'Kerala', 'Jaipur'])
    accommodation = st.sidebar.selectbox('Accommodation Type', ['Hotel', 'Hostel', 'Guest House', 'Resort'])
    travel_mode = st.sidebar.selectbox('Travel Mode', ['Train', 'Plane', 'Bus', 'Car'])
    duration = st.sidebar.slider('Duration (days)', 1, 15, 7)
    attractions = st.sidebar.slider('Attractions', 1, 20, 5)
    
    user_input = {
        'Destination': [destination],
        'Accommodation': [accommodation],
        'Travel Mode': [travel_mode],
        'Duration': [duration],
        'Attractions': [attractions]
    }
    
    # Prepare data
    data = prepare_data()
    final_data = encode_and_scale_data(data)
    
    # Train model
    model = train_model(final_data)
    
    # Explain predictions using SHAP
    shap_values = explain_shap(model, final_data.drop('Budget', axis=1))
    
    # Estimate budget
    budget = estimate_budget(model, user_input)
    
    # Output
    st.subheader('Estimated Budget')
    st.write(f'INR {budget[0]:,.2f}')

    # Display SHAP summary plot
    st.subheader('SHAP Summary Plot')

   # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

   # Plot the SHAP summary plot
    shap.summary_plot(shap_values, final_data.drop('Budget', axis=1), feature_names=final_data.drop('Budget', axis=1).columns.tolist(), show=False)

   # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)


if __name__ == '__main__':
    main()
