import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import geopy.distance

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
# Function for model training
def train_model(final_data):
    # Splitting the dataset
    X = final_data.drop('Budget', axis=1)
    y = final_data['Budget']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate Mean Squared Error
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    
    return model, mse


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
# Function for plotting
def plot_data_distribution(data):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data['Duration'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Duration (Days)')

    plt.subplot(1, 2, 2)
    sns.histplot(data['Attractions'], bins=20, kde=True, color='green')
    plt.title('Distribution of Attractions Visited')

    plt.tight_layout()
    st.pyplot(plt.gcf())  # Pass the current figure explicitly

def plot_box_plots(data):
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    sns.boxplot(x='Destination', y='Budget', data=data)
    plt.xticks(rotation=45)
    plt.title('Budget by Destination')

    plt.subplot(1, 3, 2)
    sns.boxplot(x='Accommodation', y='Budget', data=data)
    plt.xticks(rotation=45)
    plt.title('Budget by Accommodation Type')

    plt.subplot(1, 3, 3)
    sns.boxplot(x='Travel Mode', y='Budget', data=data)
    plt.xticks(rotation=45)
    plt.title('Budget by Travel Mode')

    plt.tight_layout()
    st.pyplot(plt.gcf())  # Pass the current figure explicitly

def plot_count_plots(data):
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    sns.countplot(x='Destination', data=data)
    plt.title('Frequency of Destinations')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    sns.countplot(x='Accommodation', data=data)
    plt.title('Frequency of Accommodation Types')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    sns.countplot(x='Travel Mode', data=data)
    plt.title('Frequency of Travel Modes')
    plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot(plt.gcf())  # Pass the current figure explicitly


def plot_count_plots(data):
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    sns.countplot(x='Destination', data=data)
    plt.title('Frequency of Destinations')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    sns.countplot(x='Accommodation', data=data)
    plt.title('Frequency of Accommodation Types')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    sns.countplot(x='Travel Mode', data=data)
    plt.title('Frequency of Travel Modes')
    plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot()

# Function for calculating distance to airport
def calculate_distance(city):
    # Coordinates for cities and airports (adding Kerala)
    coords_city = {
        'Delhi': (28.7041, 77.1025),
        'Goa': (15.2993, 74.1240),
        'Kerala': (10.8505, 76.2711),  # Added coordinates for Kerala
        'Mumbai': (19.0760, 72.8777),
        'Jaipur': (26.9124, 75.7873)
    }

    coords_airport = {
        'Delhi': (28.5562, 77.1000),
        'Goa': (15.3802, 73.8311),
        'Kerala': (10.1520, 76.4019),  # Added coordinates for the nearest airport in Kerala
        'Mumbai': (19.0896, 72.8656),
        'Jaipur': (26.8287, 75.8056)
    }
    
    # Default coordinates (central India, arbitrary example)
    default_coords = (22.9734, 78.6569)  
    
    city_coords = coords_city.get(city, default_coords)
    airport_coords = coords_airport.get(city, default_coords)
    
    return geopy.distance.geodesic(city_coords, airport_coords).km


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

    # Data distribution plots
    st.subheader('Data Distribution')
    plot_data_distribution(data)
    
    # Box plots
    st.subheader('Budget Analysis')
    plot_box_plots(data)
    
    # Count plots
    st.subheader('Frequency Analysis')
    plot_count_plots(data)
    
    # Calculate distance to airport
    data['Distance_to_Airport'] = data['Destination'].apply(calculate_distance)
    st.subheader('Distance to Nearest Airport')
    st.write(data)
    
    # Train model
    # Train model
    model, mse = train_model(final_data)

    # Display MSE
    st.subheader('Model Evaluation')
    st.write(f'Mean Squared Error: {mse:.2f}')

    
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
