import numpy as np
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Import Poppins font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* General app background */
    .stApp {
        background-color: #f0f0f0; /* Light grey */
        font-family: 'Poppins', sans-serif; /* Apply Poppins font */
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #333333; /* Dark grey */
        color: white; /* Default text color */
        font-family: 'Poppins', sans-serif; /* Apply Poppins font */
    }

    /* Ensure all sidebar text is white */
    [data-testid="stSidebar"] * {
        color: white !important; /* Force white text */
    }

    /* Top bar (header) background */
    header[data-testid="stHeader"] {
        background-color: #333333; /* Dark grey */
        font-family: 'Poppins', sans-serif; /* Apply Poppins font */
    }

    /* Top bar text (titles and navigation links) */
    header[data-testid="stHeader"] * {
        color: white !important; /* Force white text for top bar */
    }

    /* Remove header shadow if any */
    header[data-testid="stHeader"] {
        box-shadow: none !important;
    }

    /* Title text in Poppins font */
    .title {
        font-family: 'Poppins', sans-serif;
        font-weight: 900; /* Make the title more bold */
    }

    /* General text font also in Poppins */
    body, p, label, input, button, select, .css-145kmo2 {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Header
st.markdown(
    """
    <div class="title" style="font-size: 50px; font-weight: bold; color: #1DB954;">Global Explainability</div>
    <div class="title" style="font-size: 18px; font-weight: bold; color: #333;">Quick page overview</div>
    """,
    unsafe_allow_html=True,
)

# Introduction text
st.write("""
    This page provides insights into the XGBoost model's predictions using SHAP (SHapley Additive exPlanations).
    The model's predictions for all songs in the dataset are compared with the actual number of streams.
    Additionally, you will find visualizations for feature importances and SHAP values that explain the model's predictions.
""")

# Load the trained XGBoost model
with open('model/xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Load the dataset with track names
df_with_names = pd.read_csv('data/spotify_with_track_name.csv')

# Load the scaler
scaler = pickle.load(open('model/standard_scaler.pkl', 'rb'))

# Get the expected columns for the scaler (19 features) and the model (16 features)
scaler_columns = scaler.feature_names_in_
model_columns = xgb_model.get_booster().feature_names

# Prepare DataFrame to store results
results = []

# Iterate over each song in the dataset
for index, song_data in df_with_names.iterrows():
    # Select the features that the scaler expects
    song_data_for_scaler = song_data[scaler_columns]

    # Scale the data using the scaler
    song_data_scaled = scaler.transform(song_data_for_scaler.values.reshape(1, -1))

    # Create a DataFrame with scaled values and the scaler's columns
    song_data_scaled_df = pd.DataFrame(song_data_scaled, columns=scaler_columns)

    # Add missing columns with default values (e.g., 0 for numeric)
    for col in model_columns:
        if col not in song_data_scaled_df:
            song_data_scaled_df[col] = 0  # Default value

    # Ensure the final DataFrame has the same order as the model expects
    song_data_for_model = song_data_scaled_df[model_columns]

    # Make predictions using the XGBoost model
    prediction = xgb_model.predict(song_data_for_model.values)

    # Store the results for comparison and later display
    results.append({
        'track_name': song_data['track_name'],
        'artist_name': song_data['artist_name'],
        'real_streams': song_data['streams'],
        'predicted_streams': int(prediction[0]),
        'difference': abs(int(prediction[0]) - song_data['streams'])
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display prediction comparison for all songs
st.write("Prediction Comparison for All Songs:")
st.write(results_df)

# Plot the feature importances with barplot
st.write("Feature Importances Across All Songs:")
# Get the feature importances from the XGBoost model
feature_importances = xgb_model.feature_importances_

# Create a DataFrame with the model's feature names and their importances
features_df = pd.DataFrame({'Feature': model_columns, 'Importance': feature_importances})

# Sort the features by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
plt.gca().patch.set_facecolor('#f0f0f0')  # Inner background color
plt.gcf().patch.set_facecolor('#f0f0f0')  # Set the figure background color
plt.title('Feature Importances for Song Predictions')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)

# Load the SHAP explainer
explainer = pickle.load(open('model/shap_explainer.pkl', 'rb'))

# Prepare the data for SHAP values
all_song_data_for_model = []

for index, song_data in df_with_names.iterrows():
    song_data_for_scaler = song_data[scaler_columns]
    song_data_scaled = scaler.transform(song_data_for_scaler.values.reshape(1, -1))
    song_data_scaled_df = pd.DataFrame(song_data_scaled, columns=scaler_columns)

    # Add missing columns with default values (e.g., 0 for numeric)
    for col in model_columns:
        if col not in song_data_scaled_df:
            song_data_scaled_df[col] = 0

    # Ensure the final DataFrame has the same order as the model expects
    song_data_for_model = song_data_scaled_df[model_columns]
    all_song_data_for_model.append(song_data_for_model.values.reshape(1, -1))

# Convert the list to a 2D array (songs x features)
all_song_data_for_model = np.vstack(all_song_data_for_model)

# Calculate SHAP values for all songs
shap_values = explainer.shap_values(all_song_data_for_model)

# Plot the SHAP summary plot
st.write("SHAP Summary Plot Across All Songs:")
shap.summary_plot(shap_values, features=all_song_data_for_model, feature_names=model_columns, show=False)
plt.gca().patch.set_facecolor('#f0f0f0')  # Inner background color
plt.gcf().patch.set_facecolor('#f0f0f0')  # Set the figure background color
plt.tight_layout()
st.pyplot(plt)
