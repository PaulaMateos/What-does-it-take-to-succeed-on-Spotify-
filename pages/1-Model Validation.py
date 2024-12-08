import streamlit as st
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="centered")


# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Import Poppins font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* General app background */
    .stApp {
        background-color: #333333; /* Light grey */
        font-family: 'Poppins', sans-serif; /* Apply Poppins font */
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1c1c1c; /* Dark grey */
        color: white; /* Default text color */
        font-family: 'Poppins', sans-serif; /* Apply Poppins font */
    }

    /* Ensure all sidebar text is white */
    [data-testid="stSidebar"] * {
        color: white !important; /* Force white text */
    }

    /* Top bar (header) background */
    header[data-testid="stHeader"] {
        background-color: #1c1c1c; /* Dark grey */
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
        color: #1DB954; /* Title color */
        text-align: left; /* Center the title */
    }
    /* General text font also in Poppins */
    body, p, label, input, button, select, .css-145kmo2 {
        font-family: 'Poppins', sans-serif;
        text-align: justify; /* Justify text */
        color: white; /* Default text color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Header
st.markdown(
    """
    <div class="title" style="font-size: 50px; font-weight: bold; color: #1DB954;">Model Validation</div>
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">Quick page overview</div>
    """,
    unsafe_allow_html=True,
)

# Introduction text
st.write("""
    This page provides insights into the XGBoost model's predictions using SHAP (SHapley Additive exPlanations).
    You can select a song from the dataset, get the model's prediction, and visualize how different features contribute to the prediction.
""")

# Load the trained XGBoost model
with open('model/xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Load the dataset with track names
df_with_names = pd.read_csv('data/spotify_with_track_name.csv')

st.markdown(
    """
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">Sample Data</div>
    """,
    unsafe_allow_html=True,
)
st.write("Here is a preview of the data used for model inference and validation:")
st.write(df_with_names.head())

# Select a song from the dataset (Display song names)
st.markdown(
    """
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">Select a Song</div>
    """,
    unsafe_allow_html=True,
)

st.write("Choose a song to analyze. The model will predict its streams and compare it to the actual value.")
song_name = st.selectbox("Select a song:", df_with_names['track_name'].unique())

# Get the data for the selected song
song_data = df_with_names[df_with_names['track_name'] == song_name].iloc[0]

# Load the scaler
scaler = pickle.load(open('model/standard_scaler.pkl', 'rb'))

# Get the expected columns for the scaler (19 features) and the model (16 features)
scaler_columns = scaler.feature_names_in_
model_columns = xgb_model.get_booster().feature_names

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

# Display the real values for the number of streams and the predicted value
st.markdown(
    """
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">Prediction Comparison</div>
    """,
    unsafe_allow_html=True,
)
st.write("The following table displays the selected song's details, including its actual streams:")
st.write(song_data[['track_name', 'artist_name', 'streams']])
st.write("")
st.markdown(
    """
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">Predicted Streams</div>
    """,
    unsafe_allow_html=True,
)
st.write("This is the predicted number of streams based on the model:")
st.markdown(f"<h1 style='text-align: center; color: #1DB954;'>{int(prediction[0]):,}</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">Difference</div>
    """,
    unsafe_allow_html=True,
)
st.write("This shows the difference between the actual and predicted streams:")
st.markdown(
    f"<h2 style='text-align: center; color: #f5452a;'>{abs(int(prediction[0]) - song_data['streams']):,}</h2>",
    unsafe_allow_html=True,
)

# Plot the feature importances with a bar plot
st.markdown(
    """
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">Feature Importances</div>
    """,
    unsafe_allow_html=True,
)
st.write("""
    Feature importance shows how much each feature contributes to the model's prediction. 
    Higher importance means the feature has a larger impact on the model's decisions.
""")

# Get the feature importances from the XGBoost model
feature_importances = xgb_model.feature_importances_

# Create a DataFrame with the model's feature names and their importances
features_df = pd.DataFrame({'Feature': model_columns, 'Importance': feature_importances})

# Sort the features by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")  # Set background to light grey for seaborn
sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
plt.gca().patch.set_facecolor('#333333')  # Inner background color
plt.title('Feature Importances', color='white')
plt.xlabel('Importance', color='white')
plt.ylabel('Feature', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gcf().patch.set_facecolor('#333333')  # Set the figure background color
st.pyplot(plt)

# Load the SHAP explainer
st.markdown(
    """
    <div class="title" style="font-size: 18px; font-weight: bold; color: #1DB954;">SHAP Summary Plot</div>
    """,
    unsafe_allow_html=True,
)
st.write("""
    SHAP (SHapley Additive exPlanations) provides a detailed explanation of how each feature impacts the model's prediction.
    Each point represents the contribution of a feature for a specific song, where:
    \n- **Color**: Red indicates higher feature values, blue indicates lower feature values.
    \n- **Position**: Features with higher absolute SHAP values have a larger impact on the prediction.
""")

explainer = pickle.load(open('model/shap_explainer.pkl', 'rb'))

# Ensure song_data is in the correct format (2D array) for SHAP
song_data_for_shap = song_data_for_model.values.reshape(1, -1)

# Calculate SHAP values
shap_values = explainer.shap_values(song_data_for_shap)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, features=song_data_for_shap, feature_names=model_columns, show=False)
plt.gca().patch.set_facecolor('#333333')  # Inner background color
plt.gcf().patch.set_facecolor('#333333')  # Set the figure background color
# labels to white
plt.xticks(color='white')
plt.yticks(color='white')
# legend to white
plt.tight_layout()
st.pyplot(plt)
