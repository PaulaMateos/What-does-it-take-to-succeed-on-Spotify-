import streamlit as st
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page title
st.title("Song Inference with New Songs")

# Introduction text
st.write("""
    This page allows you to input a new song's features (excluding number_characters, which will be calculated from the song name).
    You can input the song name, and the system will predict the number of streams based on the features provided.
""")

# Load the trained XGBoost model
with open('model/xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Load the dataset with track names for artist info
df_with_names = pd.read_csv('data/spotify_with_track_name.csv')

# Load the scaler and label encoders
scaler = pickle.load(open('model/standard_scaler.pkl', 'rb'))

# Load the label encoders
with open('model/label_encoders.pkl', 'rb') as file:
    le_artists = pickle.load(file)
    le_key = pickle.load(file)
    le_mode = pickle.load(file)

# Get the expected columns for the scaler (19 features) and the model (16 features)
scaler_columns = scaler.feature_names_in_
model_columns = xgb_model.get_booster().feature_names

# Input song name
song_name = st.text_input("Enter the song name:")

# Automatically calculate number_characters
number_characters = len(song_name) if song_name else 0

# Input other features (adjust to your dataset features)
artist_name = st.selectbox("Select Artist", df_with_names['artist_name'].unique())
artist_count = st.number_input("Enter the artist's count", min_value=0, value=1)
released_year = st.number_input("Enter the release year", min_value=1900, max_value=2100, value=2023)
released_month = st.number_input("Enter the release month", min_value=1, max_value=12, value=7)
released_day = st.number_input("Enter the release day", min_value=1, max_value=31, value=14)
bpm = st.number_input("Enter Tempo (BPM)", min_value=0, max_value=200, value=120)
key = st.selectbox("Select Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
mode = st.selectbox("Select Mode", ["Major", "Minor"])

# Convert mode to numerical representation (1 for Major, 0 for Minor)
mode_value = 1 if mode == "Major" else 0

# Additional musical features
danceability = st.number_input("Enter Danceability (%)", min_value=0, max_value=100, value=80)
valence = st.number_input("Enter Valence (%)", min_value=0, max_value=100, value=89)
energy = st.number_input("Enter Energy (%)", min_value=0, max_value=100, value=83)
acousticness = st.number_input("Enter Acousticness (%)", min_value=0, max_value=100, value=31)
instrumentalness = st.number_input("Enter Instrumentalness (%)", min_value=0, max_value=100, value=0)
liveness = st.number_input("Enter Liveness (%)", min_value=0, max_value=100, value=8)
speechiness = st.number_input("Enter Speechiness (%)", min_value=0, max_value=100, value=4)

# Encode categorical features using label encoders
artist_name_encoded = le_artists.transform([artist_name])[0]
key_encoded = le_key.transform([key])[0]
mode_encoded = le_mode.transform([mode])[0]

# Prepare a new row for the song data
new_song_data = {
    "track_name": song_name,
    "artist_name": artist_name_encoded,
    "artist_count": artist_count,
    "released_year": released_year,
    "released_month": released_month,
    "released_day": released_day,
    "bpm": bpm,
    "key": key_encoded,
    "mode": mode_encoded,
    "danceability_%": danceability,
    "valence_%": valence,
    "energy_%": energy,
    "acousticness_%": acousticness,
    "instrumentalness_%": instrumentalness,
    "liveness_%": liveness,
    "speechiness_%": speechiness,
    "number_characters": number_characters
}

# Convert to DataFrame
new_song_df = pd.DataFrame([new_song_data])

# Step 1: Separate the categorical and numerical columns
numerical_columns = [col for col in new_song_df.columns if col not in ['track_name', 'artist_name', 'key', 'mode', 'number_characters']]  # Exclude 'number_characters'
categorical_columns = ['key', 'mode', 'artist_name']

# Separate the numerical data for scaling
new_song_numerical = new_song_df[numerical_columns]

# Step 2: Scale the numerical data (ensure the new input has all the features the model expects)
new_song_scaled = scaler.transform(new_song_numerical)

# Step 3: Apply encoding to the categorical columns 'key' and 'mode'
new_song_df['key'] = key_encoded
new_song_df['mode'] = mode_encoded

# Step 4: Combine the scaled numerical data with the encoded categorical features
new_song_for_model = pd.DataFrame(new_song_scaled, columns=numerical_columns)

# Add the encoded categorical features (key and mode) to the scaled data
new_song_for_model['key'] = new_song_df['key']
new_song_for_model['mode'] = new_song_df['mode']
new_song_for_model['artist_name'] = artist_name_encoded  # Add the encoded artist name

# Add 'number_characters' to the final model input, ensuring it’s in the correct position
new_song_for_model['number_characters'] = number_characters

# Reorder columns to match the model's expected feature order
new_song_for_model = new_song_for_model[model_columns]

# Check that the columns match
assert new_song_for_model.columns.tolist() == model_columns, "Mismatch in columns between input and model."


# Step 5: Make predictions using the XGBoost model
prediction = xgb_model.predict(new_song_for_model.values)

# Display the prediction
st.subheader(f"Predicted streams for the song {song_name} by {artist_name}:")
st.markdown(f"<h1 style='text-align: center; color: green;'>{int(prediction[0]):,}</h1>", unsafe_allow_html=True)


# Load the SHAP explainer for visualization
explainer = pickle.load(open('model/shap_explainer.pkl', 'rb'))

# Ensure the data is in the correct format for SHAP
new_song_for_shap = new_song_for_model.values.reshape(1, -1)

# Calculate SHAP values for explanation
shap_values = explainer.shap_values(new_song_for_shap)

# Plot the SHAP summary plot
st.subheader("SHAP Summary Plot:")
st.write("""
    SHAP (SHapley Additive exPlanations) provides a detailed explanation of how each feature impacts the model's prediction.
    Each point represents the contribution of a feature for a specific song, where:
    - **Color**: Red indicates higher feature values, blue indicates lower feature values.
    - **Position**: Features with higher absolute SHAP values have a larger impact on the prediction.
""")

shap.summary_plot(shap_values, features=new_song_for_shap, feature_names=model_columns, show=False)
plt.tight_layout()
st.pyplot(plt)
