import streamlit as st
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("Model Validation")

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

# Display the first few rows of the dataset
st.subheader("Sample Data")
st.write("Here is a preview of the data used for model inference and validation:")
st.write(df_with_names.head())

# Select a song from the dataset (Display song names)
st.subheader("Select a Song")
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
st.subheader("Prediction Comparison")
st.write("The following table displays the selected song's details, including its actual streams:")
st.write(song_data[['track_name', 'artist_name', 'streams']])
st.write("")
st.subheader("Predicted Streams")
st.write("This is the predicted number of streams based on the model:")
st.markdown(f"<h1 style='text-align: center; color: green;'>{int(prediction[0]):,}</h1>", unsafe_allow_html=True)

st.subheader("Difference")
st.write("This shows the difference between the actual and predicted streams:")
st.markdown(
    f"<h2 style='text-align: center; color: red;'>{abs(int(prediction[0]) - song_data['streams']):,}</h2>",
    unsafe_allow_html=True,
)

# Plot the feature importances with a bar plot
st.subheader("Feature Importances")
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
sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)

# Load the SHAP explainer
st.subheader("SHAP Summary Plot")
st.write("""
    SHAP (SHapley Additive exPlanations) provides a detailed explanation of how each feature impacts the model's prediction.
    Each point represents the contribution of a feature for a specific song, where:
    - **Color**: Red indicates higher feature values, blue indicates lower feature values.
    - **Position**: Features with higher absolute SHAP values have a larger impact on the prediction.
""")

explainer = pickle.load(open('model/shap_explainer.pkl', 'rb'))

# Ensure song_data is in the correct format (2D array) for SHAP
song_data_for_shap = song_data_for_model.values.reshape(1, -1)

# Calculate SHAP values
shap_values = explainer.shap_values(song_data_for_shap)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, features=song_data_for_shap, feature_names=model_columns, show=False)
plt.tight_layout()
st.pyplot(plt)
