import streamlit as st
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler

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
st.write("Sample data:")
st.write(df_with_names.head())

# Select a song from the dataset (Display song names)
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
st.write("Prediction comparison:")
st.write(song_data[['track_name', 'artist_name', 'streams']])
st.write("")
st.write("Predicted streams:")
# Transform the prediction back to number of streams if needed
st.write(int(prediction[0]))
st.write("Difference:")
st.write(abs(int(prediction[0]) - song_data['streams']))

# Plot the feature importances with barplot
st.write("Feature Importances:")
import matplotlib.pyplot as plt
import seaborn as sns

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
explainer = pickle.load(open('model/shap_explainer.pkl', 'rb'))

# Ensure song_data is in the correct format (2D array) for SHAP
song_data_for_shap = song_data_for_model.values.reshape(1, -1)

# Calculate SHAP values
shap_values = explainer.shap_values(song_data_for_shap)

# Plot the SHAP summary plot
st.write("SHAP Summary Plot:")
shap.summary_plot(shap_values, features=song_data_for_shap, feature_names=model_columns, show=False)
# Order by the absolute mean SHAP value
plt.tight_layout()
st.pyplot(plt)
