import streamlit as st
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler

# Page title
st.title("XGBoost Model Explainability")

# Introduction text
st.write("""
    This page provides insights into the XGBoost model's predictions using SHAP (SHapley Additive exPlanations).
    You can select a song from the dataset, get the model's prediction, and visualize how different features contribute to the prediction.
""")

# Load the trained XGBoost model
with open('model/xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Load the dataset for inference (after encoding and normalization)
df = pd.read_csv('data/spotify_inference.csv')

df_labels = pd.read_csv('data/spotify_cleaned.csv')

# Display the first few rows of the dataset
st.write("Sample data:")
st.write(df_labels.head())

# Load LabelEncoders
with open('model/label_encoders.pkl', 'rb') as file:
    le_song_name = pickle.load(file)
    le_artists = pickle.load(file)
    le_key = pickle.load(file)
    le_mode = pickle.load(file)

# Select a song from the dataset (Display song names) with label encoder
song_name = st.selectbox("Select a song:", le_song_name.inverse_transform(df['track_name']))

# Get the index of the selected song
song_index = df[df['track_name'] == le_song_name.transform([song_name])[0]].index[0]

# Get the data for the selected song
song_data = df.iloc[song_index]

# Import the scaler
scaler = pickle.load(open('model/standard_scaler.pkl', 'rb'))

# Get the expected columns for the scaler (19 features) and the model (16 features)
scaler_columns = scaler.feature_names_in_
model_columns = xgb_model.get_booster().feature_names

# Select the features that the scaler expects
song_data_for_scaler = song_data[scaler_columns]

# Now, scale the data using the scaler
song_data_scaled = scaler.transform(song_data_for_scaler.values.reshape(1, -1))

# Ensure song_data has the same columns as the model expects (16 features)
song_data_for_model = song_data[model_columns]

# Make predictions using the XGBoost model
prediction = xgb_model.predict(song_data_for_model.values.reshape(1, -1))

# Display the real values for the number of streams and the predicted value
st.write("Prediction comparison:")
st.write(df_labels.iloc[song_index][['track_name', 'artist_name', 'streams']])
st.write("")
st.write("Predicted streams:")
# Now transform the prediction back to number of streams
st.write(int(prediction[0]))
st.write("Difference:")
st.write(abs(int(prediction[0]) - df_labels.iloc[song_index]['streams']))

# Plot the feature importances with barplot
st.write("Feature Importances:")
import matplotlib.pyplot as plt
import seaborn as sns

# Get the feature importances from the XGBoost model
feature_importances = xgb_model.feature_importances_

# Ensure the columns match the model's expected feature names
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
song_data_for_shap = song_data[model_columns].values.reshape(1, -1)  # Ensure it's 2D (1, n_features)

# Calculate SHAP values
shap_values = explainer.shap_values(song_data_for_shap)

# Plot the SHAP summary plot
st.write("SHAP Summary Plot:")
shap.summary_plot(shap_values, features=song_data_for_shap, feature_names=model_columns, show=False)
# Order by the absolute mean SHAP value
plt.tight_layout()
st.pyplot(plt)