import streamlit as st

st.title("What does it take to succeed on Spotify?")
st.subheader("By Bruno Manzano and Paula Mateos - Analítica visual 2024")



st.write("""
This web application provides an interactive analysis of the top songs on Spotify in 2023. 
Our goal is to explore key musical features, trends, and artist performances using a detailed dataset sourced from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023?resource=download). 

### Objectives of the Project:
This project is a comprehensive web application designed to explore Spotify data, uncover trends, and predict song success. This project combines Tableau storytelling, a Streamlit application, and Machine Learning (ML) models with Explainable AI (XAI) to deliver valuable insights into Spotify's music ecosystem.
The platform provides users with insights into song release patterns, top-performing artists, albums, and genres, while also offering tools to compare metrics like Spotify Streams and Playlist Reach. By integrating advanced features such as playlist optimization, correlation analysis, and artist collaboration networks, this project provides a holistic understanding of trends in the music industry and actionable insights for artists, stakeholders, and fans.


### About the Dataset:
The dataset contains detailed information about over 10,000 tracks, including:
- Track and Artist Name
- Release Date and Popularity Metrics (e.g., number of streams)
- Audio Features like BPM, Danceability, Energy, Valence, and more
- Inclusion in Spotify, Apple Music, and Deezer playlists and charts

This app is designed to be user-friendly, allowing anyone to explore the fascinating world of music data. Enjoy discovering the secrets behind the hits of 2023! 🎶
""")
