import streamlit as st

st.set_page_config(layout="wide")

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
        padding-left: 3%;  /* Moderate left margin */
        padding-right: 3%; /* Moderate right margin */
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1c1c1c; /* Dark grey */
        color: white; /* Default text color */
        font-family: 'Poppins', sans-serif; /* Apply Poppins font */
    }

    /* Ensure all sidebar text is white */
    [data-testid="stSidebar"] * {
        color: #f0f0f0 !important; /* Force white text */
    }

    /* Top bar (header) background */
    header[data-testid="stHeader"] {
        background-color: #1c1c1c; /* Dark grey */
        font-family: 'Poppins', sans-serif; /* Apply Poppins font */
    }

    /* Top bar text (titles and navigation links) */
    header[data-testid="stHeader"] * {
        color: #f0f0f0 !important; /* Force white text for top bar */
    }

    /* Remove header shadow if any */
    header[data-testid="stHeader"] {
        box-shadow: none !important;
    }

    /* Title text in Poppins font */
    .title {
        font-family: 'Poppins', sans-serif;
        font-weight: 900; /* Make the title more bold */
        padding-left: 10%;  /* Moderate left margin */
        padding-right: 10%; /* Moderate right margin */
        color: #1DB954; 
        text-align: left; /* Center the title */
    }

    /* General text font also in Poppins */
    body, p, label, input, button, select, .css-145kmo2 {
        font-family: 'Poppins', sans-serif;
        padding-left: 10%;  /* Moderate left margin */
        padding-right: 10%; /* Moderate right margin */
        text-align: justify; /* Justify text */
        color: white;
    }

    /* Widen the content area (main body) */
    .main {
        margin-left: 10% !important;  /* Slightly increase left margin */
        margin-right: 10% !important; /* Slightly increase right margin */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Header
st.markdown(
    """
    <div class="title" style="font-size: 50px; color: #1DB954;">What Does It Take to Succeed on Spotify?</div>
    <div class="title "style="font-size: 22px; font-weight: bold; color: #1DB954;">By Bruno Manzano and Paula Mateos - Analítica Visual 2024</div>
    """,
    unsafe_allow_html=True,
)


st.write("""
This web application provides an interactive analysis of the top songs on Spotify in 2023. 
Our goal is to explore key musical features, trends, and artist performances using a detailed dataset sourced from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023?resource=download). 
""")
st.markdown(
    """
    <div class="title" style="font-size: 22px; font-weight: bold; color: #1DB954;">Objectives of the Project</div>
    """,
    unsafe_allow_html=True,
)

st.write("""
This project is a comprehensive web application designed to explore Spotify data, uncover trends, and predict song success. This project combines Tableau storytelling, a Streamlit application, and Machine Learning (ML) models with Explainable AI (XAI) to deliver valuable insights into Spotify's music ecosystem.
The platform provides users with insights into song release patterns, top-performing artists, albums, and genres, while also offering tools to compare metrics like Spotify Streams and Playlist Reach. By integrating advanced features such as playlist optimization, correlation analysis, and artist collaboration networks, this project provides a holistic understanding of trends in the music industry and actionable insights for artists, stakeholders, and fans.
""")

st.markdown(
    """
    <div class="title" style="font-size: 22px; font-weight: bold; color: #1DB954;">About the Dataset</div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
The dataset contains detailed information about over 10,000 tracks, including:
\n -- Track and Artist Name
\n -- Release Date and Popularity Metrics (e.g., number of streams)
\n -- Audio Features like BPM, Danceability, Energy, Valence, and more
\n -- Inclusion in Spotify, Apple Music, and Deezer playlists and charts

This app is designed to be user-friendly, allowing anyone to explore the fascinating world of music data. Enjoy discovering the secrets behind the hits of 2023! 🎶
""")
