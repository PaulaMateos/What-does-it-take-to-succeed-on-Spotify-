import streamlit as st

# Title of the Streamlit App
st.title("How do non musical features affect the succes of a song?")


st.write(
    """
    This page displays an interactive Tableau dashboard embedded directly into the web app. 
    Explore and interact with the data insights.
    """
)

# Dashboard 1
st.components.v1.html(
    '<iframe src="https://public.tableau.com/views/Spotyfysucces/Nonmusicalcharacteristics?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link" width="100%" height="100%"></iframe>'
)
