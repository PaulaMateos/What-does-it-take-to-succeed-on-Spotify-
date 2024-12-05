import streamlit as st
import streamlit.components.v1 as components

# Title of the Streamlit App
st.title("How do non musical features affect the succes of a song?")


st.write(
    """
    This page displays an interactive Tableau dashboard embedded directly into the web app. 
    Explore and interact with the data insights.
    """
)

# Embed Tableau Public visualization using the copied HTML
st.markdown("""
<div class='tableauPlaceholder' id='viz1733420672030' style='position: relative'>
    <noscript>
        <a href='#'>
            <img alt='Non musical characteristics ' src='https://public.tableau.com/static/images/Sp/Spotyfysucces/Nonmusicalcharacteristics/1_rss.png' style='border: none' />
        </a>
    </noscript>
    <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='Spotyfysucces/Nonmusicalcharacteristics' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/Sp/Spotyfysucces/Nonmusicalcharacteristics/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='es-ES' />
    </object>
</div>
<script type='text/javascript'>
    var divElement = document.getElementById('viz1733420672030');
    var vizElement = divElement.getElementsByTagName('object')[0];
    if ( divElement.offsetWidth > 800 ) {
        vizElement.style.width='900px';
        vizElement.style.height='2027px';
    } else if ( divElement.offsetWidth > 500 ) {
        vizElement.style.width='900px';
        vizElement.style.height='2027px';
    } else {
        vizElement.style.width='100%';
        vizElement.style.height='1277px';
    }
    var scriptElement = document.createElement('script');
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
""", unsafe_allow_html=True)

# Embed the Tableau Public visualization using iframe
tableau_url = "https://public.tableau.com/views/Spotyfysucces/Nonmusicalcharacteristics?:language=en&:display_count=y&:origin=viz_share_link"

components.iframe(tableau_url, width=800, height=600)