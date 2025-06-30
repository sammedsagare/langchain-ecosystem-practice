import streamlit as st
import yt_transcript as ytt
import textwrap

st.set_page_config(
    page_title="YouTube Assistant",
    page_icon="📹",
    layout="wide"
)

st.markdown(
    """
    <style>
    .title {
        font-size: 2.5em;
        color: #f9f9f9;
        text-align: left;
        margin-bottom: 20px;
    }
    </style>
    <div class="title">YouTube Assistant</div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## Input Details")
    with st.form(key='my_form'):
        youtube_url = st.text_area(
            label="Enter the YouTube video URL:",
            max_chars=100
        )
        query = st.text_area(
            label="Ask a question about the video:",
            max_chars=100,
            key="query"
        )
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    db = ytt.vector_db_yt_url(youtube_url)
    response, docs = ytt.get_response_from_query(db, query)
    st.markdown("## Answer:")
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 5px;">
        {textwrap.fill(response, width=100)}
        </div>
        """,
        unsafe_allow_html=True
    )