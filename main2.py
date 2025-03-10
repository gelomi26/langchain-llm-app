import streamlit as st
import langchain_helper2 as lch2
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label= "what is the Youtube video URL?",
            max_chars=50
        )
        query = st.sidebar.text_area(
            label= "Ask me about the video?",
            max_chars=50,
            key= "query"
        )

        submit_button = st.form_submit_button(label= 'Submit')

if query and youtube_url:
    db = lch2.create_vector_db_from_youtube_url(youtube_url)
    response, docs = lch2.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width= 80))
    
    