import langchain_helper as lch
import streamlit as st

st.title("Vocabulary")

user_animal_type = st.sidebar.selectbox("Choose a model", ("gpt-4o", "Llama3-7b", "Claude sonnet 3.5"))

if user_animal_type == "Cat":
    pet_color = st.sidebar.text_area("What color is your cat?", max_chars=20)

if user_animal_type == "Dog":
    pet_color = st.sidebar.text_area("What color is your dog?", max_chars=20)

if user_animal_type == "Bird":
    pet_color = st.sidebar.text_area("What color is your bird?", max_chars=20)

if pet_color:
    response = lch.generate_pet_name(user_animal_type, pet_color)
    st.text(response['pet_name'])