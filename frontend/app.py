import streamlit as st
import requests
import os



# Backend API URL
API_URL = os.getenv("API_URL", "http://backend:8080/query")  # Use Docker service name for backend

# Streamlit Page Config
st.set_page_config(page_title="HR RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 HR Policy Chatbot")
st.markdown("Ask me anything about the HR policies!")

# Chat History
if "history" not in st.session_state:
    st.session_state.history = []

# Function to get response from backend
def get_answer(question):
    try:
        response = requests.post(API_URL, json={"question": question})
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        else:
            return f"⚠️ Error: {response.status_code}"
    except Exception as e:
        return f"⚠️ Could not connect to backend: {e}"

# Input box with Enter-to-submit functionality
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", placeholder="Type your question and press Enter")
    submit = st.form_submit_button("Ask")

# Handle user input
if submit and user_input:
    answer = get_answer(user_input)
    st.session_state.history.append((user_input, answer))

# Display chat history
for q, a in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
