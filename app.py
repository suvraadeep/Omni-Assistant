# main.py
import streamlit as st
from workflow import run_user_query, app
from PIL import Image
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="lOMNI Chatbot", layout="centered")
st.title("lOMNI Chatbot with Workflow")

# Optionally display the workflow graph in an expander
with st.expander("Show Workflow Graph"):
    workflow_img = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    st.image(workflow_img, caption="Workflow Graph")

# Initialize session state for chat conversation if not already set
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

# Display the chat conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input field
user_input = st.chat_input("Ask a question...")

if user_input:
    # Append user's message to conversation
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process the query through the workflow
    result = run_user_query(user_input)
    assistant_reply = result["response"]
    
    # Append assistant's reply to conversation
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)