import streamlit as st
from workflow import run_user_query  # import your workflow function

# Optional: set a page title and layout
st.set_page_config(page_title="lOMNI Chatbot", layout="centered")

# Initialize session state for storing conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        }
    ]

# Display existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom
user_input = st.chat_input("Ask a question...")

if user_input:
    # 1) Display user message in the chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2) Run the workflow with the user's query
    result = run_user_query(user_input)  # returns {"category": ..., "response": ...}
    assistant_reply = result["response"]

    # 3) Display the assistant's response in the chat
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
