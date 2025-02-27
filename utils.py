from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
import os
from datetime import datetime
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, TypedDict
from IPython.display import display, Image, Markdown

# Define a State type for consistent state dictionaries
class State(TypedDict):
    query: str
    category: str
    response: str





def trim_conversation(prompt):
    """Trims conversation history to retain only the latest messages within the limit."""
    max_messages = 10 
    return trim_messages(
        prompt,
        max_tokens=max_messages,
        strategy="last",
        token_counter=len, 
        start_on="human", 
        include_system=True, 
        allow_partial=False,
    )

def save_file(data, filename):
    """Saves data to a markdown file with a timestamped filename."""
    folder_name = "Agent_output"  
    os.makedirs(folder_name, exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename}_{timestamp}.md"
    file_path = os.path.join(folder_name, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)
        # print(f"File '{file_path}' created successfully.")  # Commented out for Streamlit-only output
    return file_path

def show_md_file(file_path):
    """Displays the content of a markdown file in Streamlit."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    st.markdown(content)