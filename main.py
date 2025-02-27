# main.py
import streamlit as st
from workflow import run_user_query, app
from PIL import Image
from langgraph import MermaidDrawMethod

st.title("OMNI Chatbot with Workflow")

st.header("Workflow Graph")
# Generate the workflow graph as a PNG image using Mermaid's API
workflow_img = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
st.image(workflow_img, caption="Workflow Graph")

st.header("Ask Your Query")
user_query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if user_query:
        result = run_user_query(user_query)
        st.write("**Primary Category:**", result["category"])
        st.markdown("### Response")
        st.write(result["response"])