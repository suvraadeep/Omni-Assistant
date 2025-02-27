# workflow.py
from langgraph.graph import StateGraph, END, START
from IPython.display import display, Image, Markdown
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from router import (
    categorize,
    handle_learning_resource,
    handle_resume_making,
    handle_interview_preparation,
    job_search,
    mock_interview,
    interview_topics_questions,
    tutorial_agent,
    ask_query_bot,
    route_query,
    route_interview,
    route_learning
)
from agents.ResumeMaker import ResumeMaker
from agents.JobSearch import JobSearch
from agents.InterviewAgent import InterviewAgent
from agents.LearningResourceAgent import LearningResourceAgent

from typing import Dict, TypedDict

from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from your .env file



# Define a State type for consistent state dictionaries
class State(TypedDict):
    query: str
    category: str
    response: str

# Create the workflow graph using our State type
workflow = StateGraph(State)

# Add nodes for each state in the workflow
workflow.add_node("categorize", categorize)  # Initial categorization node
workflow.add_node("handle_learning_resource", handle_learning_resource)  # Learning queries
workflow.add_node("handle_resume_making", handle_resume_making)  # Resume making queries
workflow.add_node("handle_interview_preparation", handle_interview_preparation)  # Interview prep queries
workflow.add_node("job_search", job_search)  # Job search queries
workflow.add_node("mock_interview", mock_interview)  # Mock interview sessions
workflow.add_node("interview_topics_questions", interview_topics_questions)  # Interview topic queries
workflow.add_node("tutorial_agent", tutorial_agent)  # Tutorial creation agent
workflow.add_node("ask_query_bot", ask_query_bot)  # General Q&A for learning

# Define the starting edge to the categorization node
workflow.add_edge(START, "categorize")

# Add conditional edges based on primary category routing
workflow.add_conditional_edges(
    "categorize",
    route_query,
    {
        "handle_learning_resource": "handle_learning_resource",
        "handle_resume_making": "handle_resume_making",
        "handle_interview_preparation": "handle_interview_preparation",
        "job_search": "job_search"
    }
)

# Add conditional edges for interview preparation routing
workflow.add_conditional_edges(
    "handle_interview_preparation",
    route_interview,
    {
        "mock_interview": "mock_interview",
        "interview_topics_questions": "interview_topics_questions",
    }
)

# Add conditional edges for learning queries routing
workflow.add_conditional_edges(
    "handle_learning_resource",
    route_learning,
    {
        "tutorial_agent": "tutorial_agent",
        "ask_query_bot": "ask_query_bot",
    }
)

# Define edges that lead to the end of the workflow
workflow.add_edge("handle_resume_making", END)
workflow.add_edge("job_search", END)
workflow.add_edge("interview_topics_questions", END)
workflow.add_edge("mock_interview", END)
workflow.add_edge("ask_query_bot", END)
workflow.add_edge("tutorial_agent", END)

# Set the entry point
workflow.set_entry_point("categorize")

# Compile the workflow graph into an application
app = workflow.compile()

def run_user_query(query: str) -> dict:
    """Process a user query through the LangGraph workflow."""
    results = app.invoke({"query": query})
    return {
        "category": results.get("category", "Unknown"),
        "response": results.get("response", "No response generated.")
    }