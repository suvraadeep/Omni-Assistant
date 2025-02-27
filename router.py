from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_groq import ChatGroq
from utils import show_md_file
from langchain_core.prompts import ChatPromptTemplate

from agents.ResumeMaker import ResumeMaker
from agents.JobSearch import JobSearch
from agents.InterviewAgent import InterviewAgent
from agents.LearningResourceAgent import LearningResourceAgent

llm=ChatGroq(model="deepseek-r1-distill-llama-70b")

from typing import Dict, TypedDict

# Define a State type for consistent state dictionaries
class State(TypedDict):
    query: str
    category: str
    response: str

def categorize(state):
    """Categorizes the user query into one of four main categories."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n"
        "Give the number only as an output.\n\n"
        "Examples:\n"
        "1. Query: 'What are the basics of generative AI, and how can I start learning it?' -> 1\n"
        "2. Query: 'Can you help me improve my resume for a tech position?' -> 2\n"
        "3. Query: 'What are some common questions asked in AI interviews?' -> 3\n"
        "4. Query: 'Are there any job openings for AI engineers?' -> 4\n\n"
        "Now, categorize the following customer query:\n"
        "Query: {query}"
    )
    chain = prompt | llm
    # print('Categorizing the customer query...')  # Commented out
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def handle_learning_resource(state):
    """Further categorizes learning-related queries into Tutorial or Question."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Tutorial: For queries related to creating tutorials, blogs, or documentation on generative AI.\n"
        "- Question: For general queries asking about generative AI topics.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. User query: 'How to create a blog on prompt engineering for generative AI?' -> Category: Tutorial\n"
        "2. User query: 'Can you provide a step-by-step guide on fine-tuning a generative model?' -> Category: Tutorial\n"
        "3. User query: 'Provide me the documentation for Langchain?' -> Category: Tutorial\n"
        "4. User query: 'What are the main applications of generative AI?' -> Category: Question\n"
        "5. User query: 'Is there any generative AI course available?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )
    chain = prompt | llm
    # print('Further categorizing learning query...')  # Commented out
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def handle_interview_preparation(state):
    """Further categorizes interview-related queries into Mock or Question."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Mock: For requests related to mock interviews.\n"
        "- Question: For general queries asking about interview topics or preparation.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Examples:\n"
        "1. User query: 'Can you conduct a mock interview with me for a Gen AI role?' -> Category: Mock\n"
        "2. User query: 'What topics should I prepare for an AI Engineer interview?' -> Category: Question\n"
        "3. User query: 'I need to practice interview focused on Gen AI.' -> Category: Mock\n"
        "4. User query: 'Can you list important coding topics for AI tech interviews?' -> Category: Question\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )
    chain = prompt | llm
    # print('Further categorizing interview query...')  # Commented out
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def job_search(state):
    """Handles job search queries."""
    
    prompt = ChatPromptTemplate.from_template(
        '''Your task is to refactor and make a .md file for this content which includes
        the jobs available in the market. Refactor it such that the user can easily refer to it. Content: {result}'''
    )
    jobSearch = JobSearch(prompt)
    path = jobSearch.find_jobs(state["query"])
    show_md_file(path)
    return {"response": path}

def handle_resume_making(state):
    """Handles resume making queries."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a skilled resume expert with extensive experience in crafting resumes tailored for tech roles, especially in AI and Generative AI. 
        Your task is to create a resume template for an AI Engineer specializing in Generative AI, incorporating trending keywords and technologies in the current job market. 
        Ask the user for necessary details in 4-5 steps and output a final .md file resume.'''),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    resumeMaker = ResumeMaker(prompt)
    path = resumeMaker.Create_Resume(state["query"])
    show_md_file(path)
    return {"response": path}

def ask_query_bot(state):
    """Handles general learning queries in Q&A mode."""
    system_message = '''You are an expert Generative AI Engineer. Assist users by providing insightful solutions and expert advice on their queries.
    Engage in a back-and-forth chat session to address user queries.'''
    prompt = [SystemMessage(content=system_message)]
    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.QueryBot(state["query"])
    show_md_file(path)
    return {"response": path}

def tutorial_agent(state):
    """Handles tutorial creation queries."""
    system_message = '''You are a knowledgeable assistant specializing as a Senior Generative AI Developer and experienced blogger.
         Your task is to create a high-quality tutorial blog in .md format with coding examples, clear explanations, and resource references.'''
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.TutorialAgent(state["query"])
    show_md_file(path)
    return {"response": path}

def interview_topics_questions(state):
    """Handles interview topic queries."""
    system_message = '''You are a researcher skilled in gathering interview questions for Generative AI roles.
                     Provide a curated list of top interview questions (with references/links if possible) in a .md document.'''
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Interview_questions(state["query"])
    show_md_file(path)
    return {"response": path}

def mock_interview(state):
    """Handles mock interview sessions."""
    system_message = '''You are a Generative AI Interviewer. Conduct a mock interview session for a Generative AI role.
         Engage in a back-and-forth conversation and provide a final evaluation.'''
    prompt = [SystemMessage(content=system_message)]
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Mock_Interview()
    show_md_file(path)
    return {"response": path}

def route_query(state):
    """Routes the query based on its primary category."""
    if '1' in state["category"]:
        # print('Routing to learning resource handler.')  # Commented out
        return "handle_learning_resource"
    elif '2' in state["category"]:
        # print('Routing to resume maker handler.')  # Commented out
        return "handle_resume_making"
    elif '3' in state["category"]:
        # print('Routing to interview preparation handler.')  # Commented out
        return "handle_interview_preparation"
    elif '4' in state["category"]:
        # print('Routing to job search handler.')  # Commented out
        return "job_search"
    else:
        # print("Query does not match any category.")  # Commented out
        return False

def route_interview(state) -> str:
    """Routes interview queries to either mock interview or topic questions."""
    if 'question'.lower() in state["category"].lower():
        # print('Routing to interview topics/questions handler.')  # Commented out
        return "interview_topics_questions"
    elif 'mock'.lower() in state["category"].lower():
        # print('Routing to mock interview handler.')  # Commented out
        return "mock_interview"
    else:
        # print('Defaulting to mock interview.')  # Commented out
        return "mock_interview"

def route_learning(state):
    """Routes learning queries to either tutorial creation or Q&A."""
    if 'question'.lower() in state["category"].lower():
        # print('Routing to Q&A handler.')  # Commented out
        return "ask_query_bot"
    elif 'tutorial'.lower() in state["category"].lower():
        # print('Routing to tutorial handler.')  # Commented out
        return "tutorial_agent"
    else:
        # print("No clear learning sub-route found.")  # Commented out
        return False