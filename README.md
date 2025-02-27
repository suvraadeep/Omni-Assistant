# Omni-Assistant
Your All time Career Assistant

## Overview

Meet the **OMNI Career Assistant**—an AI-powered mentor designed to simplify and support your journey in your learning, resume preparation, interview assistance, and job hunting. This assistant leverages advanced language models such as deepseek and open-source tools to provide personalized guidance and resources, making your transition into the rapidly evolving field of Generative AI smoother and more efficient.

## Directory Structure

```
OMNI/
├── agents/
│   ├── __init__.py
│   ├── interview_agent.py
│   ├── job_search_agent.py
│   ├── learning_resource_agent.py
│   └── resume_maker_agent.py
├── router.py
├── utils.py
├── workflow.py
├── app.py
├── requirements.txt
└── README.md
```

## Tech Stack

- **LangChain** – For building and orchestrating AI chains.
- **LangGraph** – For designing a flexible and extensible workflow graph.
- **GROQ Deekseek LLM** – A powerful open-source language model.
- **DuckDuckGoSearchResult** – For retrieving up-to-date web information.
- **Python, Streamlit** – For building an interactive, web-based chatbot interface.

## Motivation

With GenAI rapidly evolving and more people are eager to learn and transition into careers in different fields. However, navigating the vast and often outdated resources available online can be overwhelming. Long videos, scattered materials, and deprecated code examples make it challenging to start learning and applying GenAI concepts effectively. The GenAI Career Assistant is designed to cut through the noise by:
- Organizing personalized learning pathways.
- Offering tailored resume-building and interview preparation.
- Providing curated job search assistance.
- Delivering reliable, up-to-date information and resources.

## Key Features

- **Learning & Content Creation:**  
  - Offers tailored learning pathways covering key topics and skills in Generative AI.
  - Assists in creating tutorials, blogs, and posts based on your interests.
- **Q&A Support:**  
  - Provides on-demand Q&A sessions for detailed explanations and coding support.
- **Resume Building & Review:**  
  - Offers one-on-one resume consultations and guidance.
  - Crafts personalized, market-relevant resumes optimized for current job trends.
- **Interview Preparation:**  
  - Hosts Q&A sessions on common and technical interview questions.
  - Simulates real interview scenarios with mock interviews.
- **Job Search Assistance:**  
  - Guides you through the job search process, offering tailored insights and support.

## Key Components

- **State Management:**  
  - Uses `TypedDict` to define and manage the state of each customer interaction, ensuring a consistent data structure for query, category, and response.
- **Query Categorization:**  
  - Classifies user queries into four main categories: Learning Generative AI, Resume Making, Interview Preparation, or Job Search.
- **Sub-Categorization:**  
  - For learning queries: further categorizes into Tutorial or Q&A.
  - For interview queries: differentiates between Interview Preparation and Mock Interviews.
- **Response Generation:**  
  - Generates responses based on the query category and sub-category, including creating Markdown files for tutorials, resumes, mock interviews, etc.
- **Workflow Graph:**  
  - Utilizes LangGraph to create a flexible, extensible workflow that manages state and conditional routing for each interaction.

## Method Details

1. **Initialization:**  
   - Set up the environment, load required libraries, and initialize models and tools.
2. **State Definition:**  
   - Define a `State` class using `TypedDict` to specify the structure of each interaction:
     - `query`: User input or question.
     - `category`: Category/type of query.
     - `response`: The generated expert response.
3. **Node Functions:**  
   - Implement separate functions for query categorization and response generation.
4. **Graph Construction:**  
   - Use `StateGraph` to build the workflow by adding nodes and edges representing different stages of support.
5. **Conditional Routing:**  
   - Implement logic to route queries based on their category and sub-category.
6. **Workflow Compilation:**  
   - Compile the workflow graph into an executable application.
7. **Execution:**  
   - Process user queries through the workflow and retrieve the final result.

## Agent Details

### LearningResourceAgent

- **Initialization:**  
  Initializes the chat model (using Gemini LLM), prompt, and tools (like DuckDuckGo search).
- **TutorialAgent Method:**  
  Generates a tutorial by invoking the model and saves the output as a Markdown file.
- **QueryBot Method:**  
  Conducts a Q&A session with the user, trimming the conversation as it grows to ensure up-to-date responses.

### InterviewAgent

- **Initialization:**  
  Initializes the chat model (using Gemini LLM), prompt, and search tools.
- **Interview_questions Method:**  
  Handles interview question preparation, collecting responses in a session that is eventually saved as a Markdown file.
- **Mock_Interview Method:**  
  Simulates a mock interview session, displaying both interviewer and candidate messages.

### ResumeMaker

- **Initialization:**  
  Sets up the chat model, prompt, and search tools for creating a resume.
- **Create_Resume Method:**  
  Engages in a conversation with the user to generate a tailored resume, saving the output as a Markdown file.

### JobSearch

- **Initialization:**  
  Configures the chat model and search tools to assist with job search queries.
- **find_jobs Method:**  
  Retrieves and formats job search results into a clear Markdown summary.

## Query Routing Functions

- **categorize:**  
  Classifies the initial user query into one of the four main categories.
- **handle_learning_resource:**  
  Determines if a learning-related query is for creating tutorials or for Q&A.
- **handle_interview_preparation:**  
  Distinguishes between mock interviews and interview question preparation.
- **job_search:**  
  Processes job search queries.
- **handle_resume_making:**  
  Manages resume creation queries.
- **ask_query_bot & tutorial_agent:**  
  Provide detailed Q&A sessions and comprehensive tutorial generation.
- **interview_topics_questions & mock_interview:**  
  Generate interview questions or simulate a full mock interview session.

## Workflow Graph

The workflow graph is constructed using LangGraph, where each node represents a step in the support process. Conditional routing ensures that user queries are processed by the appropriate agent based on their category and sub-category.

![langchain agents](https://github.com/user-attachments/assets/97122bf8-3e85-4354-b226-570f8d329ebf)

## Conclusion

The **OMNI Career Assistant** is more than just a tool; it's a comprehensive, personalized mentor designed to help you thrive in the rapidly evolving field of every domain. Whether you need to master key concepts, build a strong resume, prepare for interviews, or navigate the job market, this assistant equips you with the resources and guidance to achieve your career goals. With the GenAI Career Assistant by your side, your path to a successful career in Generative AI becomes clearer, more manageable, and achievable.

## Future Enhancements

- **Knowledge Base:**  
  Integrate a resource-rich library with curated links to courses, tutorials, and articles.
- **Multi-Domain Customization:**  
  Expand the assistant’s capabilities to support various career paths beyond Generative AI.
- **Advanced Job Search Tools:**  
  Implement features such as automated job application tracking, networking guidance, and global job search support.

## Setup and Run Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` File:**
   In the root directory, create a file named `.env` with the following content (replace with your actual API key):
   ```dotenv
   GROQ_API_KEY=your_actual_api_key_here
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```





