# agents/job_search_agent.py
from langchain_community.tools import DuckDuckGoSearchResults
from utils import save_file
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from utils import save_file
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from your .env file

class JobSearch:
    def __init__(self, prompt):
        self.model = ChatGroq(model="llama-3.3-70b-versatile")
        self.prompt = prompt
        self.tools = DuckDuckGoSearchResults()

    def find_jobs(self, user_input):
        results = self.tools.invoke(user_input)
        chain = self.prompt | self.model
        jobs = chain.invoke({"result": results}).content
        path = save_file(str(jobs).replace("\nmarkdown", "").strip(), 'Job_search')
        return path