from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils import save_file
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from your .env file

class ResumeMaker:
    def __init__(self, prompt):
        
        self.model = ChatGroq(model="llama-3.3-70b-versatile")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def Create_Resume(self, user_input):
        chat_history = []
        response = self.agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=user_input), response["output"]])
        path = save_file(str(response.get('output')).replace("\nmarkdown", "").strip(), 'Resume')
        return path