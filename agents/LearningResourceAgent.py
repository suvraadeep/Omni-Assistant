from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils import save_file
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import trim_conversation
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from your .env file

class LearningResourceAgent:
    def __init__(self, prompt):
        
        self.model = ChatGroq(model="llama-3.3-70b-versatile")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def TutorialAgent(self, user_input):
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        response = agent_executor.invoke({"input": user_input})
        path = save_file(str(response.get('output')).replace("\nmarkdown", "").strip(), 'Tutorial')
        return path

    def QueryBot(self, user_input):
        record_QA_session = []
        record_QA_session.append('User Query: %s \n' % user_input)
        self.prompt.append(HumanMessage(content=user_input))
        response = self.model.invoke(self.prompt)
        record_QA_session.append('\nExpert Response: %s \n' % response.content)
        self.prompt.append(AIMessage(content=response.content))
        path = save_file(''.join(record_QA_session), 'Q&A_Doubt_Session')
        return path