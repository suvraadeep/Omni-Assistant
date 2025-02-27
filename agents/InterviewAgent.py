from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from utils import save_file
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import trim_conversation

from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from your .env file


class InterviewAgent:
    def __init__(self, prompt):
        self.model = ChatGroq(model="llama-3.3-70b-versatile")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def Interview_questions(self, user_input):
        chat_history = []
        questions_bank = ''
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        response = self.agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        questions_bank += str(response.get('output')).replace("\nmarkdown", "").strip() + "\n"
        chat_history.extend([HumanMessage(content=user_input), response["output"]])
        path = save_file(questions_bank, 'Interview_questions')
        return path

    def Mock_Interview(self):
        # A simplified mock interview: single turn for demonstration
        initial_message = 'I am ready for the interview.\n'
        self.prompt.append(HumanMessage(content=initial_message))
        response = self.model.invoke(self.prompt)
        self.prompt.append(AIMessage(content=response.content))
        path = save_file(response.content, 'Mock_Interview')
        return path