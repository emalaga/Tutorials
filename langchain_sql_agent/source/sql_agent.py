from dotenv import load_dotenv
import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from IPython.display import display, Markdown #to render results

env_path = "../.env"

def render(text):
    display(Markdown(text))

class SQL_Agent:

    def __init__(self, model_provider = None, model_name=None, dialect=None, database_uri=None, top_k = 5):
        
        self.llm = None
        if model_provider and model_name:
            self.set_llm(model_provider, model_name)

        self.db = None
        if database_uri:
            self.set_db(database_uri)

        self.tools = None
        self.prompt_template = None
        self.system_message = None
        self.agent_executor = None
        if self.db and self.llm:
            self.set_tools()
            if dialect:
                self.set_prompts(dialect, top_k)
                if self.system_message:
                    self.initialize_agent()
        
    def set_llm(self, model_provider, model_name):
        load_dotenv(env_path)
        self.llm = init_chat_model(model_name, model_provider=model_provider)

    def set_db(self,database_uri):
        self.db = SQLDatabase.from_uri(database_uri, view_support = True)

    def set_tools(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = toolkit.get_tools()

    def set_prompts(self, dialect, top_k):
        self.prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        assert len(self.prompt_template.messages) == 1
        self.system_message = self.prompt_template.format(dialect=dialect, top_k=top_k)

    def set_custom_system_message(self, custom_system_message):
        self.system_message = custom_system_message

    def initialize_agent(self):
        self.agent_executor = create_react_agent(self.llm, self.tools, prompt=self.system_message)

    def answer(self,question):
        response = self.agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
        return(response["messages"][-1].content)
    
    def answer_full(self,question):
        response = self.agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
        return(response)
    
    def stream_answer(self,question):
        last_message = None

        for step in self.agent_executor.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            last_message.pretty_print()

        return last_message.content if last_message else None
