from dotenv import load_dotenv
load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


import pandas as pd
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import create_csv_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.agents import initialize_agent


from langchain.agents import ZeroShotAgent


def create_everything(df1, df2):
    #### PANDAS AGENT ####
    df1 = pd.read_csv(df1)
    pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df1, verbose=True)
    
    ###### CSV AGENT #####
    csv_agent = create_csv_agent(OpenAI(temperature=0), df2 , verbose=True)



    # Custom Tools (agents turned to tools here)
    class CustomPandasTool(BaseTool):
        name = "pd agent"
        description="will use the pd agent to access the csv files and act as a pd agent"

        def _run(self, query):
            """Use the tool."""
            return pd_agent.run(query)
        
        async def _arun(self, query):
            """Use the tool asynchronously."""
            raise NotImplementedError("Pandas Tool does not support async")

    class CustomCsvAgent(BaseTool):
        name = "csv agent"
        description = "will use csv agent to answer questions about the csv file given"

        def _run(self, query):
            """Use the tool."""

            return csv_agent.run(query)
        
        async def _arun(self, query):
            """Use the tool asynchronously."""
            raise NotImplementedError("CSV Tool does not support async")


    tools = [CustomPandasTool(), CustomCsvAgent()]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature = 0)
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True, max_iterations=3)
    return agent

def run_chatbot():
    st.title("CSV AI bot")

    st.write("Please upload two CSV files below. One will be analysed by pandas agent and another by a csv agent")

    data = st.file_uploader("Upload a CSV")
    data2 = st.file_uploader("Upload another CSV")
    user_input = st.text_area("Ask your question:")
    if st.button("Submit Query", type="primary"):
        agent = create_everything(data, data2)
        answer = agent.run(user_input)
        st.write(answer.__str__())

run_chatbot()
