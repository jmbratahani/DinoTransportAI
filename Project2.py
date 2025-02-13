## DSO 599 Project 2
# Group 9: Xiaoyi Guan, Jianan Ding, Jessica Bratahani
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
#from langchain_community.embeddings import BedrockEmbeddings
#from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent, load_tools, create_openai_tools_agent
from langchain import hub
#from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import sqlite3
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder



def initial():
    try:
        # Set up LangChain components
        llm = ChatOpenAI(temperature=0)
        pdf_folder = 'pdfs/'
        loader = PyPDFDirectoryLoader(pdf_folder, recursive=True)
        syllabus_kb = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(syllabus_kb)
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents=splits, embedding=embedding_function)
        retriever = vector_store.as_retriever()

        # Create Retriever Tool
        tool = create_retriever_tool(retriever, "search_dino_docs", "Searches and returns excerpts from the dino pdfs.")
        
        # Create Weather Tool
        os.environ["OPENWEATHERMAP_API_KEY"] = "c7417b485650c1830aefb79ae8f38319"
        openwm = load_tools(["openweathermap-api"], llm)

        # Create AWS Lambda Function Tool for sending email with SES
        aws_ses = load_tools(["awslambda"],awslambda_tool_name="email-sender", 
                           awslambda_tool_description="sends an email with the specified content to email",
                           function_name="sendSES")
        
        # SQLite Setup
        #setup_sqlite()
        db = SQLDatabase.from_uri("sqlite:///dino_database.db")

        sqltoolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
        context = sqltoolkit.get_context()
        sqltools = sqltoolkit.get_tools()
        final_toolkit = [tool] + openwm + aws_ses + sqltools # Add OpenWeatherMap tool to the list of tools

        prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
                        ),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ]
                )
        # Construct the OpenAI Tools agent
        agent = create_openai_tools_agent(llm, final_toolkit, prompt)

        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=final_toolkit, verbose=True)
        return agent_executor
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def setup_sqlite():
    conn = sqlite3.connect('dino_database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS DinoMap (ID TEXT PRIMARY KEY, Name TEXT)''')
    cursor.executemany('INSERT INTO DinoMap (ID, Name) VALUES (?, ?)', [('T88', 'T-Rex'), ('V66', 'Velociraptor')])
    conn.commit()
    conn.close()

def main():
    st.title("Hi, I am DinoTransport Copilot")
    user_input = st.text_input("Ask a question about dinosaur transport safety:")
    agent_executor = initial()
    if user_input:
        if "session_id" not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
        session_id = st.session_state.session_id  # Use Streamlit session ID
        # Implement chat message history
        memory = ChatMessageHistory(session_id=session_id)
        agent_with_chat_history = RunnableWithMessageHistory(agent_executor,
                                                             lambda session_id: memory,
                                                             input_messages_key="input",
                                                             history_messages_key="chat_history")
        try:
            result = agent_with_chat_history.invoke({"input": user_input},
                                                     config={"configurable": {"session_id": session_id}})
            # Check if the output is a string
            if isinstance(result["output"], str):
                st.write(result["output"])
            else:
                st.error("Error: Unable to parse LangChain response.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Hello, How can I help you?")

if __name__ == "__main__":
    main()
