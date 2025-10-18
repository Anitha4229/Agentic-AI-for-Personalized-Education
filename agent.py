import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

load_dotenv()

PERSIST_DIRECTORY = 'db'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = vectordb.as_retriever()

llm = ChatGoogleGenerativeAI(model='models/gemini-flash-latest', temperature=0.7, google_api_key=os.getenv("GEMINI_API_KEY"))  # Add your Gemini API Key here

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

tools = [
    Tool(
        name="Course Material Q&A System",
        func=qa_chain.run,
        description="Use this to answer questions about course content, definitions, and concepts from the provided documents. Input must be a clear question."
    )
]

prompt_template = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- NEW AGENT EXECUTION LOGIC ---

# Define the prompt for the agent
student_profile = "I am a student who learns best with examples and struggles with theoretical concepts."
goal = "I have an exam on 'Introduction to AI' in one week. My weakest topics are 'Neural Networks' and 'Reinforcement Learning'."
prompt_input = f"""
Based on the provided course materials, create a 3-day personalized study plan for me.
My student profile is: {student_profile}.
My goal is: {goal}.
The plan should break down each day's tasks, suggest specific topics to review from the documents, 
and formulate two practice questions for each topic.
"""

# Run the agent using .invoke()
print("Agent is thinking...")
response = agent_executor.invoke({"input": prompt_input})

# The final answer is in the 'output' key of the response dictionary
print("\n--- Your Personalized Study Plan ---")
print(response['output'])