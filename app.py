import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

# --- IMPORTANT: Import your ingest script ---
import ingest 

# Load environment variables for local development
load_dotenv()

# Define the path for the database
PERSIST_DIRECTORY = 'db'

# Function to get API key from either environment or Streamlit secrets
def get_api_key():
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        # Fall back to environment variable (for local development)
        return os.getenv("GEMINI_API_KEY")

@st.cache_resource
def setup_agent():
    """
    Sets up the LangChain agent and necessary components.
    This function assumes the database in PERSIST_DIRECTORY already exists.
    """
    print("Setting up agent...")
    
    # Setup embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    # Get the API key securely
    api_key = get_api_key()
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your environment or Streamlit secrets.")
        st.stop()
    
    # Setup the language model
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.7,
        google_api_key=api_key
    )

    # Create the QA chain and tools
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    tools = [
        Tool(
            name="Course Material Q&A System",
            func=qa_chain.run,
            description="Use this to answer questions and find information within the course documents."
        )
    ]
    prompt_template = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors="Check your output and try again.")

    print("Agent setup complete.")
    return agent_executor

# --- Main Application Logic ---

st.title("🎓 Personalized Learning Path Generator")

# Check if the knowledge base exists. If not, build it.
if not os.path.exists(PERSIST_DIRECTORY):
    st.info("First-time setup: Knowledge Base not found. Building it now...")
    st.write("This may take a minute. Please wait.")
    
    with st.spinner("Processing documents and creating vector store..."):
        try:
            # --- This calls your ingest.py script ---
            ingest.ingest_data() # IMPORTANT: Assumes your ingest script has a main() function
            st.success("Knowledge Base built successfully!")
            st.write("The app will now reload to use the new Knowledge Base.")
            # Reload the app to reflect the changes
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error during Knowledge Base creation: {e}")
            st.stop()

# If the knowledge base exists, run the main app
else:
    try:
        agent_executor = setup_agent()
        st.write("This AI agent will create a custom study plan based on your course materials and learning goals.")

        # --- User Input Form ---
        with st.form("study_plan_form"):
            learning_style = st.text_input("Describe your learning style (e.g., 'I'm a visual learner, I need examples')")
            exam_topic = st.text_input("What is the exam topic or subject?", "Introduction to AI")
            time_frame = st.text_input("How long until your exam? (e.g., '3 Days', '1 Week')", "3 Days")
            weak_topics = st.text_area("List your weakest topics (one per line)", "Neural Networks\nReinforcement Learning")
            
            submitted = st.form_submit_button("Generate My Study Plan")

            if submitted:
                if not learning_style or not exam_topic or not weak_topics:
                    st.error("Please fill in all the fields before generating a plan.")
                else:
                    prompt = f"""
                    Create a personalized study plan for an exam on '{exam_topic}' which is in {time_frame}.
                    My learning style is: '{learning_style}'.
                    My weakest topics are: {weak_topics}.
                    Using the 'Course Material Q&A System' tool, please structure a detailed plan. For each weak topic,
                    identify the core concepts from the documents and create two simple, practical questions to test my understanding.
                    The final output should be a well-formatted markdown plan.
                    """
                    with st.spinner("The AI agent is crafting your personalized plan..."):
                        try:
                            response = agent_executor.invoke({"input": prompt})
                            st.markdown(response['output'])
                        except Exception as e:
                            st.error(f"An error occurred while generating the plan: {e}")

    except Exception as e:
        st.error(f"An error occurred during agent setup: {e}")