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

# --- Load environment variables ---
load_dotenv()
PERSIST_DIRECTORY = 'db'

# --- Utility: Secure API key retrieval ---
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]  # for Render
    except KeyError:
        return os.getenv("GEMINI_API_KEY")  # for local use

# --- Cached resource setup: only loads once per session ---
@st.cache_resource
def setup_agent():
    """Set up the AI Agent with minimal memory usage."""
    print("Initializing AI Agent...")

    # Check if DB exists
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error("Knowledge Base ('db' folder) not found in the repo.")
        st.stop()

    # Use lightweight embedding model to save memory
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    # Load only top results to reduce memory footprint
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Fetch Gemini API key
    api_key = get_api_key()
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add it to Render secrets or .env.")
        st.stop()

    # Initialize lightweight Google Gemini model
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.7,
        google_api_key=api_key
    )

    # Define tool for document-based Q&A
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    tools = [
        Tool(
            name="Course Material Q&A System",
            func=qa_chain.run,
            description="Use this to answer questions within course documents."
        )
    ]

    # Create agent with prompt template
    prompt_template = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Check your output and try again."
    )

    print("Agent successfully initialized.")
    return agent_executor


# --- Streamlit UI ---
st.title("🎓 Personalized Learning Path Generator")

try:
    # Load the agent once and store it in session_state
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = setup_agent()

    st.success("✅ AI Agent is ready to help you personalize your learning path.")
    st.markdown("This agent uses your course materials to create a custom study plan based on your learning style and goals.")

    # --- User Input Form ---
    with st.form("study_plan_form"):
        learning_style = st.text_input("🧠 Describe your learning style", "I'm a visual learner who learns best with examples.")
        exam_topic = st.text_input("📘 Exam topic or subject", "Introduction to AI")
        time_frame = st.text_input("⏰ Time until exam", "3 Days")
        weak_topics = st.text_area("📉 List your weakest topics (one per line)", "Neural Networks\nReinforcement Learning")

        submitted = st.form_submit_button("✨ Generate My Study Plan")

        if submitted:
            if not learning_style or not exam_topic or not weak_topics:
                st.error("Please fill in all fields before generating a plan.")
            else:
                prompt = f"""
                Create a personalized study plan for an exam on '{exam_topic}' which is in {time_frame}.
                My learning style is: '{learning_style}'.
                My weakest topics are: {weak_topics}.
                Using the 'Course Material Q&A System' tool, create a detailed markdown plan.
                For each weak topic, identify core concepts from the documents and include two simple, practical questions.
                """
                with st.spinner("🤖 Crafting your personalized plan..."):
                    try:
                        response = st.session_state.agent_executor.invoke({"input": prompt})
                        st.markdown(response['output'])
                    except Exception as e:
                        st.error(f"An error occurred while generating your plan: {e}")

except Exception as e:
    st.error(f"An error occurred during setup: {e}")
