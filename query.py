import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

PERSIST_DIRECTORY = "db"

def main():
    print("Initializing embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    print("Retriever ready.")

    print("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model='models/gemini-flash-latest', temperature=0.7, google_api_key=os.getenv("GEMINI_API_KEY"))
    print("LLM ready.")

    prompt_template = """Use the following context to answer the question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.

Context:
{context}

Question: {input}

Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("RAG chain created.")

    query = "What are the SOFTWARE REQUIREMENTS SPECIFICATION mentioned in the documents?"
    print(f"\nExecuting query: '{query}'")
    
    result = qa_chain.invoke({"input": query})
    
    print("\n--- Answer ---")
    print(result["answer"])
    print("---------------")

if __name__ == "__main__":
    main()
