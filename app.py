import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from multiprocessing import Pool, cpu_count
import glob

# --- Backend Logic (Functions) ---

def load_and_parse_file(file_path):
    try:
        loader = GenericLoader.from_filesystem(path=file_path, parser=LanguageParser())
        return loader.load()
    except Exception as e:
        print(f"Skipping file {file_path} due to error: {e}")
        return []

@st.cache_resource
def create_hybrid_retriever(_path, provider, _api_key=None):
    if not os.path.isdir(_path):
        st.error("The provided path is not a valid directory.")
        return None
    try:
        ensemble_retriever = None
        # Create an expander to show the progress
        with st.expander(f"Indexing codebase at `{_path}`...", expanded=True):
            with st.spinner("Processing..."):
                st.write("Discovering Python files...")
                all_files = [f for f in glob.glob(f"{_path}/**/*.py", recursive=True)]
                if not all_files:
                    st.warning("No Python (.py) files found.")
                    return None
                
                st.write(f"Loading {len(all_files)} files in parallel...")
                with Pool(processes=cpu_count()) as pool:
                    results = pool.map(load_and_parse_file, all_files)
                docs = [doc for sublist in results for doc in sublist]

                st.write("Splitting documents and creating search indexes...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                texts = text_splitter.split_documents(docs)

                bm25_retriever = BM25Retriever.from_documents(texts)
                bm25_retriever.k = 5

                if provider == "Gemini API":
                    if not _api_key:
                        st.error("Google API Key is required for Gemini.")
                        return None
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
                else: # Ollama
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                
                vector_store = Chroma.from_documents(texts, embedding=embeddings)
                semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, semantic_retriever],
                    weights=[0.5, 0.5]
                )
        st.success("Indexing Complete!")
        return ensemble_retriever
    except Exception as e:
        st.error(f"Failed to process codebase: {e}")
        return None

# --- UI Configuration ---
st.set_page_config(page_title="Pro Codebase Assistant", page_icon="🚀", layout="wide")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- Sidebar ---
with st.sidebar:
    st.image("https://ollama.com/public/ollama.png", width=100)
    st.header("Codebase Assistant")
    st.markdown("---")
    st.header("1. Choose Model")
    model_provider = st.radio("Model Provider:", ("Ollama (Local)", "Gemini API"), label_visibility="collapsed")
    
    google_api_key = None
    if model_provider == "Gemini API":
        google_api_key = st.text_input("Google API Key", type="password", key="api_key", placeholder="Enter your key here")

    st.markdown("---")
    st.header("2. Index Code")
    codebase_path = st.text_input("Local codebase path", key="codebase_path", placeholder="e.g., ./my_project")
    
    if st.button("Index Codebase", use_container_width=True):
        if codebase_path:
            st.session_state.retriever = create_hybrid_retriever(codebase_path, model_provider, google_api_key)
            st.session_state.chat_history = []
        else:
            st.warning("Please enter a codebase path.")

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- Main Chat Interface ---
st.header("Chat with your Code 💬")

# Welcome message
if not st.session_state.chat_history and st.session_state.retriever is None:
    st.info("Welcome! Please index a codebase using the sidebar to get started.")

# Display previous messages
for message in st.session_state.chat_history:
    avatar = "🧑‍💻" if isinstance(message, HumanMessage) else "🤖"
    with st.chat_message(message.type, avatar=avatar):
        st.markdown(message.content)

# Handle new user input
if user_question := st.chat_input("Ask a question..."):
    if st.session_state.retriever is not None:
        st.chat_message("human", avatar="🧑‍💻").markdown(user_question)

        with st.chat_message("assistant", avatar="🤖"):
            llm = ChatOllama(model="llama3:8b", temperature=0.1) if model_provider == "Ollama (Local)" else \
                  ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.1)
            
            # Create the full RAG chain
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert programmer. Use the following pieces of retrieved context to answer the question. Be concise and provide code snippets when relevant.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Stream the response
            def stream_response():
                for chunk in rag_chain.stream({"input": user_question, "chat_history": st.session_state.chat_history}):
                    if "answer" in chunk:
                        yield chunk["answer"]
            
            full_response = st.write_stream(stream_response)
            
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=full_response))
    else:
        st.warning("Please index a codebase first to start the conversation.")