import streamlit as st  
import os
import json
import torch
import fitz  
import easyocr  
from pdf2image import convert_from_path  
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
import io
from dotenv import load_dotenv

# Set page config
st.set_page_config(page_title="Chat with LEXIBOT", page_icon="📝", layout="centered")

# Cyberpunk Header & CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    @keyframes colorChange {
        0% { background: #007BFF; }
        25% { background: #00C6FF; }
        50% { background: #6610f2; }
        75% { background: #ff0080; }
        100% { background: #007BFF; }
    }

    .header {
        animation: colorChange 5s infinite alternate;
        padding: 10px 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: white;
        border-radius: 8px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
        letter-spacing: 2px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        width: fit-content;
        margin: auto;
    }

    .glow-text {
        text-shadow: 0px 0px 6px rgba(255, 255, 255, 0.8), 
                     0px 0px 12px rgba(0, 198, 255, 0.6);
    }

    .cyber-button {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        border: none;
        color: white;
        padding: 12px 24px;
        font-size: 16px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
        box-shadow: 0px 0px 10px rgba(0, 198, 255, 0.8), 
                    0px 0px 20px rgba(255, 0, 255, 0.6);
    }

    .cyber-button:hover {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        box-shadow: 0px 0px 15px rgba(255, 0, 255, 1), 
                    0px 0px 25px rgba(0, 198, 255, 1);
    }
    </style>

    <div class="header">
        🤖 <span class="glow-text">LEXIBOT</span> | AI Chatbot 🚀
    </div>
    """,
    unsafe_allow_html=True
)

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load API Key
def load_groq_api_key():
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        st.error("🚨 config.json not found. Please add your GROQ API key.")
        st.stop()

groq_api_key = load_groq_api_key()
if not groq_api_key:
    st.error("🚨 GROQ_API_KEY is missing. Check your config.json file.")
    st.stop()

# Initialize OCR with GPU
reader = easyocr.Reader(["en"], gpu=True)

# PDF Text Extraction
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
    doc.close()
    return text_list if text_list else extract_text_from_images(file_path)

def extract_text_from_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
    return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]

# Vector Store Setup
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    return FAISS.from_texts(doc_chunks, embeddings)

# Chat Chain Creation
def create_chain(vectorstore):
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=st.session_state.memory,
        verbose=False
    )

# Streamlit UI
st.title("🦙 Chat with LEXIBOT - LLAMA 3.3 (GPU Accelerated)")

# File Upload Section
uploaded_files = st.file_uploader("🚀 Upload Cyber Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_extracted_text = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        extracted_text = extract_text_from_pdf(file_path)
        all_extracted_text.extend(extracted_text)
    
    st.session_state.vectorstore = setup_vectorstore(all_extracted_text)
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Chat Input
user_input = st.chat_input("💬 Ask LEXIBOT...")

if user_input:
    response = st.session_state.conversation_chain.invoke({"question": user_input})
    st.chat_message("assistant").markdown(response.get("answer", "Error processing."))

# Cyberpunk Export Chat Feature
st.subheader("⚡ Export Chat History")

export_format = st.radio("Choose format:", ["Text (.txt)", "JSON (.json)"], horizontal=True)

def export_chat():
    return "\n".join(f"{msg['type'].capitalize()}: {msg['content']}" for msg in st.session_state.memory.chat_memory.messages)

if st.button("⚡ Export Chat", key="export_chat", help="Download chat history"):
    chat_content = export_chat()
    mime_type = "text/plain" if export_format == "Text (.txt)" else "application/json"
    file_name = "chat_history.txt" if export_format == "Text (.txt)" else "chat_history.json"
    st.download_button("⚡ DOWNLOAD", chat_content, file_name, mime_type, key="download_chat")
