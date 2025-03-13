import streamlit as st  
import os
import json
import torch
import fitz  # PyMuPDF
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

st.set_page_config(page_title="Chat with LEXIBOT 🚀", page_icon="🤖", layout="wide")

# Enhanced futuristic styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    /* Animated Cyberpunk Background */
    body {
        background: linear-gradient(45deg, #0f0c29, #302b63, #24243e);
        animation: gradientAnimation 10s infinite alternate;
        font-family: 'Orbitron', sans-serif;
        color: #00ffee;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* Glowing Header */
    .header {
        animation: neonGlow 1.5s infinite alternate;
        font-size: 24px;
        text-transform: uppercase;
        text-align: center;
        letter-spacing: 2px;
        color: #00ffee;
        font-weight: bold;
        text-shadow: 0 0 5px #00ffee, 0 0 10px #00ffee, 0 0 20px #00ffee;
    }

    @keyframes neonGlow {
        0% { text-shadow: 0 0 5px #00ffee, 0 0 10px #00ffee; }
        100% { text-shadow: 0 0 10px #00ffee, 0 0 20px #00ffee; }
    }

    /* Glitch Animation */
    .glitch {
        position: relative;
        animation: glitchEffect 1s infinite alternate;
    }

    @keyframes glitchEffect {
        0% { text-shadow: -2px 0 cyan, 2px 0 magenta; }
        100% { text-shadow: 2px 0 cyan, -2px 0 magenta; }
    }

    /* 3D Hover Buttons */
    .stButton>button {
        background: linear-gradient(145deg, #00ffee, #007BFF);
        color: white;
        font-size: 16px;
        font-family: 'Orbitron', sans-serif;
        border-radius: 10px;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 5px 15px rgba(0, 255, 238, 0.8);
    }

    /* Input Fields */
    .stTextInput>div>div>input {
        background: rgba(0, 0, 0, 0.6);
        border: 2px solid cyan;
        color: #00ffee;
        font-family: 'VT323', monospace;
        font-size: 18px;
    }

    /* Chat Messages */
    .stChatMessage {
        font-family: 'VT323', monospace;
        font-size: 20px;
        padding: 10px;
        border-radius: 8px;
    }

    .user {
        color: #ff80ff;
    }

    .assistant {
        color: #00ffee;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header glitch">🤖 LEXIBOT | AI Chatbot 🚀</div>', unsafe_allow_html=True)

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

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

# Initialize EasyOCR with GPU support
reader = easyocr.Reader(["en"], gpu=True)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
    doc.close()
    return text_list if text_list else extract_text_from_images(file_path)

def extract_text_from_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
    return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    return FAISS.from_texts(doc_chunks, embeddings)

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

st.title("🦙 Chat with LEXIBOT - LLAMA 3.3 (GPU Accelerated)")

uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_extracted_text = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        extracted_text = extract_text_from_pdf(file_path)
        all_extracted_text.extend(extracted_text)
        st.success(f"✅ Extracted text from {uploaded_file.name}")
    
    if all_extracted_text:
        st.session_state.vectorstore = setup_vectorstore(all_extracted_text)
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

user_input = st.chat_input("Ask Llama...")

if user_input:
    response = st.session_state.conversation_chain.invoke({"question": user_input, "chat_history": st.session_state.memory.chat_memory.messages})
    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(response.get("answer", "I'm sorry, I couldn't process that."))
