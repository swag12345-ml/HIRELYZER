import streamlit as st  # Streamlit must be imported first

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Chat with LEXIBOT", page_icon="📝", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    /* Cyberpunk Background */
    body {
        background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
        color: #d1d5db;
        font-family: 'Orbitron', sans-serif;
    }

    /* Glowing Header */
    @keyframes colorPulse {
        0% { background: #007BFF; }
        50% { background: #ff0080; }
        100% { background: #007BFF; }
    }

    @keyframes floatEffect {
        0% { transform: translateX(0px); }
        50% { transform: translateX(5px); }
        100% { transform: translateX(0px); }
    }

    .header {
        animation: colorPulse 4s infinite alternate, floatEffect 3s infinite ease-in-out;
        padding: 8px 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 198, 255, 0.7);
        margin-bottom: 15px;
        letter-spacing: 2px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        width: fit-content;
        margin: auto;
    }

    .glow-text {
        text-shadow: 0px 0px 8px rgba(255, 255, 255, 0.9), 
                     0px 0px 15px rgba(0, 198, 255, 0.8);
    }

    /* Neon Cyberpunk Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff0080, #007BFF);
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        text-transform: uppercase;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0px 0px 15px rgba(0, 198, 255, 0.7);
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #007BFF, #ff0080);
        box-shadow: 0px 0px 20px rgba(255, 0, 128, 0.8);
        transform: scale(1.05);
    }

    /* Cyberpunk Input Field */
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: #d1d5db;
        border: 2px solid #ff0080;
        font-size: 16px;
        font-family: 'Orbitron', sans-serif;
        padding: 10px;
    }

    /* Cyberpunk Chat Messages */
    .stChatMessage {
        background: #1e293b;
        border: 2px solid #007BFF;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0px 0px 10px rgba(0, 198, 255, 0.5);
    }
    </style>

    <div class="header">
        🤖 <span class="glow-text">LEXIBOT</span> | POWERED BY SEMICOLON 🚀
    </div>
    """,
    unsafe_allow_html=True
)

import os
import json
import torch
import fitz  # PyMuPDF for text extraction
import easyocr  # GPU-accelerated OCR
from pdf2image import convert_from_path  # Convert PDFs to images
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

def load_groq_api_key():
    """Loads the GROQ API key from config.json"""
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
    """Extracts text from PDFs using PyMuPDF, falls back to GPU-based OCR if needed."""
    doc = fitz.open(file_path)
    text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
    doc.close()
    return text_list if text_list else extract_text_from_images(file_path)

def extract_text_from_images(pdf_path):
    """Extracts text from image-based PDFs using GPU-accelerated EasyOCR."""
    images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
    return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]

def setup_vectorstore(documents):
    """Creates a FAISS vector store using Hugging Face embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    return FAISS.from_texts(doc_chunks, embeddings)

def create_chain(vectorstore):
    """Creates the chat chain with optimized retriever settings."""
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_extracted_text = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            extracted_text = extract_text_from_pdf(file_path)
            all_extracted_text.extend(extracted_text)
            st.success(f"✅ Extracted text from {uploaded_file.name}")
        except Exception as e:
            st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")
    if all_extracted_text:
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = setup_vectorstore(all_extracted_text)
        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

if "conversation_chain" in st.session_state:
    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

user_input = st.chat_input("Ask Llama...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    response = st.session_state.conversation_chain.invoke({
        "question": user_input,
        "chat_history": st.session_state.memory.chat_memory.messages
    })
    assistant_response = response.get("answer", "I'm sorry, I couldn't process that.")
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})

# Export Chat Feature
st.subheader("💾 Export Chat")

export_format = st.radio("Choose export format:", ["Text (.txt)", "JSON (.json)"], horizontal=True)

def export_chat():
    return "\n".join(f"{'User' if msg.type == 'human' else 'LEXIBOT'}: {msg.content}" for msg in st.session_state.memory.chat_memory.messages)

if st.button("Download Chat History"):
    chat_content = export_chat()
    chat_file = io.BytesIO(chat_content.encode("utf-8"))
    mime_type = "text/plain" if export_format == "Text (.txt)" else "application/json"
    file_name = "chat_history.txt" if export_format == "Text (.txt)" else "chat_history.json"
    st.download_button(label="📥 Download", data=chat_file, file_name=file_name, mime=mime_type)
