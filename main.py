import streamlit as st  # Streamlit must be imported first

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Chat with LEXIBOT", page_icon="📝", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    body {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Orbitron', sans-serif;
    }

    /* Glitch Animation */
    @keyframes glitch {
        0% { text-shadow: 2px 2px #ff0000, -2px -2px #00ffcc; }
        25% { text-shadow: -2px -2px #ff0000, 2px 2px #00ffcc; }
        50% { text-shadow: 3px -3px #ff00ff, -3px 3px #ff9900; }
        75% { text-shadow: -3px -3px #ff9900, 3px 3px #33ff00; }
        100% { text-shadow: 2px 2px #ff00ff, -2px -2px #0099ff; }
    }

    /* RGB Color Animation */
    @keyframes smoothGlow {
        0% { color: #ff0000; text-shadow: 0 0 10px #ff0000; }
        25% { color: #ff9900; text-shadow: 0 0 15px #ff9900; }
        50% { color: #00ff00; text-shadow: 0 0 20px #00ff00; }
        75% { color: #0099ff; text-shadow: 0 0 15px #0099ff; }
        100% { color: #ff00ff; text-shadow: 0 0 10px #ff00ff; }
    }

    /* Header */
    .header {
        font-size: 25px;
        font-weight: bold;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
        animation: glitch 0.8s infinite, smoothGlow 3s infinite alternate;
        text-shadow: 0px 0px 20px cyan;
    }

    /* Buttons - Neon Glow Effect */
    .stButton > button {
        background: linear-gradient(45deg, #ff0080, #007BFF);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        text-transform: uppercase;
        box-shadow: 0px 0px 15px rgba(0, 198, 255, 0.8);
        transition: 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #ff0077, #00ccff);
        transform: scale(1.08);
        box-shadow: 0px 0px 30px rgba(255, 0, 128, 1);
    }

    /* Chat Answer Box */
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes neonText {
        0% { color: #ff44ff; text-shadow: 0px 0px 15px #ff44ff; }
        50% { color: #00ffff; text-shadow: 0px 0px 20px #00ffff; }
        100% { color: #ff00ff; text-shadow: 0px 0px 15px #ff00ff; }
    }
    
    .stChatMessage {
        font-size: 18px;
        background: #1e293b;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #00ffff;
        color: #00ffff;
        text-shadow: 0px 0px 8px #00ffff;
        animation: glow 1.5s infinite alternate;
    }

    /* Upload Animation - Glowing File Upload */
    @keyframes pulse {
        0% { box-shadow: 0 0 5px cyan; }
        50% { box-shadow: 0 0 20px cyan; }
        100% { box-shadow: 0 0 5px cyan; }
    }
    .stFileUploader > div > div {
        border: 2px solid cyan;
        animation: pulse 2s infinite;
        padding: 10px;
        border-radius: 8px;
        background-color: rgba(0, 255, 255, 0.1);
        text-align: center;
    }

    </style>

    <div class="header">
        🤖 LEXIBOT - POWERED BY SEMICOLON
    </div>
    """,
    unsafe_allow_html=True,
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
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor



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


reader = easyocr.Reader(['en'], gpu=True)

def process_image(img):
    """Extract text from a single image using EasyOCR."""
    return "\n".join(reader.readtext(np.array(img), detail=0))

def extract_text_from_images(pdf_path):
    """Extracts text from all images in a PDF using CPU-optimized EasyOCR with multithreading."""
    images = convert_from_path(pdf_path, dpi=150)  # Convert all pages to images
    with ThreadPoolExecutor() as executor:
        text_results = list(executor.map(process_image, images))
    return text_results
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
