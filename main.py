import streamlit as st  # Streamlit must be imported first
import os
import json
import torch
import asyncio
from dotenv import load_dotenv
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
from fpdf import FPDF  # For PDF generation

# Set Streamlit Page Config
st.set_page_config(page_title="Chat with LEXIBOT", page_icon="📝", layout="centered")

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Load GROQ API Key
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
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

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

user_input = st.chat_input("Ask Llama...")

async def get_response(user_input):
    """Runs the chatbot response inside an async event loop."""
    response = await asyncio.to_thread(
        st.session_state.conversation_chain.invoke,
        {"question": user_input, "chat_history": st.session_state.memory.chat_memory.messages}
    )
    return response.get("answer", "I'm sorry, I couldn't process that.")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        assistant_response = loop.run_until_complete(get_response(user_input))
    except Exception as e:
        assistant_response = f"⚠️ Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    
    # Save chat history in session state
    st.session_state.chat_history.append({"user": user_input, "bot": assistant_response})

    st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})

# Chat Export Feature
def export_chat():
    """Exports chat history in multiple formats."""
    
    # TXT Export
    chat_text = "\n".join([f"User: {chat['user']}\nBot: {chat['bot']}\n" for chat in st.session_state.chat_history])
    txt_path = os.path.join(working_dir, "chat_history.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(chat_text)
    
    # JSON Export
    json_path = os.path.join(working_dir, "chat_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_history, f, indent=4)

    # PDF Export
    pdf_path = os.path.join(working_dir, "chat_history.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Chat History", ln=True, align="C")
    pdf.ln(10)  # Line break

    for chat in st.session_state.chat_history:
        pdf.multi_cell(0, 10, f"User: {chat['user']}\nBot: {chat['bot']}\n", border=0)
        pdf.ln(5)

    pdf.output(pdf_path)

    return txt_path, json_path, pdf_path

# Show download buttons if chat history exists
if st.session_state.chat_history:
    txt_file, json_file, pdf_file = export_chat()

    with open(txt_file, "rb") as f:
        st.download_button(
            label="📥 Download Chat as TXT",
            data=f,
            file_name="chat_history.txt",
            mime="text/plain"
        )

    with open(json_file, "rb") as f:
        st.download_button(
            label="📥 Download Chat as JSON",
            data=f,
            file_name="chat_history.json",
            mime="application/json"
        )

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download Chat as PDF",
            data=f,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )
