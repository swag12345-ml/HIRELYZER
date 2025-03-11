import streamlit as st
import os
import json
import torch
import asyncio
from dotenv import load_dotenv
import fitz  # PyMuPDF for text extraction
import easyocr  # GPU-accelerated OCR
from pdf2image import convert_from_path
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
from fpdf import FPDF  # PDF generation

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

def extract_text_from_pdf(file_path):
    """Extracts text from PDFs using PyMuPDF, falls back to OCR if needed."""
    try:
        doc = fitz.open(file_path)
        text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return text_list if text_list else extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"⚠️ Error extracting text from PDF: {e}")
        return []

def extract_text_from_images(pdf_path):
    """Extracts text from image-based PDFs using GPU-accelerated EasyOCR."""
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]
    except Exception as e:
        st.error(f"⚠️ Error extracting text from images: {e}")
        return []

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

# Upload PDF Files
uploaded_files = st.file_uploader("📂 Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_extracted_text = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            extracted_text = extract_text_from_pdf(file_path)
            all_extracted_text.extend(extracted_text)  # Collect text from all PDFs
            st.success(f"✅ Extracted text from {uploaded_file.name}")
        except Exception as e:
            st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")

    # Process all extracted text together
    if all_extracted_text:
        st.session_state.vectorstore = setup_vectorstore(all_extracted_text)
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

if "conversation_chain" in st.session_state:
    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

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
    
    st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})

# Export Chat as JSON
def export_chat_json():
    chat_data = st.session_state.memory.load_memory_variables({})
    file_path = os.path.join(working_dir, "chat_history.json")
    with open(file_path, "w") as f:
        json.dump(chat_data, f, indent=4)
    return file_path

# Export Chat as PDF
def export_chat_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Chat History", ln=True, align="C")
    
    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        pdf.multi_cell(0, 10, f"{message.type.upper()}: {message.content}\n")
    
    file_path = os.path.join(working_dir, "chat_history.pdf")
    pdf.output(file_path)
    return file_path

st.subheader("💾 Export Chat")
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Export as JSON"):
        file_path = export_chat_json()
        st.success("✅ Chat exported successfully!")
        st.download_button("⬇️ Download JSON", file_path, file_name="chat_history.json")

with col2:
    if st.button("📥 Export as PDF"):
        file_path = export_chat_pdf()
        st.success("✅ Chat exported successfully!")
        st.download_button("⬇️ Download PDF", file_path, file_name="chat_history.pdf")
