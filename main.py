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
from langdetect import detect
from deep_translator import GoogleTranslator

# Set Streamlit Page Config
st.set_page_config(page_title="Chat with Swag AI", page_icon="📝", layout="centered")

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Load GROQ API Key
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

# Initialize EasyOCR with multiple languages
reader = easyocr.Reader(["en", "es", "fr", "de", "hi", "zh"], gpu=torch.cuda.is_available())

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return text_list if text_list else extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"⚠️ Error extracting text from PDF: {e}")
        return []

def extract_text_from_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(reader.readtext(np.array(img), detail=0, paragraph=True)) for img in images]
    except Exception as e:
        st.error(f"⚠️ Error extracting text from images: {e}")
        return []

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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

# Streamlit UI
st.title("🦙 Chat with Swag AI - LLAMA 3.3 (Multilingual)")

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
        st.session_state.vectorstore = setup_vectorstore(all_extracted_text)
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

if "conversation_chain" in st.session_state:
    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

user_input = st.chat_input("Ask Llama...")

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_text(text, target_lang="en"):
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

async def get_response(user_input, user_lang):
    translated_input = translate_text(user_input, target_lang="en") if user_lang != "en" else user_input
    response = await asyncio.to_thread(
        st.session_state.conversation_chain.invoke,
        {"question": translated_input, "chat_history": st.session_state.memory.chat_memory.messages}
    )
    answer = response.get("answer", "I'm sorry, I couldn't process that.")
    return translate_text(answer, target_lang=user_lang) if user_lang != "en" else answer

if user_input:
    user_lang = detect_language(user_input)
    st.info(f"Detected language: {user_lang.upper()}")
    with st.chat_message("user"):
        st.markdown(user_input)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        assistant_response = loop.run_until_complete(get_response(user_input, user_lang))
    except Exception as e:
        assistant_response = f"⚠️ Error: {str(e)}"
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})

