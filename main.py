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
from concurrent.futures import ThreadPoolExecutor  # For concurrent processing

# Additional libraries for voice assistant
import speech_recognition as sr  # This uses PyAudio internally
import pyttsx3

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
    """Extracts text from image-based PDFs using GPU-accelerated EasyOCR concurrently."""
    try:
        images = convert_from_path(pdf_path, dpi=150)
        
        def ocr_image(img):
            return "\n".join(reader.readtext(np.array(img), detail=0))
        
        with ThreadPoolExecutor() as executor:
            ocr_results = list(executor.map(ocr_image, images))
        
        return ocr_results
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

# Voice assistant functions
def record_voice():
    """Records audio from the microphone and returns recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Please speak now.")
        try:
            audio_data = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio_data)
            st.success(f"Voice input: {text}")
            return text
        except sr.WaitTimeoutError:
            st.error("⏰ Listening timed out. Please try again.")
        except sr.UnknownValueError:
            st.error("⚠️ Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"⚠️ Could not request results; {e}")
    return ""

def speak_text(text):
    """Converts text to speech using pyttsx3."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"⚠️ Text-to-speech error: {e}")

# Streamlit UI
st.title("🦙 Chat with Swag AI - LLAMA 3.3 (GPU Accelerated)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Allow multiple PDF uploads
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

# Display previous chat messages
if "conversation_chain" in st.session_state:
    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

# Input section: either text chat or voice chat
st.subheader("Chat Input")
text_input = st.chat_input("Ask Llama (text)...")

if st.button("Record Voice Input"):
    voice_input = record_voice()
    if voice_input:
        text_input = voice_input  # Override text input if voice input was captured

async def get_response(user_query):
    """Runs the chatbot response inside an async event loop."""
    response = await asyncio.to_thread(
        st.session_state.conversation_chain.invoke,
        {"question": user_query, "chat_history": st.session_state.memory.chat_memory.messages}
    )
    return response.get("answer", "I'm sorry, I couldn't process that.")

if text_input:
    with st.chat_message("user"):
        st.markdown(text_input)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        assistant_response = loop.run_until_complete(get_response(text_input))
    except Exception as e:
        assistant_response = f"⚠️ Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    
    # Save conversation context
    st.session_state.memory.save_context({"input": text_input}, {"output": assistant_response})
    
    # Speak out the assistant's response
    speak_text(assistant_response)
