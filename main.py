import streamlit as st  # Streamlit must be imported first
import os
import json
import torch
import asyncio
import requests
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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# Set Streamlit Page Config
st.set_page_config(page_title="Chat with Swag AI", page_icon="📝", layout="centered")

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Load API Keys from config.json
def load_api_keys():
    """Loads API keys from config.json"""
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            config = json.load(f)
            return config.get("GROQ_API_KEY"), config.get("WHISPER_API_KEY")
    except FileNotFoundError:
        st.error("🚨 config.json not found. Please add your API keys.")
        st.stop()

groq_api_key, whisper_api_key = load_api_keys()
if not groq_api_key or not whisper_api_key:
    st.error("🚨 Missing API keys in config.json.")
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

# Real-time Speech Recognition with Whisper
def transcribe_audio(audio_bytes):
    """Sends real-time audio to Whisper API for transcription."""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {whisper_api_key}"}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": "whisper-1", "language": "en"}
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            st.error(f"⚠️ Whisper API Error: {response.text}")
            return ""
    except Exception as e:
        st.error(f"⚠️ Error in transcription: {e}")
        return ""

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    ),
)

if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    if audio_frames:
        audio_data = audio_frames[0].to_ndarray().tobytes()
        user_input = transcribe_audio(audio_data)
        if user_input:
            st.text(f"🎤 You said: {user_input}")

            with st.chat_message("user"):
                st.markdown(user_input)

            async def get_response(user_input):
                """Runs the chatbot response inside an async event loop."""
                response = await asyncio.to_thread(
                    st.session_state.conversation_chain.invoke,
                    {"question": user_input, "chat_history": st.session_state.memory.chat_memory.messages}
                )
                return response.get("answer", "I'm sorry, I couldn't process that.")

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                assistant_response = loop.run_until_complete(get_response(user_input))
            except Exception as e:
                assistant_response = f"⚠️ Error: {str(e)}"

            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})
