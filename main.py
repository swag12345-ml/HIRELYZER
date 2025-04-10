import streamlit as st
import os, json, torch, asyncio, re
from dotenv import load_dotenv
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

# Setup
st.set_page_config(page_title="Bias Detector for Hiring Tools", page_icon="⚖️", layout="centered")
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# Load API key
def load_groq_api_key():
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        st.error("Missing config.json for GROQ API Key.")
        st.stop()

groq_api_key = load_groq_api_key()
if not groq_api_key:
    st.stop()

# EasyOCR
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# Text extraction
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return text_list if text_list else extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return []

def extract_text_from_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]
    except Exception as e:
        st.error(f"Error from OCR: {e}")
        return []

# Bias detection
def analyze_bias(text):
    flags = []
    context_snippets = []

    gendered_terms = ["he", "she", "his", "her", "manpower", "chairman"]
    for term in gendered_terms:
        matches = re.findall(rf"\b({term})\b", text, re.IGNORECASE)
        if matches:
            flags.append(f"Gendered term detected: **{term}**")
            index = text.lower().find(term)
            context_snippets.append(text[max(0, index-100):index+100])

    if "ivy league" in text.lower():
        flags.append("Potential elitism bias: Emphasis on 'Ivy League'")
        index = text.lower().find("ivy league")
        context_snippets.append(text[max(0, index-100):index+100])

    if re.search(r"\b(only|must have)\s+(\d+)\s+years\b", text.lower()):
        flags.append("Strict experience requirement – may limit diversity")
        match = re.search(r"\b(only|must have)\s+(\d+)\s+years\b", text.lower())
        if match:
            index = match.start()
            context_snippets.append(text[max(0, index-100):index+100])

    return flags, context_snippets

# Vector store setup
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    return FAISS.from_texts(doc_chunks, embeddings)

# Chat chain setup
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

# UI
st.title("⚖️ Bias Detector for AI Hiring Tools")
uploaded_files = st.file_uploader("Upload one or more PDFs describing hiring logic or filtering systems", type=["pdf"], accept_multiple_files=True)

bias_report = []
all_text_for_vectorstore = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        extracted_text = extract_text_from_pdf(file_path)
        full_text = "\n".join(extracted_text)
        all_text_for_vectorstore.extend(extracted_text)

        flags, contexts = analyze_bias(full_text)

        bias_report.append({
            "filename": uploaded_file.name,
            "has_bias": bool(flags),
            "description": ", ".join(set(flags)) if flags else "No major bias found.",
            "references": contexts
        })

    # Bias report summary
    st.subheader("Bias Report Summary")
    for entry in bias_report:
        st.markdown(f"### {entry['filename']}")
        if entry["has_bias"]:
            st.warning(entry["description"])
            for i, ref in enumerate(entry["references"]):
                with st.expander(f"View Bias Context #{i+1}"):
                    st.code(ref, language="text")
        else:
            st.success(entry["description"])

    # Load chatbot
    st.session_state.vectorstore = setup_vectorstore(all_text_for_vectorstore)
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Chat section
if "conversation_chain" in st.session_state:
    st.subheader("Chat with the Uploaded Hiring Docs")

    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

    user_input = st.chat_input("Ask about fairness, keywords, or biases...")

    async def get_response(user_input):
        response = await asyncio.to_thread(
            st.session_state.conversation_chain.invoke,
            {"question": user_input, "chat_history": st.session_state.memory.chat_memory.messages}
        )
        return response.get("answer", "Could not process the question.")

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