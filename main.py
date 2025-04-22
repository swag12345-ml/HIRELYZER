import os
import json
import torch
import random
import fitz
import easyocr
import asyncio
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

from collections import Counter
from dotenv import load_dotenv
from pdf2image import convert_from_path
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set page config
st.set_page_config(page_title="Chat with LEXIBOT", page_icon="📝", layout="centered")

# CSS Customization
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
        color: #ccffff;
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
        🤖 LEXIBOT - POWERED BY BROCODE
    </div>
    """,
    unsafe_allow_html=True,
)


# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# Load API Key
def load_groq_api_key():
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        st.error("❌ config.json not found.")
        st.stop()

groq_api_key = load_groq_api_key()
if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing.")
    st.stop()

# OCR Reader
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# Gender-coded language
gender_words = {
    "masculine": [
        "active", "aggressive", "ambitious", "analytical", "assertive", "autonomous", "boast", "bold",
        "challenging", "competitive", "confident", "courageous", "decisive", "determined", "dominant", "driven",
        "dynamic", "forceful", "independent", "individualistic", "intellectual", "lead", "leader", "objective",
        "outspoken", "persistent", "principled", "proactive", "resilient", "self-reliant", "self-sufficient",
        "strong", "superior", "tenacious","guru","tech guru","technical guru", "visionary", "manpower", "strongman", "command",
        "assert", "headstrong", "rockstar", "superstar", "go-getter", "trailblazer", "results-driven",
        "fast-paced", "driven", "determination", "competitive spirit"
    ],
    
    "feminine": [
        "affectionate", "agreeable", "attentive", "collaborative", "committed", "compassionate", "considerate",
        "cooperative", "dependable", "dependent", "emotional", "empathetic", "enthusiastic", "friendly", "gentle",
        "honest", "inclusive", "interpersonal", "kind", "loyal", "modest", "nurturing", "pleasant", "polite",
        "sensitive", "supportive", "sympathetic", "tactful", "tender", "trustworthy", "understanding", "warm",
        "yield", "adaptable", "communal", "helpful", "dedicated", "respectful", "nurture", "sociable",
        "relationship-oriented", "team player", "dependable", "people-oriented", "empathetic listener",
        "gentle communicator", "open-minded"
    ]
}



# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return text_list if text_list else extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"⚠ Error extracting text: {e}")
        return []

def extract_text_from_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]
    except Exception as e:
        st.error(f"⚠ Error extracting from image: {e}")
        return []

# Detect bias in resume

def detect_bias(text):
    text = text.lower()
    masc, fem = 0, 0

    masculine_words_sorted = sorted(gender_words["masculine"], key=len, reverse=True)
    feminine_words_sorted = sorted(gender_words["feminine"], key=len, reverse=True)

    for phrase in masculine_words_sorted:
        masc += len(re.findall(rf'\b{re.escape(phrase)}\b', text))

    for phrase in feminine_words_sorted:
        fem += len(re.findall(rf'\b{re.escape(phrase)}\b', text))

    total = masc + fem

    if total == 0:
        return 0.0, masc, fem  

    bias_score = min(total / 20, 1.0)

    return round(bias_score, 2), masc, fem

gender_words = {
    "masculine": [
        "active", "aggressive", "ambitious", "analytical", "assertive", "autonomous", "boast", "bold",
        "challenging", "competitive", "confident", "courageous", "decisive", "determined", "dominant", "driven",
        "dynamic", "forceful", "independent", "individualistic", "intellectual", "lead", "leader", "objective",
        "outspoken", "persistent", "principled", "proactive", "resilient", "self-reliant", "self-sufficient",
        "strong", "superior", "tenacious","guru","tech guru","technical guru", "visionary", "manpower", "strongman", "command",
        "assert", "headstrong", "rockstar", "superstar", "go-getter", "trailblazer", "results-driven",
        "fast-paced", "driven", "determination", "competitive spirit"
    ],
    
    "feminine": [
        "affectionate", "agreeable", "attentive", "collaborative", "committed", "compassionate", "considerate",
        "cooperative", "dependable", "dependent", "emotional", "empathetic", "enthusiastic", "gentle",
        "honest", "inclusive", "interpersonal", "kind", "loyal", "modest", "nurturing", "pleasant", "polite",
        "sensitive", "supportive", "sympathetic", "tactful", "tender", "trustworthy", "understanding", "warm",
        "yield", "adaptable", "communal", "helpful", "dedicated", "respectful", "nurture", "sociable",
        "relationship-oriented", "team player", "dependable", "people-oriented", "empathetic listener",
        "gentle communicator", "open-minded"
    ]
}
replacement_mapping = {
    "masculine": {
        "active": "engaged",
        "aggressive": "proactive",
        "ambitious": "motivated",
        "analytical": "detail-oriented",
        "assertive": "confident",
        "autonomous": "self-directed",
        "boast": "highlight",
        "bold": "confident",
        "challenging": "demanding",
        "competitive": "goal-oriented",
        "confident": "self-assured",
        "courageous": "bold",
        "decisive": "action-oriented",
        "determined": "focused",
        "dominant": "influential",
        "driven": "committed",
        "dynamic": "adaptable",
        "forceful": "persuasive",
        "guru":"technical expert",
        "independent": "self-sufficient",
        "individualistic": "self-motivated",
        "intellectual": "knowledgeable",
        "lead": "guide",
        "leader": "team lead",
        "objective": "unbiased",
        "outspoken": "expressive",
        "persistent": "resilient",
        "principled": "ethical",
        "proactive": "initiative-taking",
        "resilient": "adaptable",
        "self-reliant": "resourceful",
        "self-sufficient": "capable",
        "strong": "capable",
        "superior": "exceptional",
        "tenacious": "determined",
        "technical guru": "technical expert",
        "visionary": "forward-thinking",
        "manpower": "workforce",
        "strongman": "resilient individual",
        "command": "direct",
        "assert": "state confidently",
        "headstrong": "determined",
        "rockstar": "top performer",
        "superstar": "outstanding contributor",
        "go-getter": "initiative-taker",
        "trailblazer": "innovator",
        "results-driven": "outcome-focused",
        "fast-paced": "dynamic",
        "determination": "commitment",
        "competitive spirit": "goal-oriented mindset"
    },
    
    "feminine": {
        "affectionate": "approachable",
        "agreeable": "cooperative",
        "attentive": "observant",
        "collaborative": "team-oriented",
        "collaborate": "team-oriented",
        "collaborated": "worked together",
        "committed": "dedicated",
        "compassionate": "caring",
        "considerate": "thoughtful",
        "cooperative": "supportive",
        "dependable": "reliable",
        "dependent": "team-oriented",
        "emotional": "passionate",
        "empathetic": "understanding",
        "enthusiastic": "positive",
        "gentle": "respectful",
        "honest": "trustworthy",
        "inclusive": "open-minded",
        "interpersonal": "people-focused",
        "kind": "respectful",
        "loyal": "dedicated",
        "modest": "humble",
        "nurturing": "supportive",
        "pleasant": "positive",
        "polite": "professional",
        "sensitive": "attentive",
        "supportive": "encouraging",
        "sympathetic": "understanding",
        "tactful": "diplomatic",
        "tender": "considerate",
        "trustworthy": "reliable",
        "understanding": "empathetic",
        "warm": "welcoming",
        "yield": "adaptable",
        "adaptable": "flexible",
        "communal": "team-centered",
        "helpful": "supportive",
        "dedicated": "committed",
        "respectful": "considerate",
        "nurture": "develop",
        "sociable": "friendly",
        "relationship-oriented": "team-focused",
        "team player": "collaborative member",
        "people-oriented": "person-focused",
        "empathetic listener": "active listener",
        "gentle communicator": "considerate communicator",
        "open-minded": "inclusive"
    }
}

def rewrite_text_with_llm(text):
    prompt = f"""
You are an expert career advisor and professional resume language editor.

Your task is to:

1. **Rewrite the following resume text** to:
   - Remove or replace any gender-coded, biased, or non-inclusive language.
   - Use **professional, inclusive, neutral, clear, and grammatically correct language**.
   - **Retain all technical terms, job-specific keywords, certifications, and proper names.**
   - Do not add new content or remove important information.
   - Preserve the original meaning and intent of each sentence.

### 2️⃣ Structure and Organization:
- Organize the rewritten text into clearly labeled standard resume sections, such as:
  - **Name**
  - **Contact Information**
  - **Email**
  - **Professional Summary**
  - **Work Experience**
  - **Skills**
  - **Certifications**
  - **Education**
  - **Projects**
- Include **only the sections that exist in the provided text** — do not add new ones.
- If any of **Name**, **Contact Information**, or **Email** is present in the text, organize it clearly at the top under respective headings.

3. **Strictly apply the following word replacement mapping:**

{replacement_mapping}

- If a word or phrase in the text matches a key from this mapping, replace it exactly with the corresponding replacement.
- Leave all other words unchanged.

4. **Suggest 5 suitable job titles** based on the candidate's rewritten resume text.
   - Ensure the job titles are **realistic, well-matched to the candidate’s experience, skills, and qualifications**.
   - Provide a brief reason for each suggestion, explaining why it fits based on the resume content.

---

**Resume Text to Rewrite:**
\"\"\"{text}\"\"\"

---

**Final Rewritten, Organized, Bias-Free, Inclusive Resume:**

[Your rewritten and organized resume content here]

---

**Suggested Job Titles (with reasons):**
1. **Job Title 1** — Reason based on skills/experience
2. **Job Title 2** — Reason based on skills/experience
3. **Job Title 3** — Reason based on skills/experience
4. **Job Title 4** — Reason based on skills/experience
5. **Job Title 5** — Reason based on skills/experience
"""



    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
    response = llm.invoke(prompt)
    rewritten_text = response.content  
    return rewritten_text
    



def rewrite_and_highlight(text, replacement_mapping):
    highlighted_text = text
    masculine_count, feminine_count = 0, 0
    detected_masculine_words = Counter()
    detected_feminine_words = Counter()

    words = re.findall(r'\b\w+\b', text)

    for w in words:
        lemma = lemmatizer.lemmatize(w.lower())
        if lemma in gender_words["masculine"]:
            masculine_count += 1
            detected_masculine_words[lemma] += 1
            highlighted_text = re.sub(rf'\b{re.escape(w)}\b', f":blue[{w}]", highlighted_text)
 
        elif lemma in gender_words["feminine"]:
            feminine_count += 1
            detected_feminine_words[lemma] += 1
            highlighted_text = re.sub(rf'\b{re.escape(w)}\b', f":red[{w}]", highlighted_text)

    # Now rewrite the text using the LLM
    rewritten_text = rewrite_text_with_llm(text)

    return highlighted_text, rewritten_text, masculine_count, feminine_count, detected_masculine_words, detected_feminine_words



# Setup Vector DB
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    return FAISS.from_texts(doc_chunks, embeddings)

# Create Conversational Chain
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

# App Title
st.title("🦙 Chat with LEXIBOT - LLAMA 3.3 (Bias Detection + QA + GPU)")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

resume_data = []

if uploaded_files:
    all_text = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = extract_text_from_pdf(file_path)
        all_text.extend(text)

        full_text = " ".join(text)
        bias_score, masc, fem = detect_bias(full_text)
        highlighted_text, rewritten_text, masc_count, fem_count, detected_masc, detected_fem = rewrite_and_highlight(full_text,replacement_mapping)

        resume_data.append({
            "Resume Name": uploaded_file.name,
            "Bias Score (0 = Fair, 1 = Biased)": bias_score,
            "Masculine Words": masc_count,
            "Feminine Words": fem_count,
            "Detected Masculine Words": detected_masc,
            "Detected Feminine Words": detected_fem,
            "Text Preview": full_text[:300] + "...",
            "Highlighted Text": highlighted_text,
            "Rewritten Text": rewritten_text
        })

    st.success("✅ All resumes processed!")

    # Setup vectorstore if needed
    if all_text:
        st.session_state.vectorstore = setup_vectorstore(all_text)
        st.session_state.chain = create_chain(st.session_state.vectorstore)

# 📊 Dashboard and Metrics
if resume_data:
    total_masc = sum(r["Masculine Words"] for r in resume_data)
    total_fem = sum(r["Feminine Words"] for r in resume_data)
    avg_bias = round(np.mean([r["Bias Score (0 = Fair, 1 = Biased)"] for r in resume_data]), 2)
    total_resumes = len(resume_data)
   
    st.markdown("### 📈 Summary Stats")
    col1, col2, col3, col4 = st.columns(4)  # ✅ Now 4 columns instead of 3
    col1.metric("📄 Resumes Uploaded", total_resumes)
    col2.metric("🔎 Avg. Bias Score", avg_bias)
    col3.metric("🔵 Total Masculine Words", total_masc)
    col4.metric("🔴 Total Feminine Words", total_fem)

    # 📋 Data Table
    st.markdown("### 📊 Resume Bias Dashboard")
    df = pd.DataFrame(resume_data)
    st.dataframe(df, use_container_width=True)

    # 📉 Bias Score Bar Chart
    st.subheader("📉 Bias Score Comparison")
    st.bar_chart(df.set_index("Resume Name")[["Bias Score (0 = Fair, 1 = Biased)"]])

    # ⚖ Masculine vs Feminine Words Chart
    st.subheader("⚖ Masculine vs Feminine Words")
    fig, ax = plt.subplots(figsize=(10, 5))
    index = np.arange(len(df))
    bar_width = 0.35

    ax.bar(index, df["Masculine Words"], bar_width, label="Masculine", color="steelblue")
    ax.bar(index + bar_width, df["Feminine Words"], bar_width, label="Feminine", color="salmon")

    ax.set_xlabel("Resumes")
    ax.set_ylabel("Count")
    ax.set_title("Gender-Coded Language Use")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df["Resume Name"], rotation=45, ha='right')
    ax.legend()

    st.pyplot(fig)

    # 📑 Individual Resume Expanders
    for resume in resume_data:
        with st.expander(f"📝 View & Rewrite {resume['Resume Name']}"):
            st.markdown("🔎 **Bias-Highlighted Text:**")
            st.markdown(resume["Highlighted Text"], unsafe_allow_html=True)

            st.markdown("✅ **Bias-Free Rewritten Text:**")
            st.markdown(resume["Rewritten Text"])

            st.markdown("### 📌 Gender-Coded Word Counts:")
            col1, col2 = st.columns(2)
            col1.metric("🔵 Masculine Words", resume["Masculine Words"])
            col2.metric("🔴 Feminine Words", resume["Feminine Words"])

            st.markdown("### 📚 Detected Words:")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Masculine Words Found:**")
                if resume["Detected Masculine Words"]:
                    detected_masc_words = ", ".join(f"{word} ({count})" for word, count in resume["Detected Masculine Words"].items())
                    st.write(detected_masc_words)
                else:
                    st.write("None")

            with col4:
                st.markdown("**Feminine Words Found:**")
                if resume["Detected Feminine Words"]:
                    detected_fem_words = ", ".join(f"{word} ({count})" for word, count in resume["Detected Feminine Words"].items())
                    st.write(detected_fem_words)
                else:
                    st.write("None")

# 💬 Chat Section
if "chain" in st.session_state:
    for msg in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)

user_input = st.chat_input("Ask LEXIBOT anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = st.session_state.chain.invoke(
            {"question": user_input, "chat_history": st.session_state.memory.chat_memory.messages}
        )
        answer = response.get("answer", "❌ No answer found.")
    except Exception as e:
        answer = f"⚠ Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.memory.save_context({"input": user_input}, {"output": answer})
