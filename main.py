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

from docx import Document
import io
from docx import Document
from docx.shared import Inches
import io
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

from PIL import Image
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
        🤖 LEXIBOT - POWERED BY SEMICOLON
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

def add_hyperlink(paragraph, url, text, color="0000FF", underline=True):
    """Add a clickable hyperlink to a paragraph in a Word document."""
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    # Underline
    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rPr.append(u)

    # Color
    if color:
        c = OxmlElement("w:color")
        c.set(qn("w:val"), color)
        rPr.append(c)

    new_run.append(rPr)
    text_elem = OxmlElement("w:t")
    text_elem.text = text
    new_run.append(text_elem)
    hyperlink.append(new_run)

    paragraph._p.append(hyperlink)
    return paragraph



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

def rewrite_text_with_llm(text, replacement_mapping):

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
    rewritten_text = rewrite_text_with_llm(text, replacement_mapping)


    return highlighted_text, rewritten_text, masculine_count, feminine_count, detected_masculine_words, detected_feminine_words

def ats_percentage_score(resume_text, job_description):
    prompt = f"""
You are a skilled ATS (Applicant Tracking System) with expertise in technical hiring.

Task:
Evaluate the given resume against the job description and provide the following:

1. **Candidate Name**: Extract the full name of the candidate if available (from the top of the resume).
2. **Percentage Match**: Provide only the number (no percentage symbol).
3. **Missing Keywords**: List keywords from the job description that are missing in the resume.
4. **Final Thoughts**: Give a short summary of the candidate's fit for the job.

### Job Description:
\"\"\"{job_description}\"\"\"

### Resume:
\"\"\"{resume_text}\"\"\"

Return the output in the format:
Candidate Name: <name or "Not Found">
Percentage Match: <only number>
Missing Keywords: <comma-separated list>
Final Thoughts: <brief summary>
"""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
    response = llm.invoke(prompt)
    return response.content.strip()




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

    st.sidebar.markdown("### 📝 Paste Job Description")
job_description = st.sidebar.text_area("Enter the Job Description here", height=200)

if job_description.strip() == "":
    st.sidebar.warning("Please enter a job description to evaluate the resumes.")


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
        ats_result = ats_percentage_score(full_text, job_description)

        # Parse the ATS result to extract details
        name_match = re.search(r"Candidate Name:\s*(.*)", ats_result)
        percent_match = re.search(r"Percentage Match:\s*(\d+)", ats_result)
        missing_keywords = re.search(r"Missing Keywords:\s*(.*)", ats_result)
        final_thoughts = re.search(r"Final Thoughts:\s*(.*)", ats_result)

        resume_data.append({
            "Resume Name": uploaded_file.name,
            "Candidate Name": name_match.group(1) if name_match else "Not Found",
            "ATS Match %": int(percent_match.group(1)) if percent_match else 0,
            "Missing Keywords": missing_keywords.group(1) if missing_keywords else "N/A",
            "Fit Summary": final_thoughts.group(1) if final_thoughts else "N/A",
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
tab1, tab2 = st.tabs(["📊 Dashboard", "🧾 Resume Builder"])

# === TAB 1: Dashboard ===
with tab1:
    if resume_data:
        total_masc = sum(r["Masculine Words"] for r in resume_data)
        total_fem = sum(r["Feminine Words"] for r in resume_data)
        avg_bias = round(np.mean([r["Bias Score (0 = Fair, 1 = Biased)"] for r in resume_data]), 2)
        total_resumes = len(resume_data)

        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Resumes Uploaded", total_resumes)
        with col2:
            st.metric("🔎 Avg. Bias Score", avg_bias)
        with col3:
            st.metric("🔵 Total Masculine Words", total_masc)
        with col4:
            st.metric("🔴 Total Feminine Words", total_fem)

        st.markdown("### 🗂️ Resumes Overview")
        df = pd.DataFrame(resume_data)
        st.dataframe(
            df[["Resume Name", "Candidate Name", "ATS Match %", "Bias Score (0 = Fair, 1 = Biased)", "Masculine Words", "Feminine Words"]],
            use_container_width=True
        )

    # 📈 Charts Section
    st.markdown("### 📊 Visual Analysis")
    tab1, tab2 = st.tabs(["📉 Bias Score Chart", "⚖ Gender-Coded Words"])

    with tab1:
        st.subheader("Bias Score Comparison Across Resumes")
        st.bar_chart(df.set_index("Resume Name")[["Bias Score (0 = Fair, 1 = Biased)"]])

    with tab2:
        st.subheader("Masculine vs Feminine Word Usage")
        fig, ax = plt.subplots(figsize=(10, 5))
        index = np.arange(len(df))
        bar_width = 0.35

        ax.bar(index, df["Masculine Words"], bar_width, label="Masculine", color="#3498db")
        ax.bar(index + bar_width, df["Feminine Words"], bar_width, label="Feminine", color="#e74c3c")

        ax.set_xlabel("Resumes", fontsize=12)
        ax.set_ylabel("Word Count", fontsize=12)
        ax.set_title("Gender-Coded Word Usage per Resume", fontsize=14)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(df["Resume Name"], rotation=45, ha='right')
        ax.legend()
        st.pyplot(fig)

    # 📑 Individual Resume Reports
    st.markdown("### 📝 Detailed Resume Reports")
    for resume in resume_data:
        with st.expander(f"📄 {resume['Resume Name']} | {resume['Candidate Name']}", expanded=False):
            st.markdown(f"#### 🧠 ATS Evaluation for {resume['Candidate Name']}")
            st.write(f"**ATS Match %:** {resume['ATS Match %']}%")
            st.write(f"**Missing Keywords:** {resume['Missing Keywords']}")
            st.write(f"**Fit Summary:** {resume['Fit Summary']}")

            st.divider()

            # Tabs inside expander
            detail_tab1, detail_tab2 = st.tabs(["🔎 Bias Analysis", "✅ Rewritten Resume"])

            with detail_tab1:
                st.markdown("#### Bias-Highlighted Original Text")
                st.markdown(resume["Highlighted Text"], unsafe_allow_html=True)

                st.markdown("### 📌 Gender-Coded Word Counts:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔵 Masculine Words", resume["Masculine Words"])
                    if resume["Detected Masculine Words"]:
                        st.markdown("### 📚 Detected Words:")
                        st.write("**Masculine Words Detected:**")
                        st.success(", ".join(f"{word} ({count})" for word, count in resume["Detected Masculine Words"].items()))
                    else:
                        st.info("No masculine words detected.")

                with col2:
                    st.metric("🔴 Feminine Words", resume["Feminine Words"])
                    if resume["Detected Feminine Words"]:
                        st.markdown("### 📚 Detected Words:")
                        st.write("**Feminine Words Detected:**")
                        st.success(", ".join(f"{word} ({count})" for word, count in resume["Detected Feminine Words"].items()))
                    else:
                        st.info("No feminine words detected.")

            with detail_tab2:
                st.markdown("#### ✨ Bias-Free Rewritten Resume")
                st.write(resume["Rewritten Text"])

                st.download_button(
                    label="📥 Download Bias-Free Resume",
                    data=resume["Rewritten Text"],
                    file_name=f"{resume['Resume Name'].split('.')[0]}_bias_free.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

# === TAB 2: Resume Builder ===
# === TAB 2: Resume Builder ===
with tab2:
 with tab2:
  st.markdown("## 🧾 <span style='color:#336699;'>Advanced Resume Builder</span>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 2px solid #bbb;'>", unsafe_allow_html=True)

    # Session state setup
fields = ["name", "email", "phone", "linkedin", "location", "portfolio", "summary", "skills", "education", "projects"]
for f in fields:
        if f not in st.session_state:
            st.session_state[f] = ""

    # Initialize dynamic work experience blocks
if "experience_entries" not in st.session_state:
        st.session_state.experience_entries = [{"title": "", "company": "", "duration": "", "description": ""}]

    # === FORM ===
with st.form("resume_form"):
        st.markdown("### 👤 <u>Personal Information</u>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("👤 Full Name", value=st.session_state["name"], key="name", placeholder="John Doe")
            st.text_input("📞 Phone Number", value=st.session_state["phone"], key="phone", placeholder="+1234567890")
            st.text_input("📍 Location", value=st.session_state["location"], key="location", placeholder="City, Country")
        with col2:
            st.text_input("📧 Email", value=st.session_state["email"], key="email", placeholder="you@example.com")
            st.text_input("🔗 LinkedIn", value=st.session_state["linkedin"], key="linkedin", placeholder="https://linkedin.com/in/...")
            st.text_input("🌐 Portfolio", value=st.session_state["portfolio"], key="portfolio", placeholder="https://yourportfolio.com")

        st.markdown("### 📝 <u>Professional Summary</u>", unsafe_allow_html=True)
        st.text_area("Summarize your background and expertise.", value=st.session_state["summary"], key="summary", height=120)

        st.markdown("### 💼 <u>Skills</u>", unsafe_allow_html=True)
        st.text_area("🔧 Skills (comma-separated)", value=st.session_state["skills"], key="skills", height=100)

        st.markdown("### 🧱 <u>Work Experience</u>", unsafe_allow_html=True)
        for idx, exp in enumerate(st.session_state.experience_entries):
            with st.expander(f"Experience #{idx+1}", expanded=True):
                exp["title"] = st.text_input(f"Job Title #{idx+1}", value=exp["title"], key=f"title_{idx}")
                exp["company"] = st.text_input(f"Company #{idx+1}", value=exp["company"], key=f"company_{idx}")
                exp["duration"] = st.text_input(f"Duration #{idx+1}", value=exp["duration"], key=f"duration_{idx}")
                exp["description"] = st.text_area(f"Description #{idx+1}", value=exp["description"], key=f"description_{idx}")

        if st.form_submit_button("➕ Add Experience"):
            st.session_state.experience_entries.append({"title": "", "company": "", "duration": "", "description": ""})

        st.markdown("### 🎓 <u>Education & Projects</u>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("🎓 Education", value=st.session_state["education"], key="education", height=120)
        with col2:
            st.text_area("🛠 Projects", value=st.session_state["projects"], key="projects", height=120)

        submitted = st.form_submit_button("📑 Generate Resume")

    # === Generate Word Resume ===
if submitted:
        doc = Document()
        doc.add_heading(st.session_state["name"], 0)

        doc.add_paragraph(f"📞 {st.session_state['phone']} | 📍 {st.session_state['location']}")

# Add clickable email
if st.session_state["email"]:
    p = doc.add_paragraph()
    add_hyperlink(p, f"mailto:{st.session_state['email']}", st.session_state["email"])

# Add clickable LinkedIn
if st.session_state["linkedin"]:
    p = doc.add_paragraph()
    add_hyperlink(p, st.session_state["linkedin"], "LinkedIn Profile")

# Add clickable Portfolio
if st.session_state["portfolio"]:
    p = doc.add_paragraph()
    add_hyperlink(p, st.session_state["portfolio"], "Portfolio Website")


    doc.add_heading("Professional Summary", level=1)
    doc.add_paragraph(st.session_state["summary"])

    doc.add_heading("Skills", level=1)
    skills = [s.strip() for s in st.session_state["skills"].split(",") if s.strip()]
    for skill in skills:
            doc.add_paragraph(f"• {skill}", style="List Bullet")

    doc.add_heading("Work Experience", level=1)
    for exp in st.session_state.experience_entries:
            if exp["title"] or exp["company"]:
                para = doc.add_paragraph()
                para.add_run(exp["title"]).bold = True
                para.add_run(f" | {exp['company']} | {exp['duration']}")
                doc.add_paragraph(exp["description"])

    doc.add_heading("Education", level=1)
    doc.add_paragraph(st.session_state["education"])

    if st.session_state["projects"].strip():
            doc.add_heading("Projects", level=1)
            doc.add_paragraph(st.session_state["projects"])

        # Download
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)

    st.success("✅ Resume Generated Successfully!")
    st.download_button(
            label="📥 Download Resume (Word)",
            data=doc_io,
            file_name=f"{st.session_state['name'].replace(' ', '_')}_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    


                
        

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
