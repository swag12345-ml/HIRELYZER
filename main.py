# ------------------- Core Imports -------------------
import os, json, random, string, re, asyncio, io
import urllib.parse
from collections import Counter

# ------------------- External Libraries -------------------
import torch
import io
from io import BytesIO
import matplotlib.pyplot as plt
import fitz
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

# ------------------- Langchain & Embeddings -------------------
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

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        background-color: #0b0c10;
        color: #c5c6c7;
        scroll-behavior: smooth;
    }

    /* ---------- SCROLLBAR ---------- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1f2833;
    }
    ::-webkit-scrollbar-thumb {
        background: #00ffff;
        border-radius: 4px;
    }

    /* ---------- BANNER ---------- */
    .banner-container {
        width: 100%;
        height: 80px;
        background: linear-gradient(90deg, #000428, #004e92);
        border-bottom: 2px solid cyan;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        position: relative;
        margin-bottom: 20px;
    }

    .pulse-bar {
        position: absolute;
        display: flex;
        align-items: center;
        font-size: 22px;
        font-weight: bold;
        color: #00ffff;
        white-space: nowrap;
        animation: glideIn 12s linear infinite;
        text-shadow: 0 0 10px #00ffff;
    }

    .pulse-bar .bar {
        width: 10px;
        height: 30px;
        margin-right: 10px;
        background: #00ffff;
        box-shadow: 0 0 8px cyan;
        animation: pulse 1s ease-in-out infinite;
    }

    @keyframes glideIn {
        0% { left: -50%; opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { left: 110%; opacity: 0; }
    }

    @keyframes pulse {
        0%, 100% {
            height: 20px;
            background-color: #00ffff;
        }
        50% {
            height: 40px;
            background-color: #ff00ff;
        }
    }

    /* ---------- HEADER ---------- */
    .header {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 12px 0;
        animation: glowPulse 3s ease-in-out infinite;
        text-shadow: 0px 0px 10px #00ffff;
    }

    @keyframes glowPulse {
        0%, 100% {
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
        }
        50% {
            color: #ff00ff;
            text-shadow: 0 0 20px #ff00ff, 0 0 30px #ff00ff;
        }
    }

    /* ---------- FILE UPLOADER ---------- */
    .stFileUploader > div > div {
        border: 2px solid #00ffff;
        border-radius: 10px;
        background-color: rgba(0, 255, 255, 0.05);
        padding: 12px;
        box-shadow: 0 0 15px rgba(0,255,255,0.4);
        transition: box-shadow 0.3s ease-in-out;
    }
    .stFileUploader > div > div:hover {
        box-shadow: 0 0 25px rgba(0,255,255,0.8);
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: linear-gradient(45deg, #ff0080, #00bfff);
        color: white;
        font-size: 16px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        text-transform: uppercase;
        box-shadow: 0px 0px 12px #00ffff;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 24px #ff00ff;
        background: linear-gradient(45deg, #ff00aa, #00ffff);
    }

    /* ---------- CHAT MESSAGES ---------- */
    .stChatMessage {
        font-size: 18px;
        background: #1e293b;
        padding: 14px;
        border-radius: 10px;
        border: 2px solid #00ffff;
        color: #ccffff;
        text-shadow: 0px 0px 6px #00ffff;
        animation: glow 1.5s ease-in-out infinite alternate;
    }

    /* ---------- INPUTS ---------- */
    .stTextInput > div > input,
    .stTextArea > div > textarea {
        background-color: #1f2833;
        color: #00ffff;
        border: 1px solid #00ffff;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0,255,255,0.3);
    }

    /* ---------- METRICS ---------- */
    .stMetric {
        background-color: #0f172a;
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0,255,255,0.5);
        text-align: center;
    }

    /* ---------- MOBILE ---------- */
    @media (max-width: 768px) {
        .pulse-bar {
            font-size: 16px;
        }
        .header {
            font-size: 20px;
        }
    }
    </style>

    <!-- Banner -->
    <div class="banner-container">
        <div class="pulse-bar">
            <div class="bar"></div>
            <div>LEXIBOT - Elevate Your Resume Analysis</div>
        </div>
    </div>

    <!-- Header -->
    <div class="header">💼 LEXIBOT - AI ETHICAL RESUME ANALYZER</div>
    """,
    unsafe_allow_html=True
)

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Detect Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ------------------- Load API Keys -------------------
@st.cache_data(show_spinner=False)
def load_groq_api_keys():
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            data = json.load(f)
            keys = data.get("GROQ_API_KEYS", [])
            return keys if isinstance(keys, list) and keys else None
    except FileNotFoundError:
        st.error("❌ config.json not found.")
        st.stop()

groq_api_keys = load_groq_api_keys()
if not groq_api_keys:
    st.error("❌ GROQ_API_KEYS list is missing or empty in config.json.")
    st.stop()

groq_api_key = random.choice(groq_api_keys)

# ------------------- Lazy Initialization for Slow Components -------------------
@st.cache_resource(show_spinner=False)
def get_easyocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())

@st.cache_data(show_spinner=False)
def ensure_nltk():
    import nltk
    nltk.download('wordnet', quiet=True)
    return WordNetLemmatizer()

lemmatizer = ensure_nltk()
reader = get_easyocr_reader()

FEATURED_COMPANIES = {
    "tech": [
        {
            "name": "Google",
            "icon": "fab fa-google",
            "color": "#4285F4",
            "careers_url": "https://careers.google.com",
            "description": "Leading technology company known for search, cloud, and innovation",
            "categories": ["Software", "AI/ML", "Cloud", "Data Science"]
        },
        {
            "name": "Microsoft",
            "icon": "fab fa-microsoft",
            "color": "#00A4EF",
            "careers_url": "https://careers.microsoft.com",
            "description": "Global leader in software, cloud, and enterprise solutions",
            "categories": ["Software", "Cloud", "Gaming", "Enterprise"]
        },
        {
            "name": "Amazon",
            "icon": "fab fa-amazon",
            "color": "#FF9900",
            "careers_url": "https://www.amazon.jobs",
            "description": "E-commerce and cloud computing giant",
            "categories": ["Software", "Operations", "Cloud", "Retail"]
        },
        {
            "name": "Apple",
            "icon": "fab fa-apple",
            "color": "#555555",
            "careers_url": "https://www.apple.com/careers",
            "description": "Innovation leader in consumer technology",
            "categories": ["Software", "Hardware", "Design", "AI/ML"]
        },
        {
            "name": "Facebook",
            "icon": "fab fa-facebook",
            "color": "#1877F2",
            "careers_url": "https://www.metacareers.com/",
            "description": "Social media and technology company",
            "categories": ["Software", "Marketing", "Networking", "AI/ML"]
        },
        {
            "name": "Netflix",
            "icon": "fas fa-play-circle",
            "color": "#E50914",
            "careers_url": "https://explore.jobs.netflix.net/careers",
            "description": "Streaming media company",
            "categories": ["Software", "Marketing", "Design", "Service"],
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Netflix_2015_logo.svg/1920px-Netflix_2015_logo.svg.png",
            "website": "https://jobs.netflix.com/",
            "industry": "Entertainment & Technology"
        }
    ],
    "indian_tech": [
        {
            "name": "TCS",
            "icon": "fas fa-building",
            "color": "#0070C0",
            "careers_url": "https://www.tcs.com/careers",
            "description": "India's largest IT services company",
            "categories": ["IT Services", "Consulting", "Digital"]
        },
        {
            "name": "Infosys",
            "icon": "fas fa-building",
            "color": "#007CC3",
            "careers_url": "https://www.infosys.com/careers",
            "description": "Global leader in digital services and consulting",
            "categories": ["IT Services", "Consulting", "Digital"]
        },
        {
            "name": "Wipro",
            "icon": "fas fa-building",
            "color": "#341F65",
            "careers_url": "https://careers.wipro.com",
            "description": "Leading global information technology company",
            "categories": ["IT Services", "Consulting", "Digital"]
        },
        {
            "name": "HCL",
            "icon": "fas fa-building",
            "color": "#0075C9",
            "careers_url": "https://www.hcltech.com/careers",
            "description": "Global technology company",
            "categories": ["IT Services", "Engineering", "Digital"]
        }
    ],
    "global_corps": [
        {
            "name": "IBM",
            "icon": "fas fa-server",
            "color": "#1F70C1",
            "careers_url": "https://www.ibm.com/careers",
            "description": "Global leader in technology and consulting",
            "categories": ["Software", "Consulting", "AI/ML", "Cloud"],
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/1920px-IBM_logo.svg.png",
            "website": "https://www.ibm.com/careers/",
            "industry": "Technology & Consulting"
        },
        {
            "name": "Accenture",
            "icon": "fas fa-building",
            "color": "#A100FF",
            "careers_url": "https://www.accenture.com/careers",
            "description": "Global professional services company",
            "categories": ["Consulting", "Technology", "Digital"]
        },
        {
            "name": "Cognizant",
            "icon": "fas fa-building",
            "color": "#1299D8",
            "careers_url": "https://careers.cognizant.com",
            "description": "Leading professional services company",
            "categories": ["IT Services", "Consulting", "Digital"]
        }
    ]
}

JOB_MARKET_INSIGHTS = {
    "trending_skills": [
        {"name": "Artificial Intelligence", "growth": "+45%", "icon": "fas fa-brain"},
        {"name": "Cloud Computing", "growth": "+38%", "icon": "fas fa-cloud"},
        {"name": "Data Science", "growth": "+35%", "icon": "fas fa-chart-line"},
        {"name": "Cybersecurity", "growth": "+32%", "icon": "fas fa-shield-alt"},
        {"name": "DevOps", "growth": "+30%", "icon": "fas fa-code-branch"},
        {"name": "Machine Learning", "growth": "+28%", "icon": "fas fa-robot"},
        {"name": "Blockchain", "growth": "+25%", "icon": "fas fa-lock"},
        {"name": "Big Data", "growth": "+23%", "icon": "fas fa-database"},
        {"name": "Internet of Things", "growth": "+21%", "icon": "fas fa-wifi"}
    ],
    "top_locations": [
        {"name": "Bangalore", "jobs": "50,000+", "icon": "fas fa-city"},
        {"name": "Mumbai", "jobs": "35,000+", "icon": "fas fa-city"},
        {"name": "Delhi NCR", "jobs": "30,000+", "icon": "fas fa-city"},
        {"name": "Hyderabad", "jobs": "25,000+", "icon": "fas fa-city"},
        {"name": "Pune", "jobs": "20,000+", "icon": "fas fa-city"},
        {"name": "Chennai", "jobs": "15,000+", "icon": "fas fa-city"},
        {"name": "Noida", "jobs": "10,000+", "icon": "fas fa-city"},
        {"name": "Vadodara", "jobs": "7,000+", "icon": "fas fa-city"},
        {"name": "Ahmedabad", "jobs": "6,000+", "icon": "fas fa-city"},
        {"name": "Remote", "jobs": "3,000+", "icon": "fas fa-globe-americas"},
    ],
    "salary_insights": [
        {"role": "Machine Learning Engineer", "range": "10-35 LPA", "experience": "0-5 years"},
        {"role": "Big Data Engineer", "range": "8-30 LPA", "experience": "0-5 years"},
        {"role": "Software Engineer", "range": "5-25 LPA", "experience": "0-5 years"},
        {"role": "Data Scientist", "range": "8-30 LPA", "experience": "0-5 years"},
        {"role": "DevOps Engineer", "range": "6-28 LPA", "experience": "0-5 years"},
        {"role": "UI/UX Designer", "range": "5-25 LPA", "experience": "0-5 years"},
        {"role": "Full Stack Developer", "range": "8-30 LPA", "experience": "0-5 years"},
        {"role": "C++/C#/Python/Java Developer", "range": "6-26 LPA", "experience": "0-5 years"},
        {"role": "Django Developer", "range": "7-27 LPA", "experience": "0-5 years"},
        {"role": "Cloud Engineer", "range": "6-26 LPA", "experience": "0-5 years"},
        {"role": "Google Cloud/AWS/Azure Engineer", "range": "6-26 LPA", "experience": "0-5 years"},
        {"role": "Salesforce Engineer", "range": "6-26 LPA", "experience": "0-5 years"},
    ]
}

def get_featured_companies(category=None):
    """Get featured companies, optionally filtered by category"""
    if category and category in FEATURED_COMPANIES:
        return FEATURED_COMPANIES[category]
    return [company for companies in FEATURED_COMPANIES.values() for company in companies]

def get_market_insights():
    """Get job market insights"""
    return JOB_MARKET_INSIGHTS

def get_company_info(company_name):
    """Get company information by name"""
    for companies in FEATURED_COMPANIES.values():
        for company in companies:
            if company["name"] == company_name:
                return company
    return None

def get_companies_by_industry(industry):
    """Get list of companies by industry"""
    companies = []
    for companies_list in FEATURED_COMPANIES.values():
        for company in companies_list:
            if "industry" in company and company["industry"] == industry:
                companies.append(company)
    return companies

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
# Sample job search function
import uuid
import urllib.parse

def search_jobs(job_role, location, experience_level=None, job_type=None, foundit_experience=None):
    # Encode inputs
    role_encoded = urllib.parse.quote_plus(job_role.strip())
    loc_encoded = urllib.parse.quote_plus(location.strip())

    # Mappings
    experience_range_map = {
        "Internship": "0~0", "Entry Level": "1~1", "Associate": "2~3",
        "Mid-Senior Level": "4~7", "Director": "8~15", "Executive": "16~20"
    }

    experience_exact_map = {
        "Internship": "0", "Entry Level": "1", "Associate": "2",
        "Mid-Senior Level": "4", "Director": "8", "Executive": "16"
    }

    linkedin_exp_map = {
        "Internship": "1", "Entry Level": "2", "Associate": "3",
        "Mid-Senior Level": "4", "Director": "5", "Executive": "6"
    }

    job_type_map = {
        "Full-time": "F", "Part-time": "P", "Contract": "C",
        "Temporary": "T", "Volunteer": "V", "Internship": "I"
    }

    # LinkedIn
    linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={role_encoded}&location={loc_encoded}"
    if experience_level in linkedin_exp_map:
        linkedin_url += f"&f_E={linkedin_exp_map[experience_level]}"
    if job_type in job_type_map:
        linkedin_url += f"&f_JT={job_type_map[job_type]}"

    # Naukri
    naukri_url = f"https://www.naukri.com/{role_encoded}-jobs-in-{loc_encoded}"

    # FoundIt
    if foundit_experience is not None:
        experience_range = f"{foundit_experience}~{foundit_experience}"
        experience_exact = str(foundit_experience)
    else:
        experience_range = experience_range_map.get(experience_level, "")
        experience_exact = experience_exact_map.get(experience_level, "")

    search_id = uuid.uuid4()
    foundit_url = f"https://www.foundit.in/srp/results?query={role_encoded}&locations={loc_encoded}"
    if experience_range:
        foundit_url += f"&experienceRanges={urllib.parse.quote_plus(experience_range)}"
    if experience_exact:
        foundit_url += f"&experience={experience_exact}"
    foundit_url += f"&searchId={search_id}"

    return [
        {"title": f"LinkedIn: {job_role} jobs in {location}", "link": linkedin_url},
        {"title": f"Naukri: {job_role} jobs in {location}", "link": naukri_url},
        {"title": f"FoundIt (Monster): {job_role} jobs in {location}", "link": foundit_url}
    ]




def add_hyperlink(paragraph, url, text, color="0000FF", underline=True):
    """
    A function to add a hyperlink to a paragraph.
    """
    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)

    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)

    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')

    # Color and underline
    if underline:
        u = OxmlElement('w:u')
        u.set(qn('w:val'), 'single')
        rPr.append(u)

    color_element = OxmlElement('w:color')
    color_element.set(qn('w:val'), color)
    rPr.append(color_element)

    new_run.append(rPr)

    text_elem = OxmlElement('w:t')
    text_elem.text = text
    new_run.append(text_elem)

    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink

def generate_docx(text, filename="bias_free_resume.docx"):
    doc = Document()
    doc.add_heading('Bias-Free Resume', 0)
    doc.add_paragraph(text)
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

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

def rewrite_text_with_llm(text, replacement_mapping, user_location):
    # Format the replacement mapping as a readable bullet list for the prompt
    formatted_mapping = "\n".join(
        [f"- \"{key}\" → \"{value}\"" for key, value in replacement_mapping.items()]
    )

    prompt = f"""
You are an expert career advisor and professional resume language editor.

Your task is to:

1. **Rewrite the following resume text** to:
   - Remove or replace any gender-coded, biased, or non-inclusive language.
   - Use *professional, inclusive, neutral, clear, and grammatically correct language*.
   - **Retain all technical terms, job-specific keywords, certifications, and proper names.**
   - Do **not** add new content or remove important information.
   - Preserve the original meaning and intent of each sentence.

---

2. **Structure and Organize** the rewritten resume into clearly labeled standard resume sections. Only include sections that are present in the original text:
   - Name
   - Contact Information
   - Email
   - Portfolio
   - Professional Summary
   - Work Experience
   - Skills
   - Certifications
   - Education
   - Projects
   - Interests

   - If *Name*, *Contact Information*, or *Email* is present, place them clearly at the top under respective headings.

---

3. **Strictly apply the following word replacement mapping:**

{formatted_mapping}

   - If a word or phrase matches a key exactly from this list, replace it with the corresponding value.
   - Leave all other content unchanged.

---

4. **Suggest 5 suitable job titles** based on the resume content and the candidate’s location: **{user_location}**
   - Ensure titles are realistic for this location and aligned with the candidate's experience and skills.
   - Provide a brief explanation for each suggestion.

---

5. **Provide LinkedIn job search URLs** for each suggested title based on the location: **{user_location}**

---

**Original Resume Text:**
\"\"\"{text}\"\"\"

---

**✅ Bias-Free Rewritten Resume (Well-Structured):**



---

**🎯 Suggested Job Titles with Explanations and LinkedIn URLs:**

1. **Job Title 1** — Reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%201&location={user_location})

2. **Job Title 2** — Reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%202&location={user_location})

3. **Job Title 3** — Reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%203&location={user_location})

4. **Job Title 4** — Reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%204&location={user_location})

5. **Job Title 5** — Reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%205&location={user_location})
"""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
    response = llm.invoke(prompt)
    return response.content

    
    



def rewrite_and_highlight(text, replacement_mapping, user_location):
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
    rewritten_text = rewrite_text_with_llm(text, replacement_mapping, user_location)



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

user_location = st.sidebar.text_input("📍 Preferred Job Location (City, Country)")


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
        highlighted_text, rewritten_text, masc_count, fem_count, detected_masc, detected_fem = rewrite_and_highlight(full_text, replacement_mapping, user_location)

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


# === TAB 1: Dashboard ===
# 📊 Dashboard and Metrics
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧾 Resume Builder", "💼 Job Search"])

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
        chart_tab1, chart_tab2 = st.tabs(["📉 Bias Score Chart", "⚖ Gender-Coded Words"])

        with chart_tab1:
            st.subheader("Bias Score Comparison Across Resumes")
            st.bar_chart(df.set_index("Resume Name")[["Bias Score (0 = Fair, 1 = Biased)"]])

        with chart_tab2:
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

                    docx_file = generate_docx(resume["Rewritten Text"])

                    st.download_button(
                    label="📥 Download Bias-Free Resume (.docx)",
                    data=docx_file,
                    file_name=f"{resume['Resume Name'].split('.')[0]}_bias_free.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
)
    else:
        st.warning("Please upload resumes to view dashboard analytics.")
with tab2:
    st.markdown("## 🧾 <span style='color:#336699;'>Advanced Resume Builder</span>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #bbb;'>", unsafe_allow_html=True)

    # Initialize session state
    fields = ["name", "email", "phone", "linkedin", "location", "portfolio", "summary", "skills", "languages", "interests","Softskills"]
    for f in fields:
        st.session_state.setdefault(f, "")
    st.session_state.setdefault("experience_entries", [{"title": "", "company": "", "duration": "", "description": ""}])
    st.session_state.setdefault("education_entries", [{"degree": "", "institution": "", "year": "", "details": ""}])
    st.session_state.setdefault("project_entries", [{"title": "", "tech": "", "duration": "", "description": ""}])
    st.session_state.setdefault("project_links", [])
    st.session_state.setdefault("certificate_links", [{"name": "", "link": "", "duration": "", "description": ""}])
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ➕ Add More Sections")
        if st.button("➕ Add Experience"):
            st.session_state.experience_entries.append({"title": "", "company": "", "duration": "", "description": ""})
        if st.button("➕ Add Education"):
            st.session_state.education_entries.append({"degree": "", "institution": "", "year": "", "details": ""})
        if st.button("➕ Add Project"):
            st.session_state.project_entries.append({"title": "", "tech": "", "duration": "", "description": ""})
        if st.button("➕ Add Certificate"):
           st.session_state.certificate_links.append({"name": "", "link": "", "duration": "", "description": ""})


    with st.form("resume_form"):
        st.markdown("### 👤 <u>Personal Information</u>", unsafe_allow_html=True)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("👤 Full Name ", key="name")
                st.text_input("📞 Phone Number", key="phone")
                st.text_input("📍 Location", key="location")
            with col2:
                st.text_input("📧 Email", key="email")
                st.text_input("🔗 LinkedIn", key="linkedin")
                st.text_input("🌐 Portfolio", key="portfolio")
                st.text_input("💼 Job Title", key="job_title")


        st.markdown("### 📝 <u>Professional Summary</u>", unsafe_allow_html=True)
        st.text_area("Summary", key="summary")

        st.markdown("### 💼 <u>Skills, Languages, Interests & Soft Skills</u>", unsafe_allow_html=True)
        st.text_area("Skills (comma-separated)", key="skills")
        st.text_area("Languages (comma-separated)", key="languages")
        st.text_area("Interests (comma-separated)", key="interests")
        st.text_area("Softskills (comma-separated)", key="Softskills")


        st.markdown("### 🧱 <u>Work Experience</u>", unsafe_allow_html=True)
        for idx, exp in enumerate(st.session_state.experience_entries):
            with st.expander(f"Experience #{idx+1}", expanded=True):
                exp["title"] = st.text_input(f"Job Title", value=exp["title"], key=f"title_{idx}")
                exp["company"] = st.text_input(f"Company", value=exp["company"], key=f"company_{idx}")
                exp["duration"] = st.text_input(f"Duration", value=exp["duration"], key=f"duration_{idx}")
                exp["description"] = st.text_area(f"Description", value=exp["description"], key=f"description_{idx}")

        st.markdown("### 🎓 <u>Education</u>", unsafe_allow_html=True)
        for idx, edu in enumerate(st.session_state.education_entries):
            with st.expander(f"Education #{idx+1}", expanded=True):
                edu["degree"] = st.text_input(f"Degree", value=edu["degree"], key=f"degree_{idx}")
                edu["institution"] = st.text_input(f"Institution", value=edu["institution"], key=f"institution_{idx}")
                edu["year"] = st.text_input(f"Year", value=edu["year"], key=f"edu_year_{idx}")
                edu["details"] = st.text_area(f"Details", value=edu["details"], key=f"edu_details_{idx}")

        st.markdown("### 🛠 <u>Projects</u>", unsafe_allow_html=True)
        for idx, proj in enumerate(st.session_state.project_entries):
            with st.expander(f"Project #{idx+1}", expanded=True):
                proj["title"] = st.text_input(f"Project Title", value=proj["title"], key=f"proj_title_{idx}")
                proj["tech"] = st.text_input(f"Tech Stack", value=proj["tech"], key=f"proj_tech_{idx}")
                proj["duration"] = st.text_input(f"Duration", value=proj["duration"], key=f"proj_duration_{idx}")
                proj["description"] = st.text_area(f"Description", value=proj["description"], key=f"proj_desc_{idx}")


        st.markdown("### 🔗 Project Links")
        project_links_input = st.text_area("Enter one project link per line:")
        if project_links_input:
            st.session_state.project_links = [link.strip() for link in project_links_input.splitlines() if link.strip()]

        st.markdown("### 🧾 <u>Certificates</u>", unsafe_allow_html=True)
        for idx, cert in enumerate(st.session_state.certificate_links):
            with st.expander(f"Certificate #{idx+1}", expanded=True):
                cert["name"] = st.text_input(f"Certificate Name", value=cert["name"], key=f"cert_name_{idx}")
                cert["link"] = st.text_input(f"Certificate Link", value=cert["link"], key=f"cert_link_{idx}")
                cert["duration"] = st.text_input(f"Duration", value=cert["duration"], key=f"cert_duration_{idx}")
                cert["description"] = st.text_area(f"Description", value=cert["description"], key=f"cert_description_{idx}")


        submitted = st.form_submit_button("📑 Generate Resume")

    
        if submitted:
         st.success("✅ Resume Generated Successfully! Scroll down to preview or download.")

        st.markdown("""
    <style>
        .heading-large {
            font-size: 36px;
            font-weight: bold;
            color: #336699;
        }
        .subheading-large {
            font-size: 30px;
            font-weight: bold;
            color: #336699;
        }
        .tab-section {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)
 


    # --- Visual Resume Preview Section ---
        st.markdown("## 🧾 <span style='color:#336699;'>Resume Preview</span>", unsafe_allow_html=True)
        st.markdown("<hr style='border-top: 2px solid #bbb;'>", unsafe_allow_html=True)

        left, right = st.columns([1, 2])

        with left:
            st.markdown(f"""
                <h2 style='color:#2f2f2f;margin-bottom:0;'>{st.session_state['name']}</h2>
                <h4 style='margin-top:5px;color:#444;'>{st.session_state['job_title']}</h4>

                <p style='font-size:14px;'>
                📍 {st.session_state['location']}<br>
                📞 {st.session_state['phone']}<br>
                📧 <a href="mailto:{st.session_state['email']}">{st.session_state['email']}</a><br>
                🔗 <a href="{st.session_state['linkedin']}" target="_blank">LinkedIn</a><br>
                🌐 <a href="{st.session_state['portfolio']}" target="_blank">Portfolio</a>
                </p>
            """, unsafe_allow_html=True)

            st.markdown("<h4 style='color:#336699;'>Skills</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for skill in [s.strip() for s in st.session_state["skills"].split(",") if s.strip()]:
                st.markdown(f"<div style='margin-left:10px;'>• {skill}</div>", unsafe_allow_html=True)

            st.markdown("<h4 style='color:#336699;'>Languages</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for lang in [l.strip() for l in st.session_state["languages"].split(",") if l.strip()]:
               st.markdown(f"<div style='margin-left:10px;'>• {lang}</div>", unsafe_allow_html=True)

            st.markdown("<h4 style='color:#336699;'>Interests</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for interest in [i.strip() for i in st.session_state["interests"].split(",") if i.strip()]:
               st.markdown(f"<div style='margin-left:10px;'>• {interest}</div>", unsafe_allow_html=True)

            st.markdown("<h4 style='color:#336699;'>Softskills</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for Softskills  in [i.strip() for i in st.session_state["Softskills"].split(",") if i.strip()]:
               st.markdown(f"<div style='margin-left:10px;'>• {Softskills}</div>", unsafe_allow_html=True)   


        with right:
            st.markdown("<h4 style='color:#336699;'>Summary</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            summary_text = st.session_state['summary'].replace('\n', '<br>')
            st.markdown(f"<p style='font-size:14px;'>{summary_text}</p>", unsafe_allow_html=True)


            st.markdown("<h4 style='color:#336699;'>Experience</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for exp in st.session_state.experience_entries:
             if exp["company"] or exp["title"]:
              st.markdown(f"""
            <div style='margin-bottom:15px; padding:10px; border-radius:8px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <b>🏢 {exp['company']}</b><span style='color:gray;'>⏳ {exp['duration']}</span>
                </div>
                <div style='font-size:14px;'>💼 <i>{exp['title']}</i></div>
                <div style='font-size:14px;'>📝 {exp['description']}</div>
            </div>
        """, unsafe_allow_html=True)


            st.markdown("<h4 style='color:#336699;'>🎓 Education</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for edu in st.session_state.education_entries:
             if edu["institution"] or edu["degree"]:
              st.markdown(f"""
            <div style='margin-bottom: 15px; padding: 10px 15px;color: white; border-radius: 8px;'>
                <div style='display: flex; justify-content: space-between; font-size: 16px; font-weight: bold;'>
                    <span>🏫 {edu['institution']}</span>
                    <span style='color: gray;'>📅 {edu['year']}</span>
                </div>
                <div style='font-size: 14px; margin-top: 5px;'>🎓 <i>{edu['degree']}</i></div>
                <div style='font-size: 14px;'>📄 {edu['details']}</div>
            </div>
        """, unsafe_allow_html=True)


            st.markdown("<h4 style='color:#336699;'>Projects</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for proj in st.session_state.project_entries:
             st.markdown(f"""
        <div style='margin-bottom:15px; padding: 10px;'>
        <strong style='font-size:16px;'>{proj['title']}</strong><br>
        <span style='font-size:14px;'>🛠️ <strong>Tech Stack:</strong> {proj['tech']}</span><br>
        <span style='font-size:14px;'>⏳ <strong>Duration:</strong> {proj['duration']}</span><br>
        <span style='font-size:14px;'>📝 <strong>Description:</strong> {proj['description']}</span>
    </div>
    """, unsafe_allow_html=True)



        if st.session_state.project_links:
                st.markdown("<h4 style='color:#336699;'>Project Links</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
                for i, link in enumerate(st.session_state.project_links):
                    st.markdown(f"[🔗 Project {i+1}]({link})", unsafe_allow_html=True)

        if st.session_state.certificate_links:
                st.markdown("<h4 style='color:#336699;'>Certificates</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
                
                for cert in st.session_state.certificate_links:
                    if cert["name"] and cert["link"]:
                      st.markdown(f"""
            <div style='display:flex; justify-content:space-between;'>
                <a href="{cert['link']}" target="_blank"><b>📄 {cert['name']}</b></a><span style='color:gray;'>{cert['duration']}</span>
            </div>
            <div style='margin-bottom:10px; font-size:14px;'>{cert['description']}</div>
        """, unsafe_allow_html=True)

# SKILLS
skills_html = "".join(
    f"<div class='skill-list'>• {s.strip()}</div>"
    for s in st.session_state['skills'].split(',')
    if s.strip()
)

languages_html = "".join(
    f"<div class='skill-list'>• {l.strip()}</div>"
    for l in st.session_state['languages'].split(',')
    if l.strip()
)

# INTERESTS
interests_html = "".join(
    f"<div class='skill-list'>• {i.strip()}</div>"
    for i in st.session_state['interests'].split(',')
    if i.strip()
)

Softskills_html = "".join(
    f"<div class='skill-list'>• {i.strip()}</div>"
    for i in st.session_state['Softskills'].split(',')
    if i.strip()
)

# EXPERIENCE
experience_html = "".join(
    f"""
    <div class='entry' style='margin-bottom: 15px; padding: 10px;'>
        <div class='entry-header' style='display: flex; justify-content: space-between; font-weight: bold; font-size: 16px;'>
            🏢 {exp['company']} <span style='color: gray;'>⏳ {exp['duration']}</span>
        </div>
        <div class='entry-title' style='font-size: 14px; margin-top: 4px;'>💼 <i>{exp['title']}</i></div>
        <div style='font-size: 14px; margin-top: 4px;'>📝 {exp['description']}</div>
    </div>
    """
    for exp in st.session_state.experience_entries
    if exp["company"] or exp["title"]
)

# Convert experience to list if multiple lines



summary_html = st.session_state['summary'].replace('\n', '<br>')

# EDUCATION


education_html = "".join(
    f"""
    <div class='entry' style='margin-bottom: 15px; padding: 0;'>
        <div class='entry-header' style='display: flex; justify-content: space-between; align-items: center; font-size: 16px; font-weight: bold; color: #000; margin-bottom: 5px;'>
            <span>🏫 {edu['institution']}</span>
            <span style='color:#333; font-weight: normal;'>🗓️ {edu.get('year', '')}</span>
        </div>
        {f"<div style='font-size:14px; color:#333; margin-bottom: 4px;'>🎓 <b>{', '.join(edu['degree']) if isinstance(edu.get('degree'), list) else edu.get('degree', '')}</b></div>" if edu.get('degree') else ''}
        <div style='font-size:14px; color:#333;'>📝 <i>{edu['details']}</i></div>
    </div>
    """
    for edu in st.session_state.education_entries
    if edu.get("institution") or edu.get("details")
)


# PROJECTS
# PROJECTS
projects_html = "".join(
    f"""
    <div class='entry' style='margin-bottom: 15px; padding: 10px; border-radius: 8px;'>
        <div class='entry-header' style='font-size: 16px; font-weight: bold; color: #333; margin-bottom: 5px;'>
            💻 {proj['title']} <span style='color:#333; font-weight: normal;'> ⏳ {proj.get('duration','')}</span>
        </div>
        {f"<div style='font-size:14px; color:#333; margin-bottom: 4px;'><b>🛠️ Technologies:</b> {', '.join(proj['tech']) if isinstance(proj.get('tech'), list) else proj.get('tech', '')}</div>" if proj.get('tech') else ''}
        <div style='font-size:14px; color:#333;'>
            <b>📝 Description:</b>
            <ul style='margin-top: 5px; padding-left: 20px;'>
                {"".join(f"<li>{line.strip()}</li>" for line in proj['description'].splitlines() if line.strip())}
            </ul>
        </div>
    </div>
    """
    for proj in st.session_state.project_entries
    if proj.get("title") or proj.get("description")
)





# PROJECT LINKS
project_links_html = ""
if st.session_state.project_links:
    project_links_html = "<h4 class='section-title'>Project Links</h4><hr>" + "".join(
        f'<p><a href="{link}">🔗 Project {i+1}</a></p>'
        for i, link in enumerate(st.session_state.project_links)
    )



# CERTIFICATES
certificate_links_html = ""
if st.session_state.certificate_links:
    certificate_links_html = "<h4 class='section-title'>Certificates</h4><hr>" + "".join(
        f"""
        <div class='entry'>
            <div class='entry-header'>
                <b><a href="{cert['link']}" target="_blank">📄 {cert['name']}</a></b>
                <span style='color:gray;'> {cert.get('duration', '')}</span>
            </div>
            <div class='entry-title' style='font-size:14px;'>{cert.get('description', '')}</div>
        </div>
        """
        for cert in st.session_state.certificate_links
        if cert["name"] and cert["link"]
    )

        # --- Word Export Logic (Unchanged from your code) ---
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{st.session_state['name']} - Resume</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            color: #2f2f2f;
        }}
        h2 {{
            font-size: 32px;
            margin: 0;
            color: #336699;
        }}
        h4 {{
            font-size: 24px;
            margin: 0;
            color: #336699;
        }}
        a {{
            color: #007acc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: none;
            border-top: 2px solid #bbb;
            margin: 20px 0;
        }}
        .container {{
            display: flex;
            gap: 40px;
        }}
        .left {{
            flex: 1;
            border-right: 2px solid #ccc;
            padding-right: 20px;
        }}
        .right {{
            flex: 2;
            padding-left: 20px;
        }}
        .section-title {{
            color: #336699;
            margin-top: 30px;
            margin-bottom: 5px;
        }}
        .skill-list {{
            margin-left: 10px;
        }}
        .entry {{
            margin-bottom: 15px;
        }}
        .entry-header {{
            display: flex;
            justify-content: space-between;
        }}
        .entry-title {{
            font-style: italic;
        }}
    </style>
</head>
<body>

    <h2>{st.session_state['name']}</h2>
    <h4>{st.session_state['job_title']}</h4>
    <hr>

    <div class="container">
        <div class="left">
            <p>
                📍 {st.session_state['location']}<br>
                📞 {st.session_state['phone']}<br>
                📧 <a href="mailto:{st.session_state['email']}">{st.session_state['email']}</a><br>
                🔗 <a href="{st.session_state['linkedin']}">LinkedIn</a><br>
                🌐 <a href="{st.session_state['portfolio']}">Portfolio</a>
            </p>

            <h4 class="section-title">Skills</h4>
            <hr>
            {skills_html}

            <h4 class="section-title">Languages</h4>
            <hr>
            {languages_html}

            <h4 class="section-title">Interests</h4>
            <hr>
            {interests_html}

            <h4 class="section-title">Softskills</h4>
            <hr>
            {Softskills_html}
        </div>

        <div class="right">
            <h4 class="section-title">Summary</h4>
            <hr>
            <p>{summary_html}</p>

            <h4 class="section-title">Experience</h4>
            <hr>
            {experience_html}

            <h4 class="section-title">Education</h4>
            <hr>
            {education_html}

            <h4 class="section-title">Projects</h4>
            <hr>
            {projects_html}

            {project_links_html}
            {certificate_links_html}
        </div>
    </div>
</body>
</html>
"""

# Then encode it to bytes and prepare for download
html_bytes = html_content.encode("utf-8")
html_file = BytesIO(html_bytes)


with tab2:
 st.download_button (
    label="📥 Download Resume (HTML)",
    data=html_file,
    file_name=f"{st.session_state['name'].replace(' ', '_')}_Resume.html",
    mime="text/html"
)    
with tab2:
 st.markdown("""
✅ After downloading your HTML resume, you can [click here to convert it to PDF](https://www.sejda.com/html-to-pdf) using Sejda's free online tool.
""")

with tab3:
    st.header("🔍 Job Search Across LinkedIn, Naukri, and FoundIt")

    col1, col2 = st.columns(2)

    with col1:
        job_role = st.text_input("💼 Desired Job Role", placeholder="e.g., Data Scientist")
        experience_level = st.selectbox(
            "📈 Experience Level",
            ["", "Internship", "Entry Level", "Associate", "Mid-Senior Level", "Director", "Executive"]
        )

    with col2:
        location = st.text_input("📍 Preferred Location", placeholder="e.g., Bangalore, India")
        job_type = st.selectbox(
            "📋 Job Type",
            ["", "Full-time", "Part-time", "Contract", "Temporary", "Volunteer", "Internship"]
        )

    foundit_experience = st.text_input("🔢 Experience (Years) for FoundIt", placeholder="e.g., 1")

    search_clicked = st.button("🔎 Search Jobs")

    if search_clicked:
        if job_role.strip() and location.strip():
            results = search_jobs(job_role, location, experience_level, job_type, foundit_experience)

            st.markdown("## 🎯 Job Search Results")

            for job in results:
                platform = job["title"].split(":")[0].strip().lower()

                if platform == "linkedin":
                    icon = "🔵 <b style='color:#0e76a8;'>in LinkedIn</b>"
                    btn_color = "#0e76a8"
                elif platform == "naukri":
                    icon = "🏢 <b style='color:#ff5722;'>Naukri</b>"
                    btn_color = "#ff5722"
                elif "foundit" in platform:
                    icon = "🌐 <b style='color:#7c4dff;'>Foundit (Monster)</b>"
                    btn_color = "#7c4dff"
                else:
                    icon = f"📄 <b>{platform.title()}</b>"
                    btn_color = "#00c4cc"

                st.markdown(f"""
<div style="
    background-color:#1e1e1e;
    padding:20px;
    border-radius:15px;
    margin-bottom:20px;
    border-left: 5px solid {btn_color};
    box-shadow: 0 0 15px {btn_color};
">
    <div style="font-size:20px; margin-bottom:8px;">{icon}</div>
    <div style="color:#ffffff; font-size:17px; margin-bottom:15px;">
        {job['title'].split(':')[1].strip()}
    </div>
    <a href="{job['link']}" target="_blank" style="text-decoration:none;">
        <button style="
            background-color:{btn_color};
            color:white;
            padding:10px 15px;
            border:none;
            border-radius:8px;
            font-size:15px;
            cursor:pointer;
            box-shadow: 0 0 10px {btn_color};
        ">
            🚀 View Jobs on {platform.title()} &rarr;
        </button>
    </a>
</div>
""", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter both the Job Role and Location to perform the search.")


    # Inject Glowing CSS for Cards
    st.markdown("""
    <style>
    @keyframes glow {
        0% {
            box-shadow: 0 0 5px rgba(255,255,255,0.2);
        }
        50% {
            box-shadow: 0 0 20px rgba(0,255,255,0.6);
        }
        100% {
            box-shadow: 0 0 5px rgba(255,255,255,0.2);
        }
    }

    .company-card {
        background-color: #1e1e1e;
        color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        cursor: pointer;
        text-decoration: none;
        display: block;
        animation: glow 3s infinite alternate;
    }

    .company-card:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.7), 0 0 10px rgba(0, 255, 255, 0.5);
    }

    .pill {
        display: inline-block;
        background-color: #333;
        padding: 6px 12px;
        border-radius: 999px;
        margin: 4px 6px 0 0;
        font-size: 13px;
    }

    .title-header {
        color: white;
        font-size: 26px;
        margin-top: 40px;
        font-weight: bold;
    }

    .company-logo {
        font-size: 26px;
        margin-right: 8px;
    }

    .company-header {
        font-size: 22px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Featured Companies ----------
    st.markdown("### <div class='title-header'>🏢 Featured Companies</div>", unsafe_allow_html=True)

    selected_category = st.selectbox("📂 Browse Featured Companies By Category", ["All", "tech", "indian_tech", "global_corps"])
    companies_to_show = get_featured_companies() if selected_category == "All" else get_featured_companies(selected_category)

    for company in companies_to_show:
        category_tags = ''.join([f"<span class='pill'>{cat}</span>" for cat in company['categories']])
        st.markdown(f"""
        <a href="{company['careers_url']}" class="company-card" target="_blank">
            <div class="company-header">
                <span class="company-logo">{company.get('emoji', '🏢')}</span>
                {company['name']}
            </div>
            <p>{company['description']}</p>
            {category_tags}
        </a>
        """, unsafe_allow_html=True)

    # ---------- Market Insights ----------
    st.markdown("### <div class='title-header'>📈 Job Market Trends</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚀 Trending Skills")
        for skill in JOB_MARKET_INSIGHTS["trending_skills"]:
            st.markdown(f"""
            <div class="company-card">
                <h4>🔧 {skill['name']}</h4>
                <p>📈 Growth Rate: {skill['growth']}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🌍 Top Job Locations")
        for loc in JOB_MARKET_INSIGHTS["top_locations"]:
            st.markdown(f"""
            <div class="company-card">
                <h4>📍 {loc['name']}</h4>
                <p>💼 Openings: {loc['jobs']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------- Salary Insights ----------
    st.markdown("### <div class='title-header'>💰 Salary Insights</div>", unsafe_allow_html=True)
    for role in JOB_MARKET_INSIGHTS["salary_insights"]:
        st.markdown(f"""
        <div class="company-card">
            <h4>💼 {role['role']}</h4>
            <p>📅 Experience: {role['experience']}</p>
            <p>💵 Salary Range: {role['range']}</p>
        </div>
        """, unsafe_allow_html=True)





        
# 💬 Chat Section
# 1. Display existing chat history
if "memory" in st.session_state:
    history = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
    for msg in history:
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)

# 2. Wait for user input
user_input = st.chat_input("Ask LEXIBOT anything...")

# 3. Only call chain when user submits new input
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # 🧠 Only call chain ONCE
        response = st.session_state.chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.memory.chat_memory.messages
        })
        answer = response.get("answer", "❌ No answer found.")
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save interaction to memory
    st.session_state.memory.save_context({"input": user_input}, {"output": answer})
