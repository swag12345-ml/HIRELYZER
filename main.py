import pdfkit
from io import BytesIO

def html_to_pdf_bytes(html_string):
    path_to_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

    options = {
        'page-width': '400mm',     # Increased width for more horizontal space
        'page-height': '297mm',    # Keep standard A4 height
        'encoding': "UTF-8",
        'enable-local-file-access': None,
        'margin-top': '10mm',
        'margin-bottom': '10mm',
        'margin-left': '10mm',
        'margin-right': '10mm',
        'zoom': '1',               # Keep at 1 for original scale
        'disable-smart-shrinking': '',
    }

    pdf_bytes = pdfkit.from_string(html_string, False, options=options, configuration=config)
    return BytesIO(pdf_bytes)



    pdf_bytes = pdfkit.from_string(html_string, False, options=options, configuration=config)
    return BytesIO(pdf_bytes)

def generate_cover_letter_from_resume_builder():
    from datetime import datetime

    name = st.session_state.get("name", "")
    job_title = st.session_state.get("job_title", "")
    summary = st.session_state.get("summary", "")
    skills = st.session_state.get("skills", "")
    location = st.session_state.get("location", "")
    today_date = datetime.today().strftime("%B %d, %Y")
    company = st.text_input("🏢 Target Company", placeholder="e.g., Google")

    if not all([name, job_title, summary, skills, company]):
        st.warning("⚠️ Please fill in all resume fields and company name.")
        return

    prompt = f"""
You are a professional cover letter writer.

Write a formal and compelling cover letter using the information below. Format it as a real letter with:
1. Date
2. Recipient heading
3. Proper salutation
4. Three short paragraphs
5. Professional closing

Ensure you **only include the company name once** in the header or salutation, and avoid repeating it redundantly in the body.

### Heading Info:
{today_date}
Hiring Manager, {company}, {location}

### Candidate Info:
- Name: {name}
- Job Title: {job_title}
- Summary: {summary}
- Skills: {skills}
- Location: {location}

### Instructions:
- Do not repeat the company name twice.
- Focus on skills and impact.
- Make it personalized and enthusiastic.

Return only the final formatted cover letter.
"""

    cover_letter = call_llm(prompt, session=st.session_state)
    st.session_state["cover_letter"] = cover_letter

    # ✅ Styled HTML template wrapping the LLM output
    cover_letter_html = f"""
    <div style="font-family: Georgia, serif; line-height: 1.6; color: #333; border: 1px solid #ccc; padding: 20px;">
        <h1 style="color: #003366; font-size: 28px; margin-bottom: 0;">{name}</h1>
        <h2 style="font-style: italic; font-weight: normal; margin-top: 0; color: #555;">{job_title}</h2>
        <p style="margin: 4px 0;">
            {location}<br>
            {today_date}<br>
            LinkedIn: <a href="#" style="color: #003366;">Your LinkedIn URL</a>
        </p>
        <hr style="border: 1px solid #ccc;">

        <div style="white-space: pre-wrap;">{cover_letter}</div>
    </div>
    """

    st.markdown(cover_letter_html, unsafe_allow_html=True)



import streamlit as st
import streamlit.components.v1 as components
from base64 import b64encode
import streamlit as st
import re
from llm_manager import call_llm
import requests
import datetime

from user_login import (
    create_user_table,
    add_user,
    verify_user,
    get_logins_today,
    get_total_registered_users,
    log_user_action
)

# ------------------- Initialize -------------------
create_user_table()

# ------------------- Initialize Session State -------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ------------------- CSS Styling -------------------
st.markdown("""
<style>
body, .main {
    background-color: #0d1117;
    color: white;
}
.login-card {
    background: #161b22;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
    transition: all 0.4s ease;
}
.login-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 0 45px rgba(0,255,255,0.25);
}
.stTextInput > div > input {
    background-color: #0d1117;
    color: white;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.6em;
}
.stTextInput > div > input:hover {
    border: 1px solid #00BFFF;
    box-shadow: 0 0 8px rgba(0,191,255,0.2);
}
.stTextInput > label {
    color: #c9d1d9;
}
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #2ea043;
    box-shadow: 0 0 10px rgba(46,160,67,0.4);
    transform: scale(1.02);
}
.feature-card {
    background: radial-gradient(circle at top left, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,255,255,0.1);
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color: #fff;
    margin-bottom: 20px;
}
.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 0 30px rgba(0,255,255,0.4);
}
.feature-card h3 {
    color: #00BFFF;
}
.feature-card p {
    color: #c9d1d9;
}
</style>
""", unsafe_allow_html=True)

# ------------------- BEFORE LOGIN -------------------
if not st.session_state.authenticated:

    # -------- Sidebar --------
    with st.sidebar:
        st.markdown("<h1 style='color:#00BFFF;'>Smart Resume AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#c9d1d9;'>Transform your career with AI-powered resume analysis, job matching, and smart insights.</p>", unsafe_allow_html=True)

        features = [
            ("https://img.icons8.com/fluency/48/resume.png", "Resume Analyzer", "Get feedback, scores, and tips powered by AI along with the biased words detection and rewriting the resume in an inclusive way."),
            ("https://img.icons8.com/fluency/48/resume-website.png", "Resume Builder", "Build modern, eye-catching resumes easily."),
            ("https://img.icons8.com/fluency/48/job.png", "Job Search", "Find tailored job matches."),
            ("https://img.icons8.com/fluency/48/classroom.png", "Course Suggestions", "Get upskilling recommendations based on your goals."),
            ("https://img.icons8.com/fluency/48/combo-chart.png", "Interactive Dashboard", "Visualize trends, scores, and analytics."),
        ]

        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-card">
                <img src="{icon}" width="40"/>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # -------- Animated Cards --------
    image_url = "https://cdn-icons-png.flaticon.com/512/3135/3135768.png"
    response = requests.get(image_url)
    img_base64 = b64encode(response.content).decode()

    st.markdown(f"""
    <style>
    .animated-cards {{
      margin-top: 30px;
      display: flex;
      justify-content: center;
      position: relative;
      height: 300px;
    }}
    .animated-cards img {{
      position: absolute;
      width: 240px;
      animation: splitCards 2.5s ease-in-out infinite alternate;
      z-index: 1;
    }}
    .animated-cards img:nth-child(1) {{ animation-delay: 0s; z-index: 3; }}
    .animated-cards img:nth-child(2) {{ animation-delay: 0.3s; z-index: 2; }}
    .animated-cards img:nth-child(3) {{ animation-delay: 0.6s; z-index: 1; }}
    @keyframes splitCards {{
      0% {{ transform: scale(1) translateX(0) rotate(0deg); opacity: 1; }}
      100% {{ transform: scale(1) translateX(var(--x-offset)) rotate(var(--rot)); opacity: 1; }}
    }}
    .card-left {{ --x-offset: -80px; --rot: -5deg; }}
    .card-center {{ --x-offset: 0px; --rot: 0deg; }}
    .card-right {{ --x-offset: 80px; --rot: 5deg; }}
    </style>
    <div class="animated-cards">
        <img class="card-left" src="data:image/png;base64,{img_base64}" />
        <img class="card-center" src="data:image/png;base64,{img_base64}" />
        <img class="card-right" src="data:image/png;base64,{img_base64}" />
    </div>
    """, unsafe_allow_html=True)

    # -------- Counter Section --------
    total_users = get_total_registered_users()
    active_logins = get_logins_today()
    resumes_uploaded = 431
    states_accessed = 29

    components.html(f"""
    <style>
    .counter-wrapper {{
        display: flex; flex-wrap: wrap; justify-content: center; gap: 30px; margin-top: 40px;
    }}
    .counter {{
        width: 230px; height: 140px;
        background: linear-gradient(145deg, #0d1117, #0d1117);
        color: #00BFFF;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0, 191, 255, 0.4);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        transition: transform 0.3s ease;
        border: 2px solid transparent;
        border-image: linear-gradient(to right, #00BFFF, #00FFFF) 1;
    }}
    .counter:hover {{
        transform: translateY(-6px);
        box-shadow: 0 0 25px rgba(0,255,255,0.5);
    }}
    .counter h1 {{ font-size: 2.8em; margin: 0; }}
    .counter p {{ margin: 5px 0 0; font-size: 1.1em; color: #c9d1d9; }}
    </style>
    <div class="counter-wrapper">
        <div class="counter"><h1 id="totalUsers">0</h1><p>Total Users</p></div>
        <div class="counter"><h1 id="states">0</h1><p>States Accessed</p></div>
        <div class="counter"><h1 id="resumes">0</h1><p>Resumes Uploaded</p></div>
        <div class="counter"><h1 id="activeSessions">0</h1><p>Active Sessions</p></div>
    </div>
    <script>
    function animateValue(id, start, end, duration) {{
        const obj = document.getElementById(id);
        const range = end - start;
        const increment = end > start ? 1 : -1;
        const stepTime = Math.abs(Math.floor(duration / range));
        let current = start;
        const timer = setInterval(() => {{
            current += increment;
            obj.innerHTML = current;
            if (current == end) clearInterval(timer);
        }}, stepTime);
    }}
    animateValue("totalUsers", 0, {total_users}, 1500);
    animateValue("states", 0, {states_accessed}, 1200);
    animateValue("resumes", 0, {resumes_uploaded}, 1300);
    animateValue("activeSessions", 0, {active_logins}, 1500);
    </script>
    """, height=400)

if not st.session_state.authenticated:
    from base64 import b64encode
    import requests

    # ✅ Use an online image of a female employee
    image_url = "https://cdn-icons-png.flaticon.com/512/4140/4140047.png"
    response = requests.get(image_url)
    img_base64 = b64encode(response.content).decode()

    # ✅ Inject animated shuffle CSS + HTML
    st.markdown(f"""
    <style>
    .animated-cards {{
      margin-top: 40px;
      display: flex;
      justify-content: center;
      position: relative;
      height: 260px;
    }}
    .animated-cards img {{
      position: absolute;
      width: 220px;
      animation: splitCards 2.5s ease-in-out infinite alternate;
      z-index: 1;
    }}
    .animated-cards img:nth-child(1) {{
      animation-delay: 0s;
      z-index: 3;
    }}
    .animated-cards img:nth-child(2) {{
      animation-delay: 0.3s;
      z-index: 2;
    }}
    .animated-cards img:nth-child(3) {{
      animation-delay: 0.6s;
      z-index: 1;
    }}
    @keyframes splitCards {{
      0% {{
        transform: scale(1) translateX(0) rotate(0deg);
        opacity: 1;
      }}
      100% {{
        transform: scale(1) translateX(var(--x-offset)) rotate(var(--rot));
        opacity: 1;
      }}
    }}
    .card-left {{ --x-offset: -80px; --rot: -4deg; }}
    .card-center {{ --x-offset: 0px; --rot: 0deg; }}
    .card-right {{ --x-offset: 80px; --rot: 4deg; }}
    </style>

    <div class="animated-cards">
        <img class="card-left" src="data:image/png;base64,{img_base64}" />
        <img class="card-center" src="data:image/png;base64,{img_base64}" />
        <img class="card-right" src="data:image/png;base64,{img_base64}" />
    </div>
    """, unsafe_allow_html=True)

    # -------- Login/Register Layout --------
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("<div class='login-card'><h2 style='text-align:center;'>🔐 Login to <span style='color:#00BFFF;'>LEXIBOT</span></h2>", unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(["🔑 Login", "🆕 Register"])

        with login_tab:
            user = st.text_input("Username", key="login_user")
            pwd = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="login_btn"):
                if verify_user(user.strip(), pwd.strip()):
                    st.session_state.authenticated = True
                    st.session_state.username = user.strip()
                    log_user_action(user.strip(), "login")
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")

        with register_tab:
            new_user = st.text_input("Choose a Username", key="reg_user")
            new_pass = st.text_input("Choose a Password", type="password", key="reg_pass")
            if st.button("Register", key="register_btn"):
                if new_user.strip() and new_pass.strip():
                    if add_user(new_user.strip(), new_pass.strip()):
                        st.success("✅ Registered! You can now login.")
                        log_user_action(new_user.strip(), "register")
                    else:
                        st.error("🚫 Username already exists.")
                else:
                    st.warning("⚠️ Please fill in both fields.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ------------------- AFTER LOGIN -------------------
if st.session_state.authenticated:
    st.markdown(f"<h2 style='color:#00BFFF;'>Welcome to LEXIBOT, <span style='color:white;'>{st.session_state.username}</span> 👋</h2>", unsafe_allow_html=True)

    if st.button("🚪 Logout"):
        log_user_action(st.session_state.username, "logout")
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()


from user_login import get_all_user_logs

if st.session_state.username == "admin":
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00BFFF;'>📋 Admin Activity Log</h3>", unsafe_allow_html=True)

    logs = get_all_user_logs()
    if logs:
        st.dataframe(
            {
                "Username": [log[0] for log in logs],
                "Action": [log[1] for log in logs],
                "Timestamp": [log[2] for log in logs]
            },
            use_container_width=True
        )
    else:
        st.info("No logs found yet.")


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
import base64
import streamlit as st

# File uploader widget for image upload
from db_manager import insert_candidate, get_top_domains_by_score

    
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
from llm_manager import call_llm

from pydantic import BaseModel

# Set page config


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
# ------------------- Core Setup -------------------


# Load environment variables
load_dotenv()

# Detect Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
working_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------- API Key & Caching Manager -------------------
from llm_manager import get_next_groq_key  # <- NEW

# Select current API key from rotation
groq_api_key = get_next_groq_key(st.session_state)

# ------------------- Lazy Initialization -------------------
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

from courses import COURSES_BY_CATEGORY, RESUME_VIDEOS, INTERVIEW_VIDEOS, get_courses_for_role





FEATURED_COMPANIES = {
    "tech": [
        {
            "name": "Google",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
            "color": "#4285F4",
            "careers_url": "https://careers.google.com",
            "description": "Leading technology company known for search, cloud, and innovation",
            "categories": ["Software", "AI/ML", "Cloud", "Data Science"]
        },
        {
            "name": "Microsoft",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
            "color": "#00A4EF",
            "careers_url": "https://careers.microsoft.com",
            "description": "Global leader in software, cloud, and enterprise solutions",
            "categories": ["Software", "Cloud", "Gaming", "Enterprise"]
        },
        {
            "name": "Amazon",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
            "color": "#FF9900",
            "careers_url": "https://www.amazon.jobs",
            "description": "E-commerce and cloud computing giant",
            "categories": ["Software", "Operations", "Cloud", "Retail"]
        },
        {
            "name": "Apple",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
            "color": "#555555",
            "careers_url": "https://www.apple.com/careers",
            "description": "Innovation leader in consumer technology",
            "categories": ["Software", "Hardware", "Design", "AI/ML"]
        },
        {
            "name": "Facebook",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/0/05/Facebook_Logo_%282019%29.png",
            "color": "#1877F2",
            "careers_url": "https://www.metacareers.com/",
            "description": "Social media and technology company",
            "categories": ["Software", "Marketing", "Networking", "AI/ML"]
        },
        {
            "name": "Netflix",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
            "color": "#E50914",
            "careers_url": "https://explore.jobs.netflix.net/careers",
            "description": "Streaming media company",
            "categories": ["Software", "Marketing", "Design", "Service"],
            "website": "https://jobs.netflix.com/",
            "industry": "Entertainment & Technology"
        }
    ],
    "indian_tech": [
        {
            "name": "TCS",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/f/f6/TCS_New_Logo.svg",
            "color": "#0070C0",
            "careers_url": "https://www.tcs.com/careers",
            "description": "India's largest IT services company",
            "categories": ["IT Services", "Consulting", "Digital"]
        },
        {
            "name": "Infosys",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/5/55/Infosys_logo.svg",
            "color": "#007CC3",
            "careers_url": "https://www.infosys.com/careers",
            "description": "Global leader in digital services and consulting",
            "categories": ["IT Services", "Consulting", "Digital"]
        },
        {
            "name": "Wipro",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Wipro_Primary_Logo_Color_RGB.svg",
            "color": "#341F65",
            "careers_url": "https://careers.wipro.com",
            "description": "Leading global information technology company",
            "categories": ["IT Services", "Consulting", "Digital"]
        },
        {
            "name": "HCL",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/5/5e/HCL_Technologies_logo.svg",
            "color": "#0075C9",
            "careers_url": "https://www.hcltech.com/careers",
            "description": "Global technology company",
            "categories": ["IT Services", "Engineering", "Digital"]
        }
    ],
    "global_corps": [
        {
            "name": "IBM",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
            "color": "#1F70C1",
            "careers_url": "https://www.ibm.com/careers",
            "description": "Global leader in technology and consulting",
            "categories": ["Software", "Consulting", "AI/ML", "Cloud"],
            "website": "https://www.ibm.com/careers/",
            "industry": "Technology & Consulting"
        },
        {
            "name": "Accenture",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Accenture_Logo.svg",
            "color": "#A100FF",
            "careers_url": "https://www.accenture.com/careers",
            "description": "Global professional services company",
            "categories": ["Consulting", "Technology", "Digital"]
        },
        {
            "name": "Cognizant",
            "logo_url": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Cognizant_logo_2022.svg",
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
    """Get featured companies with original logos, optionally filtered by category"""
    def has_valid_logo(company):
        return "logo_url" in company and company["logo_url"].startswith("https://upload.wikimedia.org/")

    if category and category in FEATURED_COMPANIES:
        return [company for company in FEATURED_COMPANIES[category] if has_valid_logo(company)]
    
    return [
        company for companies in FEATURED_COMPANIES.values()
        for company in companies if has_valid_logo(company)
    ]


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
    # Naukri - add keyword (k), location (l), and experience if available
    naukri_url = f"https://www.naukri.com/{role_encoded}-jobs-in-{loc_encoded}?k={role_encoded}&l={loc_encoded}"
    if experience_level and experience_exact_map.get(experience_level):
     naukri_url += f"&experience={experience_exact_map[experience_level]}"


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

import spacy
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Example gender_words dictionary (use your full research-backed lists here)
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

def detect_bias(text):
    doc = nlp(text)
    
    masc, fem = 0, 0
    masculine_found = []
    feminine_found = []

    for sent in doc.sents:
        sent_text = sent.text
        sent_lower = sent_text.lower()

        # Check masculine words
        for word in gender_words["masculine"]:
            if re.search(rf'\b{re.escape(word)}\b', sent_lower):
                masc += 1
                masculine_found.append({
                    "word": word,
                    "sentence": sent_text
                })

        # Check feminine words
        for word in gender_words["feminine"]:
            if re.search(rf'\b{re.escape(word)}\b', sent_lower):
                fem += 1
                feminine_found.append({
                    "word": word,
                    "sentence": sent_text
                })

    total = masc + fem

    if total == 0:
        return 0.0, masc, fem, masculine_found, feminine_found

    # Weighted bias score (example logic)
    bias_score = min(total / 20, 1.0)

    return round(bias_score, 2), masc, fem, masculine_found, feminine_found


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
    """
    Uses LLM to rewrite a resume with bias-free language, while preserving
    the original content length. Enhances grammar, structure, and clarity.
    Ensures structured formatting and includes relevant links and job suggestions.
    """

    # Create a clear mapping in bullet format
    formatted_mapping = "\n".join(
        [f'- "{key}" → "{value}"' for key, value in replacement_mapping.items()]
    )

    # Prompt for LLM
    prompt = f"""
You are an expert resume editor and career advisor.

Your tasks:

1. ✨ Rewrite the resume text below with these rules:
   - Replace any biased or gender-coded language using the exact matches from the replacement mapping.
   - Do NOT reduce the length of any section — preserve the original **number of words per section**.
   - Improve grammar, tone, sentence clarity, and flow without shortening or removing any content.
   - Do NOT change or remove names, tools, technologies, certifications, or project details.

2. 🧾 Structure the resume using these sections **if present** in the original, keeping the original text size:
   - 🏷️ **Name**
   - 📞 **Contact Information**
   - 📧 **Email**
   - 🔗 **LinkedIn** → If missing, insert: 🔗 Please paste your LinkedIn URL here.
   - 🌐 **Portfolio** → If missing, insert: 🌐 Please paste your GitHub or portfolio link here.
   - ✍️ **Professional Summary**
   - 💼 **Work Experience**
   - 🧑‍💼 **Internships**
   - 🛠️ **Skills**
   - 🤝 **Soft Skills**
   - 🎓 **Certifications**
   - 🏫 **Education**
   - 📂 **Projects**
   - 🌟 **Interests**

   - Use bullet points (•) inside each section for clarity.
   - Maintain new lines after each points properly.
   - Keep all hyperlinks intact and show them in full where applicable (e.g., LinkedIn, GitHub, project links).
   - Do not invent or assume any information not present in the original.

3. 📌 Strictly apply this **replacement mapping** (match exact phrases only — avoid altering keywords or terminology):
{formatted_mapping}

4. 💼 Suggest **5 relevant job titles** suited for this candidate based in **{user_location}**. For each:
   - Provide a detailed  reason for relevance.
   - Attach a direct LinkedIn job search URL.

---

### 📄 Original Resume Text
\"\"\"{text}\"\"\"

---

### ✅ Bias-Free Rewritten Resume (Fully Structured, Same Length)

---

### 🎯 Suggested Job Titles with Reasoning and LinkedIn Search Links

1. **[Job Title 1]** — Brief reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%201&location={user_location})

2. **[Job Title 2]** — Brief reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%202&location={user_location})

3. **[Job Title 3]** — Brief reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%203&location={user_location})

4. **[Job Title 4]** — Brief reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%204&location={user_location})

5. **[Job Title 5]** — Brief reason  
🔗 [Search on LinkedIn](https://www.linkedin.com/jobs/search/?keywords=Job%20Title%205&location={user_location})
"""

    # Call the LLM of your choice
    response = call_llm(prompt, session=st.session_state)
    return response



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
import re
import streamlit as st
import pandas as pd
import altair as alt
from llm_manager import call_llm

# ✅ LLM-based Grammar Score Function
def get_grammar_score_with_llm(text, max_score=5):
    if not text.strip():
        return 1, "Empty or unreadable resume."

    prompt = f"""
You are a professional grammar evaluator.

Evaluate the grammar quality of the following text and give:
1. A score from 1 to {max_score} based strictly on grammar issues.
2. A short explanation justifying the score.

Text:
\"\"\"{text}\"\"\"
"""

    response = call_llm(prompt, session=st.session_state)

    # Try to extract score from the response
    import re
    score = max_score
    reason = response.strip()
    match = re.search(r"\b([1-5])\b", response)
    if match:
        score = int(match.group(1))

    return score, reason


# ✅ Optional: If you want to auto-correct grammar via LLM
def fix_grammar_with_llm(text):
    prompt = f"""
Correct all grammar, spelling, and sentence structure issues in the following text.

Return only the corrected version.

---
{text}
---
"""
    return call_llm(prompt, session=st.session_state)

# ✅ ATS Evaluation Function
def ats_percentage_score(
    resume_text,
    job_description,
    logic_profile_score=None,
    edu_weight=20,
    exp_weight=35,
    skills_weight=30,
    lang_weight=5,
    keyword_weight=10
):
    logic_score_note = (
        f"\n\nOptional Note: The system also calculated a logic-based profile score of {logic_profile_score}/100 based on resume length, experience, and skills."
        if logic_profile_score else ""
    )

    prompt = f"""
You are an AI-powered ATS evaluator. You must evaluate the candidate's resume strictly based on the scoring rules below. 
You are not allowed to guess, assume, or round scores casually. Your component scores will be validated programmatically,
so they must follow **exact arithmetic based on the provided weights**.

---

📊 **Scoring Formula (Total = 100 Points)**

1. **Education Score ({edu_weight} pts)**
   - +50% if degree title (e.g., B.Tech, MSc) matches JD
   - +50% if field (e.g., CS, IT, Engineering) matches JD
   - Partial matches get 25% each

2. **Experience Score ({exp_weight} pts)**
   - +60% if years of experience ≥ required
   - +30% if role titles match
   - +10% if domain/industry matches

3. **Skills Match ({skills_weight} pts)**
   - Extract skill sets from both resume and JD
   - Score = (# of matching skills / total JD skills) × {skills_weight}
   - Show both skill lists

4. **Language Quality ({lang_weight} pts)**
   - {lang_weight} = Very clear, formal, professional tone
   - Half for minor grammatical/style issues
   - Low score for informal or unclear writing

5. **Keyword Match ({keyword_weight} pts)**
   - Extract tools, tech, frameworks from JD
   - For each missing keyword, deduct proportionally from {keyword_weight}
   - Show missing keyword list

---

📐 **Scoring Instruction**
- Total Score = Sum of all 5 components.
- This becomes the **Overall Percentage Match**.
- Max = 100. If score exceeds, cap it at 100.
- Never return 0 unless all components are truly zero.
- Return exact numeric values.

📊 **Score Bands for Formatted Score**  
- 85–100: Excellent  
- 70–84: Good  
- 50–69: Average  
- Below 50: Poor

---

🧾 **OUTPUT FORMAT (Strictly follow this):**

Candidate Name: <full name or "Not Found">

Education Score: <0–{edu_weight}>  
Education Match Details: <e.g., "Degree title matches B.Tech; field matches Computer Science.">

Experience Score: <0–{exp_weight}>  
Experience Highlights: <e.g., "5 years experience (JD requires 4), title matches 'Software Engineer', domain matched IT.">

Skills Match Percentage: <0–{skills_weight}>  
Skills Found in Resume: <comma-separated list>  
Skills Required by JD: <comma-separated list>  
Skills Missing: <comma-separated list>

Language Quality Score: <0–{lang_weight}>  
Language Quality Comments: <e.g., "Professional tone with minor grammatical errors.">

Keyword Match Score: <0–{keyword_weight}>  
Missing Keywords: <comma-separated list>

Overall Percentage Match: <sum of above components>  
Formatted Score: <Excellent / Good / Average / Poor>

Final Thoughts:  
Provide a detailed summary (4–6 sentences) about the candidate’s overall fit. Highlight strengths.

{logic_score_note}

---

### Job Description:
\"\"\"{job_description}\"\"\"

---

### Resume:
\"\"\"{resume_text}\"\"\"
"""

    # 🔁 LLM-based ATS response
    response = call_llm(prompt, session=st.session_state)
    ats_result = response.strip()

    # 🧪 Regex-based score extraction
    def extract_score(pattern, text, default=0):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else default

    edu_score = extract_score(r"Education Score:\s*(\d+)", ats_result)
    exp_score = extract_score(r"Experience Score:\s*(\d+)", ats_result)
    skills_score = extract_score(r"Skills Match Percentage:\s*(\d+)", ats_result)
    keyword_score = extract_score(r"Keyword Match Score:\s*(\d+)", ats_result)

    # ✅ LanguageTool Python for Language Score
    lang_score, lang_comment = get_grammar_score_with_llm(resume_text, max_score=lang_weight)


    # 🩹 Patch LLM-generated language section with real grammar result
    ats_result = re.sub(r"Language Quality Score:\s*\d+", f"Language Quality Score: {lang_score}", ats_result)
    ats_result = re.sub(r"Language Quality Comments:.*", f"Language Quality Comments: {lang_comment}", ats_result)

    # 🎯 Final ATS Match Score
    total_score = min(edu_score + exp_score + skills_score + lang_score + keyword_score, 100)

    # 📊 Score band
    if total_score >= 85:
        formatted_score = "Excellent"
    elif total_score >= 70:
        formatted_score = "Good"
    elif total_score >= 50:
        formatted_score = "Average"
    else:
        formatted_score = "Poor"

    ats_result = re.sub(r"Overall Percentage Match:\s*\d+", f"Overall Percentage Match: {total_score}", ats_result)
    ats_result = re.sub(r"Formatted Score:\s*.*", f"Formatted Score: {formatted_score}", ats_result)

    return ats_result, {
        "Education Score": edu_score,
        "Experience Score": exp_score,
        "Skills Match %": skills_score,
        "Language Quality Score": lang_score,
        "Keyword Match Score": keyword_score,
        "ATS Match %": total_score,
        "Formatted Score": formatted_score
    }


# App Title
st.title("🦙 Chat with LEXIBOT - LLAMA 3.3 (Bias Detection + QA + GPU)")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

    st.sidebar.markdown("### 🏷️ Job Information")
job_title = st.sidebar.text_input("💼 Job Title")  # <-- New input for job title

st.sidebar.markdown("### 📝 Paste Job Description")
job_description = st.sidebar.text_area("Enter the Job Description here", height=200)

if job_description.strip() == "":
    st.sidebar.warning("Please enter a job description to evaluate the resumes.")

user_location = st.sidebar.text_input("📍 Preferred Job Location (City, Country)")

st.sidebar.markdown("### 🎛️ Customize ATS Scoring Weights")


edu_weight = st.sidebar.slider("🎓 Education Weight", 0, 50, 20)
exp_weight = st.sidebar.slider("💼 Experience Weight", 0, 50, 35)
skills_weight = st.sidebar.slider("🛠 Skills Match Weight", 0, 50, 30)
lang_weight = st.sidebar.slider("🗣 Language Quality Weight", 0, 10, 5)
keyword_weight = st.sidebar.slider("🔑 Keyword Match Weight", 0, 20, 10)

total_weight = edu_weight + exp_weight + skills_weight + lang_weight + keyword_weight

if total_weight != 100:
    st.sidebar.error(f"⚠️ Total = {total_weight}. Please make it exactly 100.")
else:
    st.sidebar.success("✅ Total weight = 100")


uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)


import os
import re
import streamlit as st
from datetime import datetime
import os
import re
import streamlit as st
from datetime import datetime
from db_manager import insert_candidate, detect_domain_from_title_and_description  # ✅ import db helpers
# Initialize resume data storage
# ✅ Use persistent storage for resume results
if "resume_data" not in st.session_state:
    st.session_state.resume_data = []
resume_data = st.session_state.resume_data

# ✅ Ensure processed file tracker exists
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ✏️ Resume Evaluation Logic
if uploaded_files and job_description:
    all_text = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            continue

        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = extract_text_from_pdf(file_path)
        all_text.extend(text)
        full_text = " ".join(text)

        # Bias detection
        bias_score, masc_count, fem_count, detected_masc, detected_fem = detect_bias(full_text)

# Format detected masculine and feminine words with their context sentences
        detected_masc_formatted = [
        f"{item['word']} ➔ {item['sentence']}" for item in detected_masc
    ]
        detected_fem_formatted = [
        f"{item['word']} ➔ {item['sentence']}" for item in detected_fem
    ] 
        highlighted_text, rewritten_text, masc_count, fem_count, detected_masc, detected_fem = rewrite_and_highlight(
            full_text, replacement_mapping, user_location
        )

        # ATS scoring and report
        ats_result, ats_scores = ats_percentage_score(
            resume_text=full_text,
            job_description=job_description,
            logic_profile_score=None,
            edu_weight=edu_weight,
            exp_weight=exp_weight,
            skills_weight=skills_weight,
            lang_weight=lang_weight,
            keyword_weight=keyword_weight
        )

        # Score/text extractors
        def extract_score(pattern, text, default=0):
            match = re.search(pattern, text)
            return int(match.group(1)) if match else default

        def extract_text(pattern, text, default="N/A"):
            match = re.search(pattern, text)
            return match.group(1).strip() if match else default

        candidate_name = extract_text(r"Candidate Name:\s*(.*)", ats_result)
        ats_score = extract_score(r"Overall Percentage Match:\s*(\d+)", ats_result)
        edu_score = extract_score(r"Education Score:\s*(\d+)", ats_result)
        exp_score = extract_score(r"Experience Score:\s*(\d+)", ats_result)
        skills_score = extract_score(r"Skills Match Percentage:\s*(\d+)", ats_result)
        lang_score = extract_score(r"Language Quality Score:\s*(\d+)", ats_result)
        keyword_score = extract_score(r"Keyword Match Score:\s*(\d+)", ats_result)
        formatted_score = extract_text(r"Formatted Score:\s*(.*)", ats_result)
        missing_keywords = extract_text(r"Missing Keywords:\s*(.*)", ats_result)
        fit_summary = extract_text(r"Final Thoughts:\s*(.*)", ats_result)

        # Detect domain from job info
        domain = detect_domain_from_title_and_description(job_title, job_description)

        # Flags
        bias_flag = "🔴 High Bias" if bias_score > 0.6 else "🟢 Fair"
        ats_flag = "⚠️ Low ATS" if ats_score < 50 else "✅ Good ATS"

        # Build ATS chart
        ats_df = pd.DataFrame({
            'Component': ['Education', 'Experience', 'Skills', 'Language', 'Keywords'],
            'Score': [edu_score, exp_score, skills_score, lang_score, keyword_score]
        })
        ats_chart = alt.Chart(ats_df).mark_bar().encode(
            x=alt.X('Component', sort=None),
            y=alt.Y('Score', scale=alt.Scale(domain=[0, 50])),
            color='Component',
            tooltip=['Component', 'Score']
        ).properties(
            title="ATS Evaluation Breakdown",
            width=600,
            height=300
        )

        # Save all data to session state
        st.session_state.resume_data.append({
            "Resume Name": uploaded_file.name,
            "Candidate Name": candidate_name,
            "ATS Report": ats_result,
            "ATS Match %": ats_scores["ATS Match %"],
            "Formatted Score": ats_scores["Formatted Score"],
            "Education Score": ats_scores["Education Score"],
            "Experience Score": ats_scores["Experience Score"],
            "Skills Match %": ats_scores["Skills Match %"],
            "Language Quality Score": ats_scores["Language Quality Score"],
            "Keyword Match Score": ats_scores["Keyword Match Score"],
            "Missing Keywords": missing_keywords,
            "Fit Summary": fit_summary,
            "Bias Score (0 = Fair, 1 = Biased)": bias_score,
            "Bias Status": bias_flag,
            "Masculine Words": masc_count,
            "Feminine Words": fem_count,
            "Detected Masculine Words": detected_masc_formatted,
            "Detected Feminine Words": detected_fem_formatted,
            "Text Preview": full_text[:300] + "...",
            "Highlighted Text": highlighted_text,
            "Rewritten Text": rewritten_text,
            "Domain": domain
        })

        # Save to DB
        insert_candidate((
            uploaded_file.name,
            candidate_name,
            ats_score,
            edu_score,
            exp_score,
            skills_score,
            lang_score,
            keyword_score,
            bias_score,
            domain
        ))

        st.session_state.processed_files.add(uploaded_file.name)

    st.success("✅ All resumes processed!")

    # Setup vectorstore + chain
    if all_text:
        st.session_state.vectorstore = setup_vectorstore(all_text)
        st.session_state.chain = create_chain(st.session_state.vectorstore)

# Optional dev reset
if st.button("🔄 Reset Resume Upload Memory"):
    st.session_state.processed_files.clear()
    st.session_state.resume_data.clear()
    st.success("✅ Cleared uploaded resume history. You can re-upload now.")




# === TAB 1: Dashboard ===
# 📊 Dashboard and Metrics
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🧾 Resume Builder", "💼 Job Search", 
    "📚 Course Recommendation", "📁 Admin DB View"
])
def generate_resume_report_html(resume):
    rewritten_text = resume['Rewritten Text'].replace("\n", "<br>")

    # Masculine words formatted
    if resume["Detected Masculine Words"]:
        masculine_words = ""
        for item in resume["Detected Masculine Words"]:
            if " ➔ " in item:
                word, sentence = item.split(" ➔ ", 1)
                masculine_words += f"<b>{word}</b>: {sentence}<br>"
    else:
        masculine_words = "<i>None detected.</i>"

    # Feminine words formatted
    if resume["Detected Feminine Words"]:
        feminine_words = ""
        for item in resume["Detected Feminine Words"]:
            if " ➔ " in item:
                word, sentence = item.split(" ➔ ", 1)
                feminine_words += f"<b>{word}</b>: {sentence}<br>"
    else:
        feminine_words = "<i>None detected.</i>"

    missing_keywords = "".join(
        f"<span class='keyword'>{kw.strip()}</span>"
        for kw in resume['Missing Keywords'].split(",") if kw.strip()
    ) or "<i>None</i>"

    ats_report_html = resume.get("ATS Report", "").replace("\n", "<br>")

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>{resume['Candidate Name']} - Resume Analysis Report</title>
    <style>
        body{{font-family:'Segoe UI',sans-serif;margin:40px;background:#f5f7fa;color:#333}}
        h1,h2{{color:#2f4f6f}}.section{{margin-bottom:30px}}
        .highlight{{background-color:#eef;padding:10px;border-radius:6px;margin-top:10px;font-size:14px}}
        .metric-box{{display:inline-block;background:#dbeaff;padding:10px 20px;margin:10px;border-radius:10px;font-weight:bold}}
        .keyword{{display:inline-block;background:#fbdcdc;color:#a33;margin:4px;padding:6px 12px;border-radius:12px;font-size:13px}}
        .resume-box{{background-color:#f9f9ff;padding:15px;border-radius:8px;border:1px solid #ccc;white-space:pre-wrap}}
        .report-box{{background:#fffbe6;border-left:5px solid #f7d794;padding:10px;margin-top:10px;border-radius:6px}}
    </style>
    </head><body>
    <h1>📄 Resume Analysis Report</h1>

    <div class="section">
        <h2>Candidate: {resume['Candidate Name']}</h2>
        <p><strong>Resume File:</strong> {resume['Resume Name']}</p>
    </div>

    <div class="section">
        <h2>📊 ATS Evaluation</h2>
        <div class="metric-box">ATS Match: {resume['ATS Match %']}%</div>
        <div class="metric-box">Education: {resume['Education Score']}</div>
        <div class="metric-box">Experience: {resume['Experience Score']}</div>
        <div class="metric-box">Skills Match: {resume['Skills Match %']}</div>
        <div class="metric-box">Language Score: {resume['Language Quality Score']}</div>
        <div class="metric-box">Keyword Score: {resume['Keyword Match Score']}</div>

        <div class="report-box">
            <h3>📋 ATS Evaluation Report</h3>
            {ats_report_html}
        </div>
    </div>

    <div class="section">
        <h2>⚖️ Gender Bias Analysis</h2>
        <div class="metric-box" style="background:#f0f8ff;">Masculine Words: {resume['Masculine Words']}</div>
        <div class="metric-box" style="background:#fff0f5;">Feminine Words: {resume['Feminine Words']}</div>
        <p><strong>Bias Score (0=Fair, 1=Biased):</strong> {resume['Bias Score (0 = Fair, 1 = Biased)']}</p>
        <div class="highlight"><strong>Masculine Words Detected:</strong><br>{masculine_words}</div>
        <div class="highlight"><strong>Feminine Words Detected:</strong><br>{feminine_words}</div>
    </div>

    <div class="section">
        <h2>📌 Missing Keywords</h2>
        {missing_keywords}
    </div>

    <div class="section">
        <h2>🧠 Final Fit Summary</h2>
        <div class="resume-box">{resume['Fit Summary']}</div>
    </div>

    <div class="section">
        <h2>✅ Rewritten Bias-Free Resume</h2>
        <div class="resume-box">{rewritten_text}</div>
    </div>

    </body></html>"""


# === TAB 1: Dashboard ===
with tab1:
    resume_data = st.session_state.get("resume_data", [])

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
            df[[ 
                "Resume Name", "Candidate Name", "ATS Match %", "Education Score",
                "Experience Score", "Skills Match %", "Language Quality Score", "Keyword Match Score",
                "Bias Score (0 = Fair, 1 = Biased)", "Masculine Words", "Feminine Words"
            ]],
            use_container_width=True
        )

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

        st.markdown("### 📝 Detailed Resume Reports")
        for resume in resume_data:
            with st.expander(f"📄 {resume['Resume Name']} | {resume['Candidate Name']}"):
                st.markdown("### 📊 ATS Evaluation for: **" + resume['Candidate Name'] + "**")
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("📈 Overall Match", f"{resume['ATS Match %']}%")
                with score_col2:
                    st.metric("🏆 Formatted Score", resume.get("Formatted Score", "N/A"))
                with score_col3:
                    st.metric("🧠 Language Quality", f"{resume.get('Language Quality Score', 'N/A')} / {lang_weight}")

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("🎓 Education Score", f"{resume.get('Education Score', 'N/A')} / {edu_weight}")
                with col_b:
                    st.metric("💼 Experience Score", f"{resume.get('Experience Score', 'N/A')} / {exp_weight}")
                with col_c:
                    st.metric("🛠 Skills Match", f"{resume.get('Skills Match %', 'N/A')} / {skills_weight}")
                with col_d:
                    st.metric("🔍 Keyword Score", f"{resume.get('Keyword Match Score', 'N/A')} / {keyword_weight}")

                # Fit summary
                st.markdown("### 📝 Fit Summary")
                st.write(resume['Fit Summary'])

                # ATS Report
                if "ATS Report" in resume:
                    st.markdown("### 📋 ATS Evaluation Report")
                    st.markdown(resume["ATS Report"], unsafe_allow_html=True)

                # ATS Chart
                st.markdown("### 📊 ATS Score Breakdown Chart")
                ats_df = pd.DataFrame({
                    'Component': ['Education', 'Experience', 'Skills', 'Language', 'Keywords'],
                    'Score': [
                        resume.get("Education Score", 0),
                        resume.get("Experience Score", 0),
                        resume.get("Skills Match %", 0),
                        resume.get("Language Quality Score", 0),
                        resume.get("Keyword Match Score", 0)
                    ]
                })
                ats_chart = alt.Chart(ats_df).mark_bar().encode(
                    x=alt.X('Component', sort=None),
                    y=alt.Y('Score', scale=alt.Scale(domain=[0, 50])),
                    color='Component',
                    tooltip=['Component', 'Score']
                ).properties(
                    title="ATS Evaluation Breakdown",
                    width=600,
                    height=300
                )
                st.altair_chart(ats_chart, use_container_width=True)

                # Missing keywords
                st.markdown("**❗ Missing Keywords:**")
                missing_list = resume["Missing Keywords"].split(",") if resume["Missing Keywords"] else []
                if missing_list and any(kw.strip() for kw in missing_list):
                    for kw in missing_list:
                        st.error(f"- {kw.strip()}")
                else:
                    st.info("No missing keywords detected.")

                st.divider()
                detail_tab1, detail_tab2 = st.tabs(["🔎 Bias Analysis", "✅ Rewritten Resume"])
                with detail_tab1:
                    st.markdown("#### Bias-Highlighted Original Text")
                    st.markdown(resume["Highlighted Text"], unsafe_allow_html=True)
                    st.markdown("### 📌 Gender-Coded Word Counts:")
                    bias_col1, bias_col2 = st.columns(2)
                    with bias_col1:
                        st.metric("🔵 Masculine Words", resume["Masculine Words"])
                        if resume["Detected Masculine Words"]:
                            st.markdown("### 📚 Detected Masculine Words with Context:")
                            for item in resume["Detected Masculine Words"]:
                                word, sentence = item.split(" ➔ ", 1)
                                st.write(f"🔵 **{word}**: {sentence}")
                        else:
                            st.info("No masculine words detected.")
                    with bias_col2:
                        st.metric("🔴 Feminine Words", resume["Feminine Words"])
                        if resume["Detected Feminine Words"]:
                            st.markdown("### 📚 Detected Feminine Words with Context:")
                            for item in resume["Detected Feminine Words"]:
                                word, sentence = item.split(" ➔ ", 1)
                                st.write(f"🔴 **{word}**: {sentence}")
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
                        key=f"download_docx_{resume['Resume Name']}"
                    )
                    html_report = generate_resume_report_html(resume)
                    pdf_bytes = html_to_pdf_bytes(html_report)

                    st.download_button(
                        label="📥 Download ATS Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"{resume['Candidate Name']}_ATS_Report.pdf",
                        mime="application/pdf"
                    )

                    st.download_button(
                        label="📥 Download Full Analysis Report (.html)",
                        data=html_report,
                        file_name=f"{resume['Resume Name'].split('.')[0]}_report.html",
                        mime="text/html",
                        use_container_width=True,
                        key=f"download_html_{resume['Resume Name']}"
                    )
    else:
        st.warning("⚠️ Please upload resumes to view dashboard analytics.")


with tab2:
    st.session_state.active_tab = "Resume Builder"

    st.markdown("## 🧾 <span style='color:#336699;'>Advanced Resume Builder</span>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #bbb;'>", unsafe_allow_html=True)

    # 📸 Upload profile photo
    uploaded_image = st.file_uploader("Upload a Profile Image", type=["png", "jpg", "jpeg"])

    profile_img_html = ""

    if uploaded_image:
        import base64
        encoded_image = base64.b64encode(uploaded_image.read()).decode()

        # 🔄 Save to session state for reuse in preview/export
        st.session_state["encoded_profile_image"] = encoded_image

        # 🖼️ Show image preview
        profile_img_html = f"""
        <div style="display: flex; justify-content: flex-end; margin-top: 20px;">
            <img src="data:image/png;base64,{encoded_image}" alt="Profile Photo"
                 style="
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    object-fit: cover;
                    object-position: top center;
                    border: 3px solid #4da6ff;
                    box-shadow:
                        0 0 8px #4da6ff,
                        0 0 16px #4da6ff,
                        0 0 24px #4da6ff;
                " />
        </div>
        """
        st.markdown(profile_img_html, unsafe_allow_html=True)
    else:
        st.info("📸 Please upload a clear, front-facing profile photo (square or vertical preferred).")


    # 🔽 Your form fields continue below this...


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
                    <b>🏢 {exp['company']}</b><span style='color:gray;'>📆  {exp['duration']}</span>
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
    f"""
    <div style='display:inline-block; background-color:#e6f0fa; color:#333; 
                padding:8px 16px; margin:6px 6px 6px 0; 
                border-radius:20px; font-size:15px; font-weight:500;'>
        {s.strip()}
    </div>
    """
    for s in st.session_state['skills'].split(',')
    if s.strip()
)




languages_html = "".join(
    f"""
    <div style='display:inline-block; background-color:#e6f0fa; color:#333; 
                padding:8px 16px; margin:6px 6px 6px 0; 
                border-radius:20px; font-size:15px; font-weight:500;'>
        {lang.strip()}
    </div>
    """
    for lang in st.session_state['languages'].split(',')
    if lang.strip()
)


# INTERESTS
interests_html = "".join(
    f"""
    <div style='display:inline-block; background-color:#e6f0fa; color:#333; 
                padding:8px 16px; margin:6px 6px 6px 0; 
                border-radius:20px; font-size:15px; font-weight:500;'>
        {interest.strip()}
    </div>
    """
    for interest in st.session_state['interests'].split(',')
    if interest.strip()
)


Softskills_html = "".join(
    f"""
    <div style='display:inline-block; background-color:#eef3f8; color:#1a1a1a; 
                padding:8px 18px; margin:6px 6px 6px 0; 
                border-radius:25px; font-size:14.5px; font-family:"Segoe UI", sans-serif; 
                font-weight:500; box-shadow: 1px 1px 3px rgba(0,0,0,0.05);'>
        {skill.strip().capitalize()}
    </div>
    """
    for skill in st.session_state['Softskills'].split(',')
    if skill.strip()
)



experience_html = ""
for exp in st.session_state.experience_entries:
    if exp["company"] or exp["title"]:
        # Handle paragraphs and single line breaks
        description_lines = [line.strip() for line in exp["description"].strip().split("\n\n")]
        description_html = "".join(
            f"<div style='margin-bottom: 8px;'>{line.replace(chr(10), '<br>')}</div>"
            for line in description_lines if line
        )

        experience_html += f"""
<div style='
    margin-bottom: 20px;
    padding: 16px 20px;
    border-radius: 12px;
    background-color: #dbeaff;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    color: #0a1a33;
    line-height: 1.35;
'>
    <!-- Header Shadow Card -->
    <div style='
        background-color: #e6f0ff;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    '>
        <div style='
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: 16.5px;
            margin-bottom: 6px;
            color: #08244c;
            width: 100%;
        '>
            <div style='display: inline-flex; align-items: center;'>
                <img src="https://img.icons8.com/ios-filled/50/000000/company.png" style="width:16px; height:16px; margin-right:5px;"/>
                <span>{exp['company']}</span>
            </div>
            <div style='display: inline-flex; align-items: center; font-size: 14px;'>
                <img src="https://img.icons8.com/ios-filled/50/000000/calendar.png" style="width:16px; height:16px; margin-right:5px;"/>
                <span>{exp['duration']}</span>
            </div>
        </div>

        <div style='
            display: inline-flex;
            align-items: center;
            font-size: 16px;
            font-weight: 700;
            color: #0b2545;
        '>
            <img src="https://img.icons8.com/ios-filled/50/000000/briefcase.png" style="width:16px; height:16px; margin-right:5px;"/>
            <span>{exp['title']}</span>
        </div>
    </div>

    <!-- Description -->
    <div style='
        display: inline-flex;
        align-items: flex-start;
        font-size: 15px;
        font-weight: 500;
        color: #102a43;
        line-height: 1.35;
    '>
        <img src="https://img.icons8.com/ios-filled/50/000000/task.png" style="width:16px; height:16px; margin-right:5px; margin-top:2px;"/>
        <div>{description_html}</div>
    </div>
</div>
"""



# Convert experience to list if multiple lines

# Escape HTML and convert line breaks
summary_html = st.session_state['summary'].replace('\n', '<br>')


# EDUCATION
education_html = ""
for edu in st.session_state.education_entries:
    if edu.get("institution") or edu.get("details"):
        degree_text = ""
        if edu.get("degree"):
            degree_val = edu["degree"]
            if isinstance(degree_val, list):
                degree_val = ", ".join(degree_val)
            degree_text = f"""
            <div style='display: inline-flex; align-items: center; font-size: 14px; color: #273c75; margin-bottom: 6px;'>
                <img src="https://img.icons8.com/ios-filled/50/000000/graduation-cap.png" style="width:16px; height:16px; margin-right:5px;"/>
                <b>{degree_val}</b>
            </div>
            """

        education_html += f"""
        <div style='
            margin-bottom: 20px;
            padding: 16px 20px;
            border-radius: 12px;
            background-color: #e3ebf8;  /* Light Gray Blue */
            box-shadow: 0 3px 8px rgba(39, 60, 117, 0.15);
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            color: #273c75;  /* Dark Blue */
            line-height: 1.4;
        '>
            <div style='
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 8px;
                width: 100%;
            '>
                <div style='display: inline-flex; align-items: center;'>
                    <img src="https://img.icons8.com/ios-filled/50/000000/school.png" style="width:16px; height:16px; margin-right:5px;"/>
                    <span>{edu.get('institution', '')}</span>
                </div>
                <div style='display: inline-flex; align-items: center; font-weight: 500;'>
                    <img src="https://img.icons8.com/ios-filled/50/000000/calendar.png" style="width:16px; height:16px; margin-right:5px;"/>
                    <span>{edu.get('year', '')}</span>
                </div>
            </div>
            {degree_text}
            <div style='display: inline-flex; align-items: flex-start; font-size: 14px; font-style: italic;'>
                <img src="https://img.icons8.com/ios-filled/50/000000/task.png" style="width:16px; height:16px; margin-right:5px; margin-top:2px;"/>
                <div>{edu.get('details', '')}</div>
            </div>
        </div>
        """






# PROJECTS
# PROJECTS
projects_html = ""
for proj in st.session_state.project_entries:
    if proj.get("title") or proj.get("description"):
        tech_val = proj.get("tech")
        if isinstance(tech_val, list):
            tech_val = ", ".join(tech_val)
        tech_text = f"""
        <div style='display: inline-flex; align-items: center; font-size: 14px; color: #1b2330; margin-bottom: 8px; text-shadow: 1px 1px 2px rgba(0,0,0,0.15);'>
            <img src="https://img.icons8.com/ios-filled/50/000000/maintenance.png" style="width:16px; height:16px; margin-right:5px;"/>
            <b>Technologies:</b> {tech_val if tech_val else ''}
        </div>
        """ if tech_val else ""

        description_items = ""
        if proj.get("description"):
            description_lines = [line.strip() for line in proj["description"].splitlines() if line.strip()]
            description_items = "".join(f"<li>{line}</li>" for line in description_lines)

        projects_html += f"""
        <div style='
            margin-bottom: 22px;
            padding: 18px 24px;
            border-radius: 14px;
            background-color: #d7e1ec;  /* Deep Blue Slate */
            box-shadow: 0 4px 10px rgba(27, 35, 48, 0.15);
            font-family: "Roboto", "Helvetica Neue", Arial, sans-serif;
            color: #1b2330;  /* Deep Slate Blue */
            line-height: 1.5;
        '>
            <div style='
                font-size: 17px;
                font-weight: 700;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                color: #141a22;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
                width: 100%;
            '>
                <div style='display: inline-flex; align-items: center;'>
                    <img src="https://img.icons8.com/ios-filled/50/000000/laptop.png" style="width:16px; height:16px; margin-right:5px;"/>
                    <span>{proj.get('title', '')}</span>
                </div>
                <div style='display: inline-flex; align-items: center; font-weight: 600; font-size: 14.5px;'>
                    <img src="https://img.icons8.com/ios-filled/50/000000/time.png" style="width:16px; height:16px; margin-right:5px;"/>
                    <span>{proj.get('duration', '')}</span>
                </div>
            </div>
            {tech_text}
            <div style='display: inline-flex; align-items: flex-start; font-size: 15px; color: #1b2330; text-shadow: 1px 1px 2px rgba(0,0,0,0.15);'>
                <img src="https://img.icons8.com/ios-filled/50/000000/task.png" style="width:16px; height:16px; margin-right:5px; margin-top:2px;"/>
                <div>
                    <b>Description:</b>
                    <ul style='margin-top: 6px; padding-left: 22px; color: #1b2330;'>
                        {description_items}
                    </ul>
                </div>
            </div>
        </div>
        """





# PROJECT LINKS
project_links_html = ""
if st.session_state.project_links:
    project_links_html = "<h4 class='section-title'>Project Links</h4><hr>" + "".join(
        f'''
        <p>
            <img src="https://img.icons8.com/ios-filled/50/000000/link.png" style="width:16px; height:16px; vertical-align:middle; margin-right:5px;"/>
            <a href="{link}">Project {i+1}</a>
        </p>
        '''
        for i, link in enumerate(st.session_state.project_links)
    )




certificate_links_html = ""
if st.session_state.certificate_links:
    certificate_links_html = "<h4 class='section-title'>Certificates</h4><hr>"
    for cert in st.session_state.certificate_links:
        if cert["name"] and cert["link"]:
            description = cert.get('description', '').replace('\n', '<br>')
            name = cert['name']
            link = cert['link']
            duration = cert.get('duration', '')

            card_html = f"""
            <div style='
                background-color: #f9fbe7;  /* Green-Yellow pastel */
                padding: 20px 24px;
                border-radius: 16px;
                margin-bottom: 22px;
                box-shadow: 0 4px 14px rgba(34, 60, 80, 0.15);
                font-family: "Poppins", "Segoe UI", sans-serif;
                color: #1a1a1a;
                position: relative;
                line-height: 1.6;
            '>
                <!-- Duration Top Right -->
                <div style='
                    position: absolute;
                    top: 18px;
                    right: 24px;
                    font-size: 13.5px;
                    font-weight: 600;
                    color: #37474f;
                    text-shadow: 0.5px 0.5px 1px rgba(0, 0, 0, 0.15);

                    background-color: #fffde7;  /* pastel yellow */
                    padding: 4px 12px;
                    border-radius: 14px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
                '>
                    <img src="https://img.icons8.com/ios-filled/50/000000/time.png" style="width:14px; height:14px; vertical-align:middle; margin-right:4px;"/>
                    {duration}
                </div>

                <!-- Certificate Title -->
                <div style='
                    font-size: 17px;
                    font-weight: 700;
                    color: #263238;
                    margin-bottom: 8px;
                    text-shadow: 0.5px 0.5px 1.5px rgba(0, 0, 0, 0.1);
                '>
                    <img src="https://img.icons8.com/ios-filled/50/000000/certificate.png" style="width:16px; height:16px; vertical-align:middle; margin-right:5px;"/>
                    <a href="{link}" target="_blank" style='
                        color: #263238;
                        text-decoration: none;
                    '>{name}</a>
                </div>

                <!-- Description -->
                <div style='
                    font-size: 15px;
                    color: #37474f;
                    margin-top: 6px;
                    text-shadow: 0 0 1px rgba(0, 0, 0, 0.08);
                '>
                    <img src="https://img.icons8.com/ios-filled/50/000000/task.png" style="width:16px; height:16px; vertical-align:middle; margin-right:5px;"/>
                    {description}
                </div>
            </div>
            """
            certificate_links_html += card_html



html_content = f"""
<!DOCTYPE html>
<html lang="en">
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
            display: table;
            width: 100%;
        }}
        .left, .right {{
            display: table-cell;
            vertical-align: top;
        }}
        .left {{
            width: 30%;
            border-right: 2px solid #ccc;
            padding-right: 20px;
        }}
        .right {{
            width: 70%;
            padding-left: 20px;
        }}
        .icon {{
            width: 16px;
            height: 16px;
            margin-right: 6px;
        }}
        .contact-row {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }}
        .section-title {{
            color: #336699;
            margin-top: 30px;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>

    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <div>
            <h2>{st.session_state['name']}</h2>
            <h4>{st.session_state['job_title']}</h4>
        </div>
        <div>
            {profile_img_html}
        </div>
    </div>
    <hr>

    <div class="container">
        <div class="left">
            <div class="contact-row">
                <img src="https://img.icons8.com/ios-filled/50/000000/marker.png" class="icon"/>
                <span>{st.session_state['location']}</span>
            </div>
            <div class="contact-row">
                <img src="https://img.icons8.com/ios-filled/50/000000/phone.png" class="icon"/>
                <span>{st.session_state['phone']}</span>
            </div>
            <div class="contact-row">
                <img src="https://img.icons8.com/ios-filled/50/000000/email.png" class="icon"/>
                <a href="mailto:{st.session_state['email']}">{st.session_state['email']}</a>
            </div>
            <div class="contact-row">
                <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" class="icon"/>
                <a href="{st.session_state['linkedin']}">LinkedIn</a>
            </div>
            <div class="contact-row">
                <img src="https://img.icons8.com/ios-filled/50/000000/domain.png" class="icon"/>
                <a href="{st.session_state['portfolio']}">Portfolio</a>
            </div>

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
# Convert HTML resume to PDF bytes
pdf_resume_bytes = html_to_pdf_bytes(html_content)


with tab2:
    # Download Resume buttons
    st.download_button(
        label="📥 Download Resume (HTML)",
        data=html_file,
        file_name=f"{st.session_state['name'].replace(' ', '_')}_Resume.html",
        mime="text/html"
    )
    st.download_button(
        label="📥 Download Resume (PDF)",
        data=pdf_resume_bytes,
        file_name=f"{st.session_state['name'].replace(' ', '_')}_Resume.pdf",
        mime="application/pdf"
    )

    # Cover Letter Expander (INSIDE tab2)
    with st.expander("📩 Generate Cover Letter from This Resume"):
        generate_cover_letter_from_resume_builder()

        if "cover_letter" in st.session_state:
            st.markdown("### ✉️ Generated Cover Letter")

            # ✅ Display with styled header template
            name = st.session_state.get("name", "")
            job_title = st.session_state.get("job_title", "")
            location = st.session_state.get("location", "")
            today_date = datetime.today().strftime("%B %d, %Y")
            cover_letter_body = st.session_state["cover_letter"]

            styled_cover_letter = f"""
            <div style="font-family: Georgia, serif; line-height: 1.6; color: #333; border: 1px solid #ccc; padding: 20px;">
                <h1 style="color: #003366; font-size: 28px; margin-bottom: 0;">{name}</h1>
                <h2 style="font-style: italic; font-weight: normal; margin-top: 0; color: #555;">{job_title}</h2>
                <p style="margin: 4px 0;">
                    {location}<br>
                    {today_date}<br>
                    LinkedIn: <a href="#" style="color: #003366;">Your LinkedIn URL</a>
                </p>
                <hr style="border: 1px solid #ccc;">
                <div style="white-space: pre-wrap;">{cover_letter_body}</div>
            </div>
            """

            st.markdown(styled_cover_letter, unsafe_allow_html=True)

            from io import BytesIO
            from docx import Document

            def create_docx(text, filename="cover_letter.docx"):
                doc = Document()
                doc.add_heading("Cover Letter", 0)
                doc.add_paragraph(text)
                bio = BytesIO()
                doc.save(bio)
                bio.seek(0)
                return bio

            # ✅ Download as .docx (plain text only, not HTML formatting)
            st.download_button(
                label="📥 Download Cover Letter (.docx)",
                data=create_docx(st.session_state["cover_letter"]),
                file_name="Cover_Letter.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )


    # Sejda HTML-to-PDF link
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
with tab4:
    # Inject CSS styles
    st.markdown("""
        <style>
        .header-box {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            border: 2px solid #00c3ff;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 0 15px #00c3ff88;
        }

        .header-box h2 {
            font-size: 30px;
            color: #fff;
            margin: 0;
            font-weight: bold;
        }

        .glow-header {
            font-size: 22px;
            text-align: center;
            color: #00c3ff;
            text-shadow: 0 0 10px #00c3ff;
            margin-top: 10px;
            margin-bottom: 5px;
            font-weight: 600;
        }

        .stRadio > div {
            flex-direction: row !important;
            justify-content: center !important;
            gap: 12px;
        }

        .stRadio label {
            background: #1a1a1a;
            border: 1px solid #00c3ff;
            color: #00c3ff;
            padding: 10px 20px;
            margin: 4px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
            min-width: 180px;
            text-align: center;
        }

        .stRadio label:hover {
            background-color: #00c3ff33;
        }

        .stRadio input:checked + div > label {
            background-color: #00c3ff;
            color: #000;
            font-weight: bold;
        }

        .card {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border: 2px solid #00c3ff;
            border-radius: 15px;
            padding: 15px 20px;
            margin: 10px 0;
            box-shadow: 0 0 15px #00c3ff88;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: scale(1.02);
            box-shadow: 0 0 25px #00c3ffcc;
        }

        .card a {
            color: #00c3ff;
            font-weight: bold;
            font-size: 16px;
            text-decoration: none;
        }

        .card a:hover {
            color: #ffffff;
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="header-box">
            <h2>📚 Recommended Learning Hub</h2>
        </div>
    """, unsafe_allow_html=True)

    # Subheader
    st.markdown('<div class="glow-header">🎓 Explore Career Resources</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ccc;'>Curated courses and videos for your career growth, resume tips, and interview success.</p>", unsafe_allow_html=True)

    # Learning path label
    st.markdown("""
        <div style="text-align:center; margin-top: 25px; margin-bottom: 10px;">
            <span style="color: #00c3ff; font-weight: bold; font-size: 20px; text-shadow: 0 0 10px #00c3ff;">
                🧭 Choose Your Learning Path
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Centered Radio buttons
    st.markdown("""
        <div style="display: flex; justify-content: center; width: 100%;">
            <div style="display: flex; justify-content: center; gap: 16px;">
    """, unsafe_allow_html=True)

    page = st.radio(
        label="Select Learning Option",
        options=["Courses by Role", "Resume Videos", "Interview Videos"],
        horizontal=True,
        key="page_selection",
        label_visibility="collapsed"
    )

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Section 1: Courses by Role
    if page == "Courses by Role":
        st.subheader("🎯 Courses by Career Role")
        category = st.selectbox(
            "Select Career Category",
            options=list(COURSES_BY_CATEGORY.keys()),
            key="category_selection"
        )
        if category:
            roles = list(COURSES_BY_CATEGORY[category].keys())
            role = st.selectbox(
                "Select Role / Job Title",
                options=roles,
                key="role_selection"
            )
            if role:
                st.subheader(f"📘 Courses for **{role}** in **{category}**:")
                courses = get_courses_for_role(category, role)
                if courses:
                    for title, url in courses:
                        st.markdown(f"""
                            <div class="card">
                                <a href="{url}" target="_blank">🔗 {title}</a>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🚫 No courses found for this role.")

    # Section 2: Resume Videos
    elif page == "Resume Videos":
        st.subheader("📄 Resume Writing Videos")
        categories = list(RESUME_VIDEOS.keys())
        selected_cat = st.selectbox(
            "Select Resume Video Category",
            options=categories,
            key="resume_vid_cat"
        )
        if selected_cat:
            st.subheader(f"📂 {selected_cat}")
            videos = RESUME_VIDEOS[selected_cat]
            cols = st.columns(2)
            for idx, (title, url) in enumerate(videos):
                with cols[idx % 2]:
                    st.markdown(f"**{title}**")
                    st.video(url)

    # Section 3: Interview Videos
    elif page == "Interview Videos":
        st.subheader("🗣️ Interview Preparation Videos")
        categories = list(INTERVIEW_VIDEOS.keys())
        selected_cat = st.selectbox(
            "Select Interview Video Category",
            options=categories,
            key="interview_vid_cat"
        )
        if selected_cat:
            st.subheader(f"📂 {selected_cat}")
            videos = INTERVIEW_VIDEOS[selected_cat]
            cols = st.columns(2)
            for idx, (title, url) in enumerate(videos):
                with cols[idx % 2]:
                    st.markdown(f"**{title}**")
                    st.video(url)


with tab5:
    import sqlite3
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st

    from db_manager import (
        get_top_domains_by_score,
        get_resume_count_by_day,
        get_average_ats_by_domain,
        get_domain_distribution,
        get_bias_distribution,
        filter_candidates_by_date,
        delete_candidate_by_id,
        get_all_candidates,
        get_candidate_by_id,
    )

    def draw_clear_pie_chart(df):
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            df["count"],
            labels=df["domain"],
            autopct="%1.1f%%",
            startangle=90,
            textprops=dict(color="black", fontsize=8),
            pctdistance=0.8,
            labeldistance=1.1
        )
        ax.axis("equal")
        for t in autotexts:
            t.set_fontsize(7)
            t.set_color("white")
            t.set_weight("bold")
        st.pyplot(fig)

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.markdown("## 🔐 Admin Login Required")
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == "lexiadmin123":
                st.session_state.admin_logged_in = True
                st.success("✅ Login successful! You now have access to the Admin Dashboard.")
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
        st.stop()

    st.markdown("## 🛡️ <span style='color:#336699;'>Admin Database Panel</span>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    with col2:
        if st.button("🚪 Logout now"):
            st.session_state.admin_logged_in = False
            st.success("👋 Logged out successfully.")
            st.rerun()

    st.markdown("<hr style='border-top: 2px solid #bbb;'>", unsafe_allow_html=True)

    conn = sqlite3.connect("resume_data.db")
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY timestamp DESC", conn)

    search = st.text_input("🔍 Search Candidate by Name")
    if search:
        df = df[df["candidate_name"].str.contains(search, case=False, na=False)]

    st.markdown("### 📅 Filter by Date")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    if st.button("Apply Date Filter"):
        df = filter_candidates_by_date(str(start_date), str(end_date))

    if df.empty:
        st.info("ℹ️ No candidate data available.")
    else:
        st.markdown("### 📋 All Candidates")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="📥 Download Full CSV",
            data=df.to_csv(index=False),
            file_name="all_candidates.csv",
            mime="text/csv"
        )

        st.markdown("### 🗑️ Delete Candidate by ID")
        delete_id = st.number_input("Enter Candidate ID", min_value=1, step=1)
        if st.button("❌ Delete Candidate"):
            if delete_id in df["id"].values:
                st.warning(get_candidate_by_id(delete_id).to_markdown(index=False), icon="📄")
                delete_candidate_by_id(delete_id)
                st.success(f"✅ Candidate with ID {delete_id} deleted.")
                st.rerun()
            else:
                st.error("ID not found.")

    st.markdown("### 📊 Top Domains by ATS Score")
    top_domains = get_top_domains_by_score()
    if top_domains:
        for domain, avg, count in top_domains:
            st.info(f"📁 {domain} — Avg ATS: {avg:.2f} | Total: {count}")
    else:
        st.info("No domain data available.")

    st.markdown("### 📊 Domain Distribution by Count")
    df_domain_dist = get_domain_distribution()

    if not df_domain_dist.empty:
        total_count = df_domain_dist["count"].sum()
        df_domain_dist["percent"] = (df_domain_dist["count"] / total_count) * 100

        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        bars = ax_dist.bar(df_domain_dist["domain"], df_domain_dist["count"], color="#ff9933")

        # Show actual count on Y-axis aligned with bar top
        for bar in bars:
            height = bar.get_height()
            ax_dist.axhline(y=height, color='gray', linestyle=':', linewidth=0.5)
            ax_dist.text(
                -0.5,
                height,
                f"{int(height)}",
                va='center',
                ha='right',
                fontsize=9,
                color="gray"
            )

        # Show percentage on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percent = df_domain_dist["percent"].iloc[i]
            ax_dist.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{percent:.1f}%",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color="black"
            )

        ax_dist.set_ylabel("Resume Count", fontsize=12, fontweight='bold')
        ax_dist.set_title("Resumes per Domain", fontsize=14, fontweight='bold')
        ax_dist.set_xticks(np.arange(len(df_domain_dist["domain"])))
        ax_dist.set_xticklabels(df_domain_dist["domain"], rotation=30, ha="right", fontsize=10)
        max_count = df_domain_dist["count"].max()
        ax_dist.set_yticks(np.arange(0, max_count + 6, 5))

        st.pyplot(fig_dist)
    else:
        st.info("No domain data found.")

    st.markdown("### 📊 Average ATS Score by Domain")
    df_bar = get_average_ats_by_domain()
    if not df_bar.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.bar(df_bar["domain"], df_bar["avg_ats_score"], color="#3399ff")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}",
                     ha='center', va='bottom', fontsize=8)

        ax2.set_ylabel("Avg ATS Score")
        ax2.set_title("ATS by Domain")
        max_score = df_bar["avg_ats_score"].max()
        ax2.set_yticks(np.arange(0, max_score + 5, 5))
        ax2.set_xticks(np.arange(len(df_bar["domain"])))
        ax2.set_xticklabels(df_bar["domain"], rotation=45, ha="right")
        st.pyplot(fig2)
    else:
        st.info("No ATS domain data.")

    st.markdown("### 📈 Resume Uploads Over Time")
    df_timeline = get_resume_count_by_day()
    if not df_timeline.empty:
        df_timeline = df_timeline.sort_values("day")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(df_timeline["day"], df_timeline["count"], marker="o", color="green", linewidth=2)
        for i in range(len(df_timeline)):
            ax3.annotate(
                str(df_timeline["count"].iloc[i]),
                (df_timeline["day"].iloc[i], df_timeline["count"].iloc[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=9,
                color='black'
            )
        ax3.set_ylabel("Uploads")
        ax3.set_xlabel("Date")
        ax3.set_title("Resume Upload Timeline")
        plt.xticks(rotation=60, ha='right')
        fig3.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("ℹ️ No upload trend data to display.")

    st.markdown("### 🧠 Fair vs Biased Resumes")
    df_bias = get_bias_distribution()
    if not df_bias.empty:
        fig4, ax4 = plt.subplots()
        ax4.pie(df_bias["count"], labels=df_bias["bias_category"], autopct="%1.1f%%", startangle=90,
                colors=["#ff6666", "#00cc66"])
        ax4.axis("equal")
        st.pyplot(fig4)
    else:
        st.info("No bias data to display.")

    st.markdown("### 🚩 Flagged Candidates (Bias Score > 0.6)")
    flagged_df = get_all_candidates(bias_threshold=0.6)
    if not flagged_df.empty:
        st.dataframe(
    flagged_df[[
        "id", "resume_name", "candidate_name",
        "bias_score", "ats_score", "domain", "timestamp"
    ]].sort_values(by="bias_score", key=lambda x: pd.to_numeric(x, errors="coerce"), ascending=False),
    use_container_width=True
)

    else:
        st.success("✅ No flagged candidates.")


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
