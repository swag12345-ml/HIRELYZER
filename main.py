from xhtml2pdf import pisa
from io import BytesIO

def html_to_pdf_bytes(html_string):
    styled_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: 400mm 297mm;  /* Original custom large page size */
                margin-top: 10mm;
                margin-bottom: 10mm;
                margin-left: 10mm;
                margin-right: 10mm;
            }}
            body {{
                font-size: 14pt;
                font-family: "Segoe UI", "Helvetica", sans-serif;
                line-height: 1.5;
                color: #000;
            }}
            h1, h2, h3 {{
                color: #2f4f6f;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 15px;
            }}
            td {{
                padding: 4px;
                vertical-align: top;
                border: 1px solid #ccc;
            }}
            .section-title {{
                background-color: #e0e0e0;
                font-weight: bold;
                padding: 6px;
                margin-top: 10px;
            }}
            .box {{
                padding: 8px;
                margin-top: 6px;
                background-color: #f9f9f9;
                border-left: 4px solid #999;  /* More elegant than full border */
            }}
            ul {{
                margin: 0.5em 0;
                padding-left: 1.5em;
            }}
            li {{
                margin-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        {html_string}
    </body>
    </html>
    """

    pdf_io = BytesIO()
    pisa.CreatePDF(styled_html, dest=pdf_io)
    pdf_io.seek(0)
    return pdf_io




def generate_cover_letter_from_resume_builder():
    import streamlit as st
    from datetime import datetime
    import re
    from llm_manager import call_llm  # Ensure you import this

    name = st.session_state.get("name", "")
    job_title = st.session_state.get("job_title", "")
    summary = st.session_state.get("summary", "")
    skills = st.session_state.get("skills", "")
    location = st.session_state.get("location", "")
    today_date = datetime.today().strftime("%B %d, %Y")

    # ✅ Input boxes for contact info
    company = st.text_input("🏢 Target Company", placeholder="e.g., Google")
    linkedin = st.text_input("🔗 LinkedIn URL", placeholder="e.g., https://linkedin.com/in/username")
    email = st.text_input("📧 Email", placeholder="e.g., you@example.com")
    mobile = st.text_input("📞 Mobile Number", placeholder="e.g., +91 9876543210")

    # ✅ Button to prevent relooping
    if st.button("✉️ Generate Cover Letter"):
        # ✅ Validate input before generating
        if not all([name, job_title, summary, skills, company, linkedin, email, mobile]):
            st.warning("⚠️ Please fill in all fields including LinkedIn, email, and mobile.")
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
- Return only the final formatted cover letter without any HTML tags.
"""

        # ✅ Call LLM
        cover_letter = call_llm(prompt, session=st.session_state)

        # ✅ Clean leading line if needed
        lines = cover_letter.strip().split("\n")
        if len(lines) > 0 and (re.match(r'^\w+ \d{1,2}, \d{4}$', lines[0].strip()) or lines[0].strip().startswith('<div')):
            lines = lines[1:]
        cover_letter = "\n".join(lines)
        st.session_state["cover_letter"] = cover_letter

        # ✅ xhtml2pdf-compatible table layout
        cover_letter_html = f"""
<table width="100%" style="font-family: Georgia, serif; font-size: 12pt; color: #000;">
    <tr>
        <td align="center" colspan="2" style="font-size: 16pt; font-weight: bold; color: #003366;">{name}</td>
    </tr>
    <tr>
        <td align="center" colspan="2" style="font-size: 14pt; color: #555;">{job_title}</td>
    </tr>
    <tr>
        <td align="center" colspan="2" style="font-size: 10pt; padding: 5px;">
            <a href="{linkedin}" style="color: #003366;">{linkedin}</a><br/>
            📧 {email} | 📞 {mobile}
        </td>
    </tr>
    <tr><td colspan="2" style="padding-top:10px;">{today_date}</td></tr>
    <tr>
        <td colspan="2">
            <hr style="border: 0; border-top: 1px solid #000; margin: 10px 0;" />
        </td>
    </tr>
    <tr>
        <td colspan="2" style="white-space: pre-wrap; line-height: 1.6;">{cover_letter}</td>
    </tr>
</table>
"""
        st.session_state["cover_letter_html"] = cover_letter_html




import streamlit as st
import streamlit.components.v1 as components
from base64 import b64encode
import requests
import datetime
from io import BytesIO
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components
from base64 import b64encode
import re
from llm_manager import call_llm
import requests
import datetime
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

import base64
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

from user_login import (
    create_user_table,
    add_user,
    verify_user,
    get_logins_today,
    get_total_registered_users,
    log_user_action,
    username_exists  # 👈 add this line
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
# 🔹 VIDEO BACKGROUND & GLOW TEXT




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

    # -------- Counter Section (Updated Layout & Style with tighter spacing) --------

    # Fetch counters
    total_users = get_total_registered_users()
    active_logins = get_logins_today()
    resumes_uploaded = 15
    states_accessed = 29

    neon_counter_style = """
    <style>
    .counter-grid {
        display: grid;
        grid-template-columns: repeat(2, 250px);
        column-gap: 40px;
        row-gap: 25px;
        justify-content: center;
        padding: 30px 10px;
        max-width: 600px;
        margin: 0 auto;
    }

    .counter-box {
        background-color: #0d1117;
        border: 2px solid #00FFFF;
        border-radius: 10px;
        width: 100%;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 0 12px rgba(0, 255, 255, 0.4);
        transition: transform 0.2s ease;
    }

    .counter-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 18px rgba(0, 255, 255, 0.7);
    }

    .counter-number {
        font-size: 2.2em;
        font-weight: bold;
        color: #00BFFF;
        margin: 0;
    }

    .counter-label {
        margin-top: 8px;
        font-size: 1em;
        color: #c9d1d9;
    }
    </style>
    """

    st.markdown(neon_counter_style, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="counter-grid">
        <div class="counter-box">
            <div class="counter-number">{total_users}</div>
            <div class="counter-label">Total Users</div>
        </div>
        <div class="counter-box">
            <div class="counter-number">{states_accessed}</div>
            <div class="counter-label">States Accessed</div>
        </div>
        <div class="counter-box">
            <div class="counter-number">{resumes_uploaded}</div>
            <div class="counter-label">Resumes Uploaded</div>
        </div>
        <div class="counter-box">
            <div class="counter-number">{active_logins}</div>
            <div class="counter-label">Active Sessions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)





if not st.session_state.get("authenticated", False):
    

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
        st.markdown(
            "<div class='login-card'><h2 style='text-align:center;'>🔐 Login to <span style='color:#00BFFF;'>LEXIBOT</span></h2>",
            unsafe_allow_html=True,
        )

        login_tab, register_tab = st.tabs(["🔑 Login", "🆕 Register"])

        # ---------------- LOGIN TAB ----------------
        with login_tab:
            user = st.text_input("Username", key="login_user")
            pwd = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", key="login_btn"):
                success, saved_key = verify_user(user.strip(), pwd.strip())
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = user.strip()

                    # ✅ Load saved Groq key into session
                    if saved_key:
                        st.session_state["user_groq_key"] = saved_key

                    log_user_action(user.strip(), "login")
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")

        # ---------------- REGISTER TAB ----------------
        with register_tab:
            new_user = st.text_input("Choose a Username", key="reg_user")
            new_pass = st.text_input("Choose a Password", type="password", key="reg_pass")
            st.caption("🔒 Password must be at least 8 characters and include uppercase, lowercase, number, and special character.")

            # ✅ Live Username Availability Check
            if new_user.strip():
                if username_exists(new_user.strip()):
                    st.error("🚫 Username already exists.")
                else:
                    st.info("✅ Username is available.")

            if st.button("Register", key="register_btn"):
                if new_user.strip() and new_pass.strip():
                    success, message = add_user(new_user.strip(), new_pass.strip())
                    if success:
                        st.success(message)
                        log_user_action(new_user.strip(), "register")
                    else:
                        st.error(message)
                else:
                    st.warning("⚠️ Please fill in both fields.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()




# ------------------- AFTER LOGIN -------------------
from user_login import save_user_api_key, get_user_api_key  # Ensure both are imported

if st.session_state.get("authenticated"):
    st.markdown(
        f"<h2 style='color:#00BFFF;'>Welcome to LEXIBOT, <span style='color:white;'>{st.session_state.username}</span> 👋</h2>",
        unsafe_allow_html=True,
    )

    # 🔓 LOGOUT BUTTON
    if st.button("🚪 Logout"):
        log_user_action(st.session_state.get("username", "unknown"), "logout")

        # ✅ Clear all session keys safely
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.success("✅ Logged out successfully.")
        st.rerun()  # Force rerun to prevent stale UI

    # 🔑 GROQ API KEY SECTION (SIDEBAR)
    st.sidebar.markdown("### 🔑 Groq API Key")

    # ✅ Load saved key from DB
    saved_key = get_user_api_key(st.session_state.username)
    masked_preview = f"****{saved_key[-6:]}" if saved_key else ""

    user_api_key_input = st.sidebar.text_input(
        "Your Groq API Key (Optional)",
        placeholder=masked_preview,
        type="password"
    )

    # ✅ Save or reuse key
    if user_api_key_input:
        st.session_state["user_groq_key"] = user_api_key_input
        save_user_api_key(st.session_state.username, user_api_key_input)
        st.sidebar.success("✅ New key saved and in use.")
    elif saved_key:
        st.session_state["user_groq_key"] = saved_key
        st.sidebar.info(f"ℹ️ Using your previously saved API key ({masked_preview})")
    else:
        st.sidebar.warning("⚠ Using shared admin key with possible usage limits")

    # 🧹 Clear saved key
    if st.sidebar.button("🗑️ Clear My API Key"):
        st.session_state["user_groq_key"] = None
        save_user_api_key(st.session_state.username, None)
        st.sidebar.success("✅ Cleared saved Groq API key. Now using shared admin key.")






from user_login import get_all_user_logs, get_total_registered_users, get_logins_today
import streamlit as st

if st.session_state.username == "admin":
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#00BFFF;'>📊 Admin Dashboard</h2>", unsafe_allow_html=True)

    # Metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👤 Total Registered Users", get_total_registered_users())
    with col2:
        st.metric("📅 Logins Today (IST)", get_logins_today())

    # Removed API key usage section (no longer tracked)
    # Activity log
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
from llm_manager import call_llm


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
    # Encode values for query
    role_encoded = urllib.parse.quote_plus(job_role.strip())
    loc_encoded = urllib.parse.quote_plus(location.strip())

    # Create role/city slugs for path
    role_path = job_role.strip().lower().replace(" ", "-")
    city = location.strip().split(",")[0].strip().lower()
    city_path = city.replace(" ", "-")
    city_query = city.replace(" ", "%20") + "%20and%20india"

    # Experience mappings
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

    # Experience values
    if foundit_experience is not None:
        experience_range = f"{foundit_experience}~{foundit_experience}"
        experience_exact = str(foundit_experience)
    else:
        experience_range = experience_range_map.get(experience_level, "")
        experience_exact = experience_exact_map.get(experience_level, "")

    # ✅ Naukri (cleaned)
    naukri_url = (
        f"https://www.naukri.com/{role_path}-jobs-in-{city_path}-and-india"
        f"?k={role_encoded}"
        f"&l={city_query}"
    )
    if experience_exact:
        naukri_url += f"&experience={experience_exact}"
    naukri_url += "&nignbevent_src=jobsearchDeskGNB"

    # ✅ FoundIt
    search_id = uuid.uuid4()
    child_search_id = uuid.uuid4()
    foundit_url = (
        f"https://www.foundit.in/search/{role_path}-jobs-in-{city_path}"
        f"?query={role_encoded}"
        f"&locations={loc_encoded}"
        f"&experienceRanges={urllib.parse.quote_plus(experience_range)}"
        f"&experience={experience_exact}"
        f"&queryDerived=true"
        f"&searchId={search_id}"
        f"&child_search_id={child_search_id}"
    )

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

import re

# Predefined gender-coded word lists
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
    # Split into sentences using simple delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    masc_set, fem_set = set(), set()
    masculine_found, feminine_found = [] , []

    masculine_words = sorted(gender_words["masculine"], key=len, reverse=True)
    feminine_words = sorted(gender_words["feminine"], key=len, reverse=True)

    for sent in sentences:
        sent_text = sent.strip()
        sent_lower = sent_text.lower()
        matched_spans = []

        def is_overlapping(start, end):
            return any(start < e and end > s for s, e in matched_spans)

        # 🔵 Highlight masculine words in blue
        for word in masculine_words:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            for match in pattern.finditer(sent_lower):
                start, end = match.span()
                if not is_overlapping(start, end):
                    matched_spans.append((start, end))
                    key = (word.lower(), sent_text)
                    if key not in masc_set:
                        masc_set.add(key)
                        highlighted = re.sub(
                            rf'\b({re.escape(word)})\b',
                            r'<span style="color:blue;">\1</span>',
                            sent_text,
                            flags=re.IGNORECASE
                        )
                        masculine_found.append({
                            "word": word,
                            "sentence": highlighted
                        })

        # 🔴 Highlight feminine words in red
        for word in feminine_words:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            for match in pattern.finditer(sent_lower):
                start, end = match.span()
                if not is_overlapping(start, end):
                    matched_spans.append((start, end))
                    key = (word.lower(), sent_text)
                    if key not in fem_set:
                        fem_set.add(key)
                        highlighted = re.sub(
                            rf'\b({re.escape(word)})\b',
                            r'<span style="color:red;">\1</span>',
                            sent_text,
                            flags=re.IGNORECASE
                        )
                        feminine_found.append({
                            "word": word,
                            "sentence": highlighted
                        })

    masc = len(masculine_found)
    fem = len(feminine_found)
    total = masc + fem
    bias_score = min(total / 20, 1.0) if total > 0 else 0.0

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


import re

def rewrite_and_highlight(text, replacement_mapping, user_location):
    highlighted_text = text
    masculine_count, feminine_count = 0, 0
    detected_masculine_words, detected_feminine_words = [], []
    matched_spans = []

    masculine_words = sorted(gender_words["masculine"], key=len, reverse=True)
    feminine_words = sorted(gender_words["feminine"], key=len, reverse=True)

    def span_overlaps(start, end):
        return any(s < end and e > start for s, e in matched_spans)

    # Highlight and count masculine words
    for word in masculine_words:
        pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
        for match in pattern.finditer(highlighted_text):
            start, end = match.span()
            if span_overlaps(start, end):
                continue

            word_match = match.group(0)
            colored = f"<span style='color:blue;'>{word_match}</span>"

            # Replace word in the highlighted text
            highlighted_text = highlighted_text[:start] + colored + highlighted_text[end:]
            shift = len(colored) - len(word_match)
            matched_spans = [(s if s < start else s + shift, e if s < start else e + shift) for s, e in matched_spans]
            matched_spans.append((start, start + len(colored)))

            masculine_count += 1

            # Get sentence context and highlight
            sentence_match = re.search(r'([^.]*?\b' + re.escape(word_match) + r'\b[^.]*\.)', text, re.IGNORECASE)
            if sentence_match:
                sentence = sentence_match.group(1).strip()
                colored_sentence = re.sub(
                    rf'\b({re.escape(word_match)})\b',
                    r"<span style='color:blue;'>\1</span>",
                    sentence,
                    flags=re.IGNORECASE
                )
                detected_masculine_words.append({
                    "word": word_match,
                    "sentence": colored_sentence
                })
            break  # Only one match per word

    # Highlight and count feminine words
    for word in feminine_words:
        pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
        for match in pattern.finditer(highlighted_text):
            start, end = match.span()
            if span_overlaps(start, end):
                continue

            word_match = match.group(0)
            colored = f"<span style='color:red;'>{word_match}</span>"

            # Replace word in the highlighted text
            highlighted_text = highlighted_text[:start] + colored + highlighted_text[end:]
            shift = len(colored) - len(word_match)
            matched_spans = [(s if s < start else s + shift, e if s < start else e + shift) for s, e in matched_spans]
            matched_spans.append((start, start + len(colored)))

            feminine_count += 1

            # Get sentence context and highlight
            sentence_match = re.search(r'([^.]*?\b' + re.escape(word_match) + r'\b[^.]*\.)', text, re.IGNORECASE)
            if sentence_match:
                sentence = sentence_match.group(1).strip()
                colored_sentence = re.sub(
                    rf'\b({re.escape(word_match)})\b',
                    r"<span style='color:red;'>\1</span>",
                    sentence,
                    flags=re.IGNORECASE
                )
                detected_feminine_words.append({
                    "word": word_match,
                    "sentence": colored_sentence
                })
            break  # Only one match per word

    # Rewrite text with neutral terms
    rewritten_text = rewrite_text_with_llm(
        text,
        replacement_mapping["masculine"] | replacement_mapping["feminine"],
        user_location
    )

    return highlighted_text, rewritten_text, masculine_count, feminine_count, detected_masculine_words, detected_feminine_words

import re
import pandas as pd
import altair as alt
import streamlit as st
from llm_manager import call_llm
from db_manager import detect_domain_from_title_and_description, get_domain_similarity

# ✅ Enhanced Grammar evaluation using LLM with suggestions
def get_grammar_score_with_llm(text, max_score=5):
    grammar_prompt = f"""
You are a grammar and tone evaluator AI. Analyze the following resume text and:

1. Give a grammar score out of {max_score} based on grammar quality, sentence structure, clarity, and tone.
2. Return a 1-sentence summary of the grammar and tone.
3. Provide 3 to 5 **specific improvement suggestions** (bullet points) for enhancing grammar, clarity, tone, or structure.

Return response in the exact format below:

Score: <number>
Feedback: <summary>
Suggestions:
- <suggestion 1>
- <suggestion 2>
...

---
{text}
---
"""
    response = call_llm(grammar_prompt, session=st.session_state).strip()
    score_match = re.search(r"Score:\s*(\d+)", response)
    feedback_match = re.search(r"Feedback:\s*(.+)", response)
    suggestions = re.findall(r"- (.+)", response)

    score = int(score_match.group(1)) if score_match else 3
    feedback = feedback_match.group(1).strip() if feedback_match else "No grammar feedback provided."
    return score, feedback, suggestions


# ✅ Main ATS Evaluation Function (Balanced Scoring)
def ats_percentage_score(
    resume_text,
    job_description,
    job_title="Unknown",
    logic_profile_score=None,
    edu_weight=20,
    exp_weight=35,
    skills_weight=30,
    lang_weight=5,
    keyword_weight=10
):
    grammar_score, grammar_feedback, grammar_suggestions = get_grammar_score_with_llm(resume_text, max_score=lang_weight)

    resume_domain = detect_domain_from_title_and_description("Unknown", resume_text)
    job_domain = detect_domain_from_title_and_description(job_title, job_description)
    similarity_score = get_domain_similarity(resume_domain, job_domain)

    MAX_DOMAIN_PENALTY = 15
    domain_penalty = round((1 - similarity_score) * MAX_DOMAIN_PENALTY)

    logic_score_note = (
        f"\n\nOptional Note: The system also calculated a logic-based profile score of {logic_profile_score}/100 based on resume length, experience, and skills."
        if logic_profile_score else ""
    )

    prompt = f"""
You are an AI-powered ATS evaluator. Assess the candidate's resume against the job description. Return a detailed, **section-by-section analysis**, with **scoring for each area**. Follow the format precisely below.

🎯 Section Breakdown:

1. **Candidate Name** — Extract the full name clearly from the resume header or first few lines.

2. **Education Analysis** — Evaluate:
   - Degree level (e.g., Bachelor’s, Master’s, PhD)
   - Field of study alignment with job requirements
   - Institution reputation (if mentioned)
   - Graduation year (recency)
   - Any certifications or training programs relevant to the role

3. **Experience Analysis** — Scoring Rules (Balanced):
   - Award full points only if candidate meets or exceeds required years of relevant full-time experience.
   - Internships = 0.5 years each (max 2 points per internship).
   - Academic projects = max 1 point each if directly relevant.
   - If total calculated experience is less than JD requirement, cap score at 60% of {exp_weight}.
   - Strong, high-impact projects or leadership roles can partially offset lack of years.
   - Prioritize quality and relevance over number of roles.

4. **Skills Analysis** — Scoring Rules (Balanced):
   - Award higher points for technical/domain skills that directly match JD.
   - Give partial credit for related/transferable skills.
   - Deduct moderately for missing secondary skills; stronger deductions for missing core JD skills.
   - Highlight missing critical skills from JD (at least 3 if absent).
   - Consider recency of usage; outdated skills get reduced points.

5. **Language Quality** — Use grammar score provided. Evaluate:
   - Grammar and spelling quality
   - Tone (professional, casual, inconsistent)
   - Sentence clarity and structure
   - Use of active voice and action verbs
   - Formatting professionalism

6. **Keyword Analysis** — Scoring Rules (Balanced):
   - Identify critical keywords from JD — these should heavily influence the score.
   - Award partial credit for related terms if exact match not found.
   - Deduct more for missing essential keywords; less for secondary ones.
   - List missing JD keywords as bullet points (at least 3 if applicable).

7. **Final Thoughts** — Provide a 4–6 sentence holistic evaluation:
   - Resume's overall alignment with the job
   - Highlight major strengths
   - Point out red flags
   - Mention if the resume deserves shortlisting or further screening

Use this context:

- Grammar Score: {grammar_score} / {lang_weight}
- Grammar Feedback: {grammar_feedback}
- Resume Domain: {resume_domain}
- Job Domain: {job_domain}
- Penalty if domains don't match: {domain_penalty} (Based on domain similarity score {similarity_score:.2f}, max penalty is {MAX_DOMAIN_PENALTY})

---

### 🏷️ Candidate Name
<Full name or "Not Found">

### 🏫 Education Analysis
**Score:** <0–{edu_weight}> / {edu_weight}  
**Degree Match:** <Discuss degree level, specialization, and how it matches the job.>

### 💼 Experience Analysis
**Score:** <0–{exp_weight}> / {exp_weight}  
**Experience Details:**  
<Give breakdown of years, relevance, and notable achievements. Explicitly mention when years of experience are below the requirement.>

### 🛠 Skills Analysis
**Score:** <0–{skills_weight}> / {skills_weight}  
**Current Skills:**
- Technical: <list>
- Soft Skills: <list>
- Domain-Specific: <list>

**Skill Proficiency:**  
<Evaluate depth of knowledge and recency. Mention if skills are supported by projects or work experience.>

**Missing Skills:**  
- Skill 1  
- Skill 2  
- Skill 3  

### 🗣 Language Quality Analysis
**Score:** {grammar_score} / {lang_weight}  
**Grammar & Tone:** <LLM-based comment on clarity, fluency, tone>  
**Feedback Summary:** **{grammar_feedback}**

### 🔑 Keyword Analysis
**Score:** <0–{keyword_weight}> / {keyword_weight}  
**Missing Keywords:**  
- Keyword 1  
- Keyword 2  
- Keyword 3  

**Keyword Analysis:**  
<Discuss importance of missing/present keywords and how they affect job match.>

### ✅ Final Thoughts
<Summarize domain fit, core strengths, red flags, and whether this resume deserves further review.>

---

**Instructions:**
- Keep tone professional and ATS-focused.
- Follow the exact section structure.
- Always list missing skills and keywords if applicable.
- Be consistent — same resume and JD should always yield the same score.

📄 Job Description:
\"\"\"{job_description}\"\"\"  

📄 Resume:
\"\"\"{resume_text}\"\"\"  

{logic_score_note}
"""



    ats_result = call_llm(prompt, session=st.session_state).strip()

    def extract_section(pattern, text, default="N/A"):
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else default

    def extract_score(pattern, text, default=0):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else default

    # Extract key sections
    candidate_name = extract_section(r"### 🏷️ Candidate Name(.*?)###", ats_result, "Not Found")
    edu_analysis = extract_section(r"### 🏫 Education Analysis(.*?)###", ats_result)
    exp_analysis = extract_section(r"### 💼 Experience Analysis(.*?)###", ats_result)
    skills_analysis = extract_section(r"### 🛠 Skills Analysis(.*?)###", ats_result)
    lang_analysis = extract_section(r"### 🗣 Language Quality Analysis(.*?)###", ats_result)
    keyword_analysis = extract_section(r"### 🔑 Keyword Analysis(.*?)###", ats_result)
    final_thoughts = extract_section(r"### ✅ Final Thoughts(.*)", ats_result)

    # Extract scores
    edu_score = extract_score(r"\*\*Score:\*\*\s*(\d+)", edu_analysis)
    exp_score = extract_score(r"\*\*Score:\*\*\s*(\d+)", exp_analysis)
    skills_score = extract_score(r"\*\*Score:\*\*\s*(\d+)", skills_analysis)
    keyword_score = extract_score(r"\*\*Score:\*\*\s*(\d+)", keyword_analysis)

    missing_keywords = extract_section(r"\*\*Missing Keywords:\*\*(.*?)(?:###|\Z)", keyword_analysis).replace("-", "").strip().replace("\n", ", ")
    missing_skills = extract_section(r"\*\*Missing Skills:\*\*(.*?)(?:###|\Z)", skills_analysis).replace("-", "").strip().replace("\n", ", ")

    total_score = edu_score + exp_score + skills_score + grammar_score + keyword_score
    total_score = max(total_score - domain_penalty, 0)
    total_score = min(total_score, 100)

    formatted_score = (
        "🌟 Excellent" if total_score >= 85 else
        "✅ Good" if total_score >= 70 else
        "⚠️ Average" if total_score >= 50 else
        "❌ Poor"
    )

    # ✅ Format suggestions nicely
    suggestions_html = ""
    if grammar_suggestions:
        suggestions_html = "<ul>" + "".join([f"<li>{s}</li>" for s in grammar_suggestions]) + "</ul>"

    updated_lang_analysis = f"""
{lang_analysis}
<br><b>LLM Feedback Summary:</b> {grammar_feedback}
<br><b>Suggestions to Improve:</b> {suggestions_html}
"""

    final_thoughts += f"\n\n📉 Domain Similarity Score: {similarity_score:.2f}\n🔻 Domain Penalty Applied: {domain_penalty} / {MAX_DOMAIN_PENALTY}"

    return ats_result, {
        "Candidate Name": candidate_name,
        "Education Score": edu_score,
        "Experience Score": exp_score,
        "Skills Score": skills_score,
        "Language Score": grammar_score,
        "Keyword Score": keyword_score,
        "ATS Match %": total_score,
        "Formatted Score": formatted_score,
        "Education Analysis": edu_analysis,
        "Experience Analysis": exp_analysis,
        "Skills Analysis": skills_analysis,
        "Language Analysis": updated_lang_analysis,
        "Keyword Analysis": keyword_analysis,
        "Final Thoughts": final_thoughts,
        "Missing Keywords": missing_keywords,
        "Missing Skills": missing_skills,
        "Resume Domain": resume_domain,
        "Job Domain": job_domain,
        "Domain Penalty": domain_penalty,
        "Domain Similarity Score": similarity_score
    }
# Setup Vector DB
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    return FAISS.from_texts(doc_chunks, embeddings)

# Create Conversational Chain
from llm_manager import load_groq_api_keys

def create_chain(vectorstore):
    # 🔁 Get a rotated admin key
    keys = load_groq_api_keys()
    index = st.session_state.get("key_index", 0)
    groq_api_key = keys[index % len(keys)]
    st.session_state["key_index"] = index + 1

    # ✅ Create the ChatGroq object
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)

    # ✅ Build the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return chain

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
import pandas as pd
import altair as alt
from datetime import datetime
from db_manager import insert_candidate, detect_domain_from_title_and_description
from llm_manager import call_llm  # ensure this calls your LLM

# ✅ Initialize state
if "resume_data" not in st.session_state:
    st.session_state.resume_data = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

resume_data = st.session_state.resume_data

# ✏️ Resume Evaluation Logic
if uploaded_files and job_description:
    with st.spinner("✨ Creating magic for you... Hold on a minute!"):
        all_text = []

        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.processed_files:
                continue

            # ✅ Save uploaded file
            file_path = os.path.join(working_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ✅ Extract text from PDF
            text = extract_text_from_pdf(file_path)
            if not text:
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}. Skipping.")
                continue

            all_text.append(" ".join(text))
            full_text = " ".join(text)

            # ✅ Bias detection
            bias_score, masc_count, fem_count, detected_masc, detected_fem = detect_bias(full_text)

            # ✅ Rewrite and highlight gender-biased words
            highlighted_text, rewritten_text, _, _, _, _ = rewrite_and_highlight(
                full_text, replacement_mapping, user_location
            )

            # ✅ LLM-based ATS Evaluation
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

            # ✅ Extract structured ATS values
            candidate_name = ats_scores.get("Candidate Name", "Not Found")
            ats_score = ats_scores.get("ATS Match %", 0)
            edu_score = ats_scores.get("Education Score", 0)
            exp_score = ats_scores.get("Experience Score", 0)
            skills_score = ats_scores.get("Skills Score", 0)
            lang_score = ats_scores.get("Language Score", 0)
            keyword_score = ats_scores.get("Keyword Score", 0)
            formatted_score = ats_scores.get("Formatted Score", "N/A")
            fit_summary = ats_scores.get("Final Thoughts", "N/A")
            language_analysis_full = ats_scores.get("Language Analysis", "N/A")

            missing_keywords_raw = ats_scores.get("Missing Keywords", "N/A")
            missing_skills_raw = ats_scores.get("Missing Skills", "N/A")
            missing_keywords = [kw.strip() for kw in missing_keywords_raw.split(",") if kw.strip()] if missing_keywords_raw != "N/A" else []
            missing_skills = [sk.strip() for sk in missing_skills_raw.split(",") if sk.strip()] if missing_skills_raw != "N/A" else []

            domain = detect_domain_from_title_and_description(job_title, job_description)

            bias_flag = "🔴 High Bias" if bias_score > 0.6 else "🟢 Fair"
            ats_flag = "⚠️ Low ATS" if ats_score < 50 else "✅ Good ATS"

            # 📊 ATS Chart
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

            # ✅ Store everything in session state
            st.session_state.resume_data.append({
                "Resume Name": uploaded_file.name,
                "Candidate Name": candidate_name,
                "ATS Report": ats_result,
                "ATS Match %": ats_score,
                "Formatted Score": formatted_score,
                "Education Score": edu_score,
                "Experience Score": exp_score,
                "Skills Score": skills_score,
                "Language Score": lang_score,
                "Keyword Score": keyword_score,
                "Education Analysis": ats_scores.get("Education Analysis", ""),
                "Experience Analysis": ats_scores.get("Experience Analysis", ""),
                "Skills Analysis": ats_scores.get("Skills Analysis", ""),
                "Language Analysis": language_analysis_full,
                "Keyword Analysis": ats_scores.get("Keyword Analysis", ""),
                "Final Thoughts": fit_summary,
                "Missing Keywords": missing_keywords,
                "Missing Skills": missing_skills,
                "Bias Score (0 = Fair, 1 = Biased)": bias_score,
                "Bias Status": bias_flag,
                "Masculine Words": masc_count,
                "Feminine Words": fem_count,
                "Detected Masculine Words": detected_masc,
                "Detected Feminine Words": detected_fem,
                "Text Preview": full_text[:300] + "...",
                "Highlighted Text": highlighted_text,
                "Rewritten Text": rewritten_text,
                "Domain": domain
            })

            insert_candidate(
                (
                    uploaded_file.name,
                    candidate_name,
                    ats_score,
                    edu_score,
                    exp_score,
                    skills_score,
                    lang_score,
                    keyword_score,
                    bias_score
                ),
                job_title=job_title,
                job_description=job_description
            )

            st.session_state.processed_files.add(uploaded_file.name)

    st.success("✅ All resumes processed!")




    # ✅ Optional vectorstore setup
    if all_text:
        st.session_state.vectorstore = setup_vectorstore(all_text)
        st.session_state.chain = create_chain(st.session_state.vectorstore)

# 🔄 Developer Reset Button
if st.button("🔄 Reset Resume Upload Memory"):
    st.session_state.processed_files.clear()
    st.session_state.resume_data.clear()
    st.success("✅ Cleared uploaded resume history. You can re-upload now.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🧾 Resume Builder", "💼 Job Search", 
    "📚 Course Recommendation", "📁 Admin DB View"
])

def generate_resume_report_html(resume):
    candidate_name = resume.get('Candidate Name', 'Not Found')
    resume_name = resume.get('Resume Name', 'Unknown')
    rewritten_text = resume.get('Rewritten Text', '').replace("\n", "<br/>")

    masculine_words_list = resume.get("Detected Masculine Words", [])
    masculine_words = "".join(
        f"<b>{item.get('word','')}</b>: {item.get('sentence','')}<br/>"
        for item in masculine_words_list
    ) if masculine_words_list else "<i>None detected.</i>"

    feminine_words_list = resume.get("Detected Feminine Words", [])
    feminine_words = "".join(
        f"<b>{item.get('word','')}</b>: {item.get('sentence','')}<br/>"
        for item in feminine_words_list
    ) if feminine_words_list else "<i>None detected.</i>"

    ats_report_html = resume.get("ATS Report", "").replace("\n", "<br/>")

    def style_analysis(analysis, fallback="N/A"):
        if not analysis or analysis == "N/A":
            return f"<p><i>{fallback}</i></p>"

        if "**Score:**" in analysis:
            parts = analysis.split("**Score:**")
            rest = parts[1].split("**", 1)
            score_text = rest[0].strip()
            remaining = rest[1].strip() if len(rest) > 1 else ""
            return f"<p><b>Score:</b> {score_text}</p><p>{remaining}</p>"
        else:
            return f"<p>{analysis}</p>"

    edu_analysis = style_analysis(resume.get("Education Analysis", "").replace("\n", "<br/>"))
    exp_analysis = style_analysis(resume.get("Experience Analysis", "").replace("\n", "<br/>"))
    skills_analysis = style_analysis(resume.get("Skills Analysis", "").replace("\n", "<br/>"))
    keyword_analysis = style_analysis(resume.get("Keyword Analysis", "").replace("\n", "<br/>"))
    final_thoughts = resume.get("Final Thoughts", "N/A").replace("\n", "<br/>")

    lang_analysis_raw = resume.get("Language Analysis", "").replace("\n", "<br/>")
    lang_analysis = f"<div>{lang_analysis_raw}</div>" if lang_analysis_raw else "<p><i>No language analysis available.</i></p>"

    ats_match = resume.get('ATS Match %', 'N/A')
    edu_score = resume.get('Education Score', 'N/A')
    exp_score = resume.get('Experience Score', 'N/A')
    skills_score = resume.get('Skills Score', 'N/A')
    lang_score = resume.get('Language Score', 'N/A')
    keyword_score = resume.get('Keyword Score', 'N/A')
    masculine_count = len(masculine_words_list)
    feminine_count = len(feminine_words_list)
    bias_score = resume.get('Bias Score (0 = Fair, 1 = Biased)', 'N/A')

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Helvetica, sans-serif;
                font-size: 12pt;
                line-height: 1.5;
                color: #000;
            }}
            h1, h2 {{
                color: #2f4f6f;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 15px;
            }}
            td {{
                padding: 4px;
                vertical-align: top;
                border: 1px solid #ccc;
            }}
            ul {{
                margin: 0.5em 0;
                padding-left: 1.4em;
            }}
            li {{
                margin-bottom: 5px;
            }}
            .section-title {{
                background-color: #e0e0e0;
                font-weight: bold;
                padding: 6px;
                margin-top: 12px;
                border-left: 4px solid #666;
            }}
            .box {{
                padding: 10px;
                margin-top: 6px;
                background-color: #f9f9f9;
                border-left: 4px solid #999;
            }}
        </style>
    </head>
    <body>

    <h1>Resume Analysis Report</h1>

    <h2>Candidate: {candidate_name}</h2>
    <p><b>Resume File:</b> {resume_name}</p>

    <h2>ATS Evaluation</h2>
    <table>
        <tr><td><b>ATS Match</b></td><td>{ats_match}%</td></tr>
        <tr><td><b>Education</b></td><td>{edu_score}</td></tr>
        <tr><td><b>Experience</b></td><td>{exp_score}</td></tr>
        <tr><td><b>Skills</b></td><td>{skills_score}</td></tr>
        <tr><td><b>Language</b></td><td>{lang_score}</td></tr>
        <tr><td><b>Keyword</b></td><td>{keyword_score}</td></tr>
    </table>

    <div class="section-title">ATS Report</div>
    <div class="box">{ats_report_html}</div>

    <div class="section-title">Education Analysis</div>
    <div class="box">{edu_analysis}</div>

    <div class="section-title">Experience Analysis</div>
    <div class="box">{exp_analysis}</div>

    <div class="section-title">Skills Analysis</div>
    <div class="box">{skills_analysis}</div>

    <div class="section-title">Language Analysis</div>
    <div class="box">{lang_analysis}</div>

    <div class="section-title">Keyword Analysis</div>
    <div class="box">{keyword_analysis}</div>

    <div class="section-title">Final Thoughts</div>
    <div class="box">{final_thoughts}</div>

    <h2>Gender Bias Analysis</h2>
    <table>
        <tr><td><b>Masculine Words</b></td><td>{masculine_count}</td></tr>
        <tr><td><b>Feminine Words</b></td><td>{feminine_count}</td></tr>
        <tr><td><b>Bias Score (0 = Fair, 1 = Biased)</b></td><td>{bias_score}</td></tr>
    </table>

    <div class="section-title">Masculine Words Detected</div>
    <div class="box">{masculine_words}</div>

    <div class="section-title">Feminine Words Detected</div>
    <div class="box">{feminine_words}</div>

    <h2>Rewritten Bias-Free Resume</h2>
    <div class="box">{rewritten_text}</div>

    </body>
    </html>
    """




# === TAB 1: Dashboard ===
with tab1:
    resume_data = st.session_state.get("resume_data", [])

    if resume_data:
        # ✅ Calculate total counts safely
        total_masc = sum(len(r.get("Detected Masculine Words", [])) for r in resume_data)
        total_fem = sum(len(r.get("Detected Feminine Words", [])) for r in resume_data)
        avg_bias = round(np.mean([r.get("Bias Score (0 = Fair, 1 = Biased)", 0) for r in resume_data]), 2)
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

        # ✅ Add calculated count columns safely
        df["Masculine Words Count"] = df["Detected Masculine Words"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df["Feminine Words Count"] = df["Detected Feminine Words"].apply(lambda x: len(x) if isinstance(x, list) else 0)

        overview_cols = [
            "Resume Name", "Candidate Name", "ATS Match %", "Education Score",
            "Experience Score", "Skills Score", "Language Score", "Keyword Score",
            "Bias Score (0 = Fair, 1 = Biased)", "Masculine Words Count", "Feminine Words Count"
        ]

        st.dataframe(df[overview_cols], use_container_width=True)

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
            ax.bar(index, df["Masculine Words Count"], bar_width, label="Masculine", color="#3498db")
            ax.bar(index + bar_width, df["Feminine Words Count"], bar_width, label="Feminine", color="#e74c3c")
            ax.set_xlabel("Resumes", fontsize=12)
            ax.set_ylabel("Word Count", fontsize=12)
            ax.set_title("Gender-Coded Word Usage per Resume", fontsize=14)
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(df["Resume Name"], rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)

        st.markdown("### 📝 Detailed Resume Reports")
        for resume in resume_data:
            candidate_name = resume.get("Candidate Name", "Not Found")
            resume_name = resume.get("Resume Name", "Unknown")
            missing_keywords = resume.get("Missing Keywords", [])
            missing_skills = resume.get("Missing Skills", [])

            with st.expander(f"📄 {resume_name} | {candidate_name}"):
                st.markdown(f"### 📊 ATS Evaluation for: **{candidate_name}**")
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("📈 Overall Match", f"{resume.get('ATS Match %', 'N/A')}%")
                with score_col2:
                    st.metric("🏆 Formatted Score", resume.get("Formatted Score", "N/A"))
                with score_col3:
                    st.metric("🧠 Language Quality", f"{resume.get('Language Score', 'N/A')} / {lang_weight}")

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("🎓 Education Score", f"{resume.get('Education Score', 'N/A')} / {edu_weight}")
                with col_b:
                    st.metric("💼 Experience Score", f"{resume.get('Experience Score', 'N/A')} / {exp_weight}")
                with col_c:
                    st.metric("🛠 Skills Score", f"{resume.get('Skills Score', 'N/A')} / {skills_weight}")
                with col_d:
                    st.metric("🔍 Keyword Score", f"{resume.get('Keyword Score', 'N/A')} / {keyword_weight}")

                # Fit summary
                st.markdown("### 📝 Fit Summary")
                st.write(resume.get('Final Thoughts', 'N/A'))

                # ATS Report
                if resume.get("ATS Report"):
                    st.markdown("### 📋 ATS Evaluation Report")
                    st.markdown(resume["ATS Report"], unsafe_allow_html=True)

                # ATS Chart
                st.markdown("### 📊 ATS Score Breakdown Chart")
                ats_df = pd.DataFrame({
                    'Component': ['Education', 'Experience', 'Skills', 'Language', 'Keywords'],
                    'Score': [
                        resume.get("Education Score", 0),
                        resume.get("Experience Score", 0),
                        resume.get("Skills Score", 0),
                        resume.get("Language Score", 0),
                        resume.get("Keyword Score", 0)
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

                # 🔷 Detailed ATS Analysis Cards
                st.markdown("### 🔍 Detailed ATS Section Analyses")
                for section_title, key in [
                    ("🏫 Education Analysis", "Education Analysis"),
                    ("💼 Experience Analysis", "Experience Analysis"),
                    ("🛠 Skills Analysis", "Skills Analysis"),
                    ("🗣 Language Quality Analysis", "Language Analysis"),
                    ("🔑 Keyword Analysis", "Keyword Analysis"),
                    ("✅ Final Thoughts", "Final Thoughts")
                ]:
                    analysis_content = resume.get(key, "N/A")
                    if "**Score:**" in analysis_content:
                        parts = analysis_content.split("**Score:**")
                        rest = parts[1].split("**", 1)
                        score_text = rest[0].strip()
                        remaining = rest[1].strip() if len(rest) > 1 else ""
                        formatted_score = f"<div style='background:#4c1d95;color:white;padding:8px;border-radius:6px;margin-bottom:5px;'><b>Score:</b> {score_text}</div>"
                        analysis_html = formatted_score + f"<p>{remaining}</p>"
                    else:
                        analysis_html = f"<p>{analysis_content}</p>"

                    st.markdown(f"""
<div style="background:#5b3cc4; color:white; padding:10px; border-radius:6px;">
  <h3>{section_title}</h3>
</div>
<div style="background:#2d2d3a; color:white; padding:10px; border-radius:6px;">
{analysis_html}
</div>
""", unsafe_allow_html=True)


                # ✅ Display Missing Skills and Keywords as badges
                # ✅ Display Missing Skills as multiline bullet points
                

                # ✅ Show Missing Keywords and Missing Skills
                     # Missing keywords
                


                st.divider()

                detail_tab1, detail_tab2 = st.tabs(["🔎 Bias Analysis", "✅ Rewritten Resume"])

                with detail_tab1:
                    st.markdown("#### Bias-Highlighted Original Text")
                    st.markdown(resume["Highlighted Text"], unsafe_allow_html=True)

                    st.markdown("### 📌 Gender-Coded Word Counts:")
                    bias_col1, bias_col2 = st.columns(2)

                    with bias_col1:
                        st.metric("🔵 Masculine Words", len(resume["Detected Masculine Words"]))
                        if resume["Detected Masculine Words"]:
                            st.markdown("### 📚 Detected Masculine Words with Context:")
                            for item in resume["Detected Masculine Words"]:
                                word = item['word']
                                sentence = item['sentence']
                                st.write(f"🔵 **{word}**: {sentence}", unsafe_allow_html=True)
                        else:
                            st.info("No masculine words detected.")

                    with bias_col2:
                        st.metric("🔴 Feminine Words", len(resume["Detected Feminine Words"]))
                        if resume["Detected Feminine Words"]:
                            st.markdown("### 📚 Detected Feminine Words with Context:")
                            for item in resume["Detected Feminine Words"]:
                                word = item['word']
                                sentence = item['sentence']
                                st.write(f"🔴 **{word}**: {sentence}", unsafe_allow_html=True)
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
                    

                    

                    st.download_button(
                        label="📥 Download Full Analysis Report (.html)",
                        data=html_report,
                        file_name=f"{resume['Resume Name'].split('.')[0]}_report.html",
                        mime="text/html",
                        use_container_width=True,
                        key=f"download_html_{resume['Resume Name']}"
                    )
                    pdf_file = html_to_pdf_bytes(html_report)
                    st.download_button(
                    label="📄 Download Full Analysis Report (.pdf)",
                    data=pdf_file,
                    file_name=f"{resume['Resume Name'].split('.')[0]}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"download_pdf_{resume['Resume Name']}"
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
        # Keys for each project field
             title_key = f"proj_title_{idx}"
             tech_key = f"proj_tech_{idx}"
             duration_key = f"proj_duration_{idx}"
             desc_key = f"proj_desc_{idx}"

        # Initialize only once if key doesn't exist
        if title_key not in st.session_state:
            st.session_state[title_key] = proj["title"]
        if tech_key not in st.session_state:
            st.session_state[tech_key] = proj["tech"]
        if duration_key not in st.session_state:
            st.session_state[duration_key] = proj["duration"]
        if desc_key not in st.session_state:
            st.session_state[desc_key] = proj["description"]

        # Inputs (no value=..., only key)
        st.text_input("Project Title", key=title_key)
        st.text_input("Tech Stack", key=tech_key)
        st.text_input("Duration", key=duration_key)
        st.text_area("Description", key=desc_key)

        # Sync back to project_entries
        proj["title"] = st.session_state[title_key]
        proj["tech"] = st.session_state[tech_key]
        proj["duration"] = st.session_state[duration_key]
        proj["description"] = st.session_state[desc_key]



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
            st.markdown(f"<p style='font-size:17px;'>{summary_text}</p>", unsafe_allow_html=True)


            st.markdown("<h4 style='color:#336699;'>Experience</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for exp in st.session_state.experience_entries:
             if exp["company"] or exp["title"]:
              st.markdown(f"""
            <div style='margin-bottom:15px; padding:10px; border-radius:8px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <b>🏢 {exp['company']}</b><span style='color:gray;'>📆  {exp['duration']}</span>
                </div>
                <div style='font-size:14px;'>💼 <i>{exp['title']}</i></div>
                <div style='font-size:17px;'>📝 {exp['description']}</div>
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
        <span style='font-size:17px;'>📝 <strong>Description:</strong> {proj['description']}</span>
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
import re

with tab2:
    st.markdown("## ✨ <span style='color:#336699;'>Enhanced AI Resume Preview</span>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #bbb;'>", unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.2, 1])

    with col1:
        if st.button("🔁 Clear Preview"):
            st.session_state.pop("ai_output", None)

    with col2:
        if st.button("🚀 Generate AI Resume Preview"):
            # Normalize and ensure at least 2 experience entries
            experience_entries = st.session_state.get('experience_entries', [])
            normalized_experience_entries = []
            for entry in experience_entries:
                if isinstance(entry, dict):
                    title = entry.get("title", "")
                    desc = entry.get("description", "")
                    formatted = f"{title}\n{desc}".strip()
                else:
                    formatted = entry.strip()
                normalized_experience_entries.append(formatted)
            while len(normalized_experience_entries) < 2:
                normalized_experience_entries.append("Placeholder Experience")

            # Normalize and ensure at least 2 project entries
            project_entries = st.session_state.get('project_entries', [])
            normalized_project_entries = []
            for entry in project_entries:
                if isinstance(entry, dict):
                    title = entry.get("title", "")
                    desc = entry.get("description", "")
                    formatted = f"{title}\n{desc}".strip()
                else:
                    formatted = entry.strip()
                normalized_project_entries.append(formatted)
            while len(normalized_project_entries) < 2:
                normalized_project_entries.append("Placeholder Project")

            enhance_prompt = f"""
            You are a professional resume builder AI.

            Enhance the following resume sections based on the user's job title: "{st.session_state['job_title']}". The enhancements must be aligned **strictly and specifically** with the responsibilities, tools, skills, and certifications relevant to that job title.
            
            Instructions:
            1. Rewrite the summary to sound professional, achievement-driven, and role-specific using strong action verbs.
            2. Expand experience and project descriptions into structured bullet points (• or A., B., C.). Highlight domain-specific responsibilities and achievements.
            3. Maintain paragraph structure and meaningful line breaks.
            4. Infer and recommend **only domain-accurate** items, even if not explicitly provided:
               - 6–8 modern **technical Skills** (relevant to the job title; e.g., for Cyber Security: SIEM, Kali Linux, Wireshark, Burp Suite, Splunk, Nmap, Firewalls, OWASP Top 10, etc.)
               - 6–8 strong **Soft Skills**
               - 3–6 job-aligned **Interests** (e.g., bug bounty, ethical hacking, network defense)
               - Only **spoken Languages**
               - 3–6 globally recognized **Certificates** (e.g., CompTIA Security+, CEH, IBM Cybersecurity Analyst, Google Cybersecurity, Cisco CCNA Security)

            Important:
            - Do not include irrelevant frontend/backend tools if the job title is from a different domain like Cyber Security, DevOps, Data Science, etc.
            - The certificate names must match real-world course titles from platforms like Coursera, Udemy, Google, IBM, Cisco, Microsoft, etc.

            📌 Format the output exactly like this:

            Summary:
            • ...

            Experience:
            A. Company Name (Duration)
               • Role
               • Responsibility 1
               • Responsibility 2

            Projects:
            A. <Project Title>
               • Tech Stack: <Job-relevant tools only>
               • Duration: <Start – End>
               • Description:
                 - Clearly describe a specific feature, functionality, or implementation aligned with the job role.
                 - Mention a tool or technology used and explain its context in the project.
                 - Highlight performance improvements, solved challenges, or measurable impacts.
                 - Include a technical or collaborative achievement that enhanced project success.
                 - (Optional) Add an additional impactful point if it meaningfully supports the role.

            Skills:
            Kali Linux, Splunk, SIEM, ...

            SoftSkills:
            Problem Solving, Critical Thinking...

            Languages:
            English, Hindi...

            Interests:
            Capture The Flag (CTF), Ethical Hacking...

            Certificates:
            Google Cybersecurity – Coursera (6 months)
            IBM Cybersecurity Analyst – IBM (Professional Certificate)
            CompTIA Security+ – CompTIA (5 months)

            Use ONLY the user's inputs below as a reference. Rewrite and improve them meaningfully and accurately.

            Summary:
            {st.session_state['summary']}

            Experience:
            {normalized_experience_entries}

            Projects:
            {normalized_project_entries}

            Skills:
            {st.session_state['skills']}

            SoftSkills:
            {st.session_state['Softskills']}

            Languages:
            {st.session_state['languages']}

            Interests:
            {st.session_state['interests']}

            Certificates:
            {[cert['name'] for cert in st.session_state['certificate_links'] if cert['name']]}
            """

            with st.spinner("🧠 Thinking..."):
                ai_output = call_llm(enhance_prompt, session=st.session_state)
                st.session_state["ai_output"] = ai_output



    # ------------------------- PARSE + RENDER -------------------------
    if "ai_output" in st.session_state:
        ai_output = st.session_state["ai_output"]

        def extract_section(label, output, default=""):
            match = re.search(rf"{label}:\s*(.*?)(?=\n\w+:|\Z)", output, re.DOTALL)
            return match.group(1).strip() if match else default

        summary_enhanced = extract_section("Summary", ai_output, st.session_state['summary'])
        experience_raw = extract_section("Experience", ai_output)
        experience_blocks = re.split(r"\n(?=[A-Z]\. )", experience_raw.strip())
        projects_raw = extract_section("Projects", ai_output)
        projects_blocks = re.split(r"\n(?=[A-Z]\. )", projects_raw.strip())
        skills_list = extract_section("Skills", ai_output, st.session_state['skills'])
        softskills_list = extract_section("SoftSkills", ai_output, st.session_state['Softskills'])
        languages_list = extract_section("Languages", ai_output, st.session_state['languages'])
        interests_list = extract_section("Interests", ai_output, st.session_state['interests'])
        certificates_list = extract_section("Certificates", ai_output)

        # ------------------------- UI RENDER -------------------------
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

            def render_bullet_section(title, items):
                st.markdown(f"<h4 style='color:#336699;'>{title}</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
                for item in [i.strip() for i in items.split(",") if i.strip()]:
                    st.markdown(f"<div style='margin-left:10px;'>• {item}</div>", unsafe_allow_html=True)

            render_bullet_section("Skills", skills_list)
            render_bullet_section("Languages", languages_list)
            render_bullet_section("Interests", interests_list)
            render_bullet_section("Soft Skills", softskills_list)

        with right:
            formatted_summary = summary_enhanced.replace('\n• ', '<br>• ').replace('\n', '<br>')
            st.markdown("<h4 style='color:#336699;'>Summary</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:17px;'>{formatted_summary}</p>", unsafe_allow_html=True)

            # Experience
            if experience_blocks:
                st.markdown("<h4 style='color:#336699;'>Experience</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
                experience_titles = [entry.get("title", "").strip().upper() for entry in st.session_state.experience_entries]
                for idx, exp_block in enumerate(experience_blocks):
                    lines = exp_block.strip().split("\n")
                    if not lines:
                        continue
                    heading = lines[0]
                    description_lines = lines[1:]
                    match = re.match(r"[A-Z]\.\s*(.+?)\s*\((.*?)\)", heading)
                    company, duration = (match.group(1).strip(), match.group(2).strip()) if match else (heading, "")
                    role = experience_titles[idx] if idx < len(experience_titles) else ""
                    formatted_exp = "<br>".join(description_lines)

                    st.markdown(f"""
                    <div style='margin-bottom:15px; padding:10px; border-radius:8px;'>
                        <div style='display:flex; justify-content:space-between;'>
                            <b>🏢 {company.upper()}</b><span style='color:gray;'>📆 {duration}</span>
                        </div>
                        <div style='font-size:14px;'>💼 <i>{role}</i></div>
                        <div style='font-size:17px;'>📝 {formatted_exp}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Education
            st.markdown("<h4 style='color:#336699;'>🎓 Education</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
            for edu in st.session_state.education_entries:
                st.markdown(f"""
                <div style='margin-bottom:15px; padding:10px 15px; border-radius:8px;'>
                    <div style='display: flex; justify-content: space-between; font-size: 16px; font-weight: bold;'>
                        <span>🏫 {edu['institution']}</span>
                        <span style='color: gray;'>📅 {edu['year']}</span>
                    </div>
                    <div style='font-size: 14px;'>🎓 <i>{edu['degree']}</i></div>
                    <div style='font-size: 14px;'>📄 {edu['details']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Projects
            if projects_blocks:
                st.markdown("<h4 style='color:#336699;'>Projects</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
                for idx, proj_block in enumerate(projects_blocks):
                    proj = st.session_state.project_entries[idx] if idx < len(st.session_state.project_entries) else {}
                    title = proj.get("title", "")
                    tech = proj.get("tech", "")
                    duration = proj.get("duration", "")
                    description = proj_block
                    for keyword in [title, f"Tech Stack: {tech}", f"Duration: {duration}"]:
                        if keyword and keyword in description:
                            description = description.replace(keyword, "")
                    formatted_proj = description.strip().replace('\n• ', '<br>• ').replace('\n', '<br>')
                    label = chr(65 + idx)

                    st.markdown(f"""
                    <div style='margin-bottom:15px; padding: 10px;'>
                        <strong style='font-size:16px;'>📌 <span style='color:#444;'>{label}. </span>{title}</strong><br>
                        <span style='font-size:14px;'>🛠️ <strong>Tech Stack:</strong> {tech}</span><br>
                        <span style='font-size:14px;'>⏳ <strong>Duration:</strong> {duration}</span><br>
                        <span style='font-size:17px;'>📄 <strong>Description:</strong></span><br>
                        <div style='margin-top:4px; font-size:15px;'>{formatted_proj}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Certificates
            if certificates_list:
                st.markdown("<h4 style='color:#336699;'>📜 Certificates</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
                certs = re.split(r"\n|(?<=\))(?=\s*[A-Z])|(?<=[a-z]\))(?= [A-Z])", certificates_list)
                for cert in [c.strip() for c in certs if c.strip()]:
                    st.markdown(f"<div style='margin-left:10px;'>• {cert}</div>", unsafe_allow_html=True)

            if st.session_state.project_links:
                st.markdown("<h4 style='color:#336699;'>Project Links</h4><hr style='margin-top:-10px;'>", unsafe_allow_html=True)
                for i, link in enumerate(st.session_state.project_links):
                    st.markdown(f"[🔗 Project {i+1}]({link})", unsafe_allow_html=True)


                  
        
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

from io import BytesIO

# Convert HTML to bytes for download
html_bytes = html_content.encode("utf-8")
html_file = BytesIO(html_bytes)

# Convert HTML to PDF using XHTML2PDF-compatible function
pdf_resume_bytes = html_to_pdf_bytes(html_content)

with tab2:
    # ==========================
    # Resume Download Header (Clean & Professional)
    # ==========================
    st.markdown(
        """
        <div style='text-align: center; margin-top: 20px; margin-bottom: 30px;'>
            <h2 style='color: #2f4f6f; font-family: Arial, sans-serif; font-size: 24px;'>
                📥 Download Your Resume
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    # HTML Resume Download Button
    with col1:
        st.download_button(
            label="⬇️ Download as HTML",
            data=html_file,
            file_name=f"{st.session_state['name'].replace(' ', '_')}_Resume.html",
            mime="text/html",
            key="download_resume_html"
        )

    # PDF Resume Download Button
    with col2:
        st.download_button(
            label="⬇️ Download as PDF",
            data=pdf_resume_bytes,
            file_name=f"{st.session_state['name'].replace(' ', '_')}_Resume.pdf",
            mime="application/pdf",
            key="download_resume_pdf"
        )

    # ==========================
    # 📩 Cover Letter Expander
    # ==========================
    with st.expander("📩 Generate Cover Letter from This Resume"):
        generate_cover_letter_from_resume_builder()

    # ==========================
    # ✉️ Generated Cover Letter Preview & Downloads
    # ==========================
    if "cover_letter" in st.session_state:
        st.markdown("""
        <h3 style="color: #003366; margin-top: 30px;">✉️ Generated Cover Letter</h3>
        """, unsafe_allow_html=True)

        styled_cover_letter = st.session_state.get("cover_letter_html", "")
        st.markdown(styled_cover_letter, unsafe_allow_html=True)

        # ✅ Generate PDF from styled HTML
        pdf_file = html_to_pdf_bytes(styled_cover_letter)

        # ✅ Create DOCX function
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

        # ==========================
        # 📥 Cover Letter Download Buttons
        # ==========================
        st.markdown("""
        <div style="margin-top: 20px; margin-bottom: 10px;">
            <strong>⬇️ Download Your Cover Letter:</strong>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 , col3= st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download Cover Letter (.docx)",
                data=create_docx(st.session_state["cover_letter"]),
                file_name=f"{st.session_state['name'].replace(' ', '_')}_Cover_Letter.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_coverletter_docx"
            )
        with col2:
            st.download_button(
                label="📥 Download Cover Letter (PDF)",
                data=pdf_file,
                file_name=f"{st.session_state['name'].replace(' ', '_')}_Cover_Letter.pdf",
                mime="application/pdf",
                key="download_coverletter_pdf"
            )
        with col3:
            st.download_button(
                label="📥 Download Cover Letter (HTML)",
                data=st.session_state["cover_letter_html"],
                file_name=f"{st.session_state['name'].replace(' ', '_')}_Cover_Letter.html",
                mime="text/html",
                key="download_coverletter_html"
            )

    st.markdown("""
    ✅ After downloading your HTML resume, you can [click here to convert it to PDF](https://www.sejda.com/html-to-pdf) using Sejda's free online tool.
    """)

# Convert HTML resume to PDF bytes



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

    def load_domain_distribution():
        df = get_domain_distribution()
        if not df.empty:
            total = df["count"].sum()
            df["percent"] = (df["count"] / total) * 100
            df = df.sort_values(by="count", ascending=True).reset_index(drop=True)
        return df

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

    st.markdown("### 📊 <span style='color:#336699;'>Top Domains by ATS Score</span>", unsafe_allow_html=True)

    top_domains = get_top_domains_by_score()

    if top_domains:
        df_top = pd.DataFrame(top_domains, columns=["domain", "avg_ats", "count"])

        col1, col2 = st.columns([1, 2])
        with col1:
            sort_order = st.radio("Sort by ATS", ["⬆️ High to Low", "⬇️ Low to High"], horizontal=True)
        with col2:
            limit = st.slider("Show Top N Domains", 1, len(df_top), value=min(5, len(df_top)))

        ascending = sort_order == "⬇️ Low to High"
        df_sorted = df_top.sort_values(by="avg_ats", ascending=ascending).head(limit).reset_index(drop=True)

        for i, row in df_sorted.iterrows():
            st.markdown(f"""
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                <h5 style="margin: 0; color: #336699;">📁 {row['domain']}</h5>
                <p style="margin: 5px 0;"><b>🧠 Average ATS:</b> <span style="color:#007acc;">{row['avg_ats']:.2f}</span></p>
                <p style="margin: 5px 0;"><b>📄 Total Resumes:</b> {row['count']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No domain data available.")

    st.markdown("### 📊 <span style='color:#336699;'>Domain Distribution by Count</span>", unsafe_allow_html=True)

    df_domain_dist = load_domain_distribution()

    if not df_domain_dist.empty:
        chart_type = st.radio("📊 Select View Type:", ["📈 Percentage View", "📉 Count View"], horizontal=True)
        orientation = st.radio("📐 Chart Orientation:", ["Horizontal", "Vertical"], horizontal=True)
        bar_color = st.color_picker("🎨 Pick Bar Color", "#66c2a5")

        if chart_type == "📈 Percentage View":
            fig, ax = plt.subplots(figsize=(8, 5))
            if orientation == "Horizontal":
                bars = ax.barh(df_domain_dist["domain"], df_domain_dist["percent"], color=bar_color)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                            f"{width:.1f}%", va='center', fontsize=9, fontweight='bold')
                ax.set_xlabel("Percentage (%)", fontsize=11)
                ax.set_xlim(0, df_domain_dist["percent"].max() * 1.25)
                ax.grid(axis='x', linestyle='--', alpha=0.3)
            else:
                bars = ax.bar(df_domain_dist["domain"], df_domain_dist["percent"], color=bar_color)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f"{height:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax.set_ylabel("Percentage (%)", fontsize=11)
                ax.set_ylim(0, df_domain_dist["percent"].max() * 1.25)
                ax.set_xticklabels(df_domain_dist["domain"], rotation=45, ha="right", fontsize=9)
                ax.grid(axis='y', linestyle='--', alpha=0.3)

            ax.set_title("Resume Distribution by Domain (%)", fontsize=13, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig)

        elif chart_type == "📉 Count View":
            fig, ax = plt.subplots(figsize=(8, 5))
            if orientation == "Horizontal":
                bars = ax.barh(df_domain_dist["domain"], df_domain_dist["count"], color=bar_color)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{int(width)}", va='center', fontsize=9, fontweight='bold')
                ax.set_xlabel("Resume Count", fontsize=11)
                ax.set_xlim(0, df_domain_dist["count"].max() * 1.25)
                ax.grid(axis='x', linestyle='--', alpha=0.3)
            else:
                bars = ax.bar(df_domain_dist["domain"], df_domain_dist["count"], color=bar_color)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f"{int(height)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax.set_ylabel("Resume Count", fontsize=11)
                ax.set_ylim(0, df_domain_dist["count"].max() * 1.25)
                ax.set_xticklabels(df_domain_dist["domain"], rotation=45, ha="right", fontsize=9)
                ax.grid(axis='y', linestyle='--', alpha=0.3)

            ax.set_title("Resume Count by Domain", fontsize=13, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig)

    else:
        st.info("ℹ️ No domain data found.")

    st.markdown("### 📊 <span style='color:#336699;'>Average ATS Score by Domain</span>", unsafe_allow_html=True)
    df_bar = get_average_ats_by_domain()
    if not df_bar.empty:
        chart_style = st.radio("Select Chart Style", ["Vertical Bar", "Horizontal Bar"], horizontal=True)
        bar_color = st.color_picker("Pick Bar Color", "#3399ff")
        fig, ax = plt.subplots(figsize=(8, 5))

        if chart_style == "Vertical Bar":
            bars = ax.bar(df_bar["domain"], df_bar["avg_ats_score"], color=bar_color)
            ax.set_ylabel("Avg ATS Score")
            ax.set_title("ATS by Domain", fontsize=13, fontweight="bold")
            ax.set_xticks(np.arange(len(df_bar["domain"])))
            ax.set_xticklabels(df_bar["domain"], rotation=45, ha="right", fontsize=9)
            ax.set_ylim(0, df_bar["avg_ats_score"].max() * 1.25)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}",
                        ha='center', va='bottom', fontsize=8)

        elif chart_style == "Horizontal Bar":
            bars = ax.barh(df_bar["domain"], df_bar["avg_ats_score"], color=bar_color)
            ax.set_xlabel("Avg ATS Score")
            ax.set_title("ATS by Domain", fontsize=13, fontweight="bold")
            ax.set_yticks(np.arange(len(df_bar["domain"])))
            ax.set_yticklabels(df_bar["domain"], fontsize=9)
            ax.set_xlim(0, df_bar["avg_ats_score"].max() * 1.25)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{width:.1f}", ha="left", va="center", fontsize=8)

        ax.grid(axis='y' if chart_style == "Vertical Bar" else 'x', linestyle='--', alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ No ATS domain data to display.")

    st.markdown("### 📈 Resume Upload Timeline (Daily + 7-Day Trend)")
    df_timeline = get_resume_count_by_day()

    if not df_timeline.empty:
        df_timeline = df_timeline.sort_values("day")
        df_timeline["7_day_avg"] = df_timeline["count"].rolling(window=7, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_timeline["day"], df_timeline["count"], label="Daily Uploads", color="green", marker='o', linewidth=1)
        for x, y in zip(df_timeline["day"], df_timeline["count"]):
            ax.text(x, y + 0.5, str(int(y)), ha="center", va="bottom", fontsize=8)
        ax.plot(df_timeline["day"], df_timeline["7_day_avg"], label="7-Day Avg", color="red", linestyle='--', linewidth=2)
        ax.set_title("📈 Resume Upload Timeline (Daily + 7-Day Trend)", fontsize=13, weight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Upload Count")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ No upload trend data to display.")

    st.markdown("### 🧠 <span style='color:#336699;'>Fair vs Biased Resumes</span>", unsafe_allow_html=True)

    bias_threshold_pie = st.slider("Select Bias Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    df_bias = get_bias_distribution(threshold=bias_threshold_pie)

    if not df_bias.empty and "bias_category" in df_bias.columns:
        fig4, ax4 = plt.subplots()
        ax4.pie(
            df_bias["count"],
            labels=df_bias["bias_category"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#ff6666", "#00cc66"],
            wedgeprops={"edgecolor": "black"}
        )
        ax4.axis("equal")
        st.pyplot(fig4)

        with st.expander("📋 View Bias Distribution Table"):
            st.dataframe(df_bias, use_container_width=True)
    else:
        st.info("📭 No bias data available for the selected threshold.")

    st.markdown("### 🚩 Flagged Candidates (Bias Score > Threshold)")

    bias_threshold = st.slider(
        "Set Bias Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Only candidates with bias score above this value will be shown."
    )

    flagged_df = get_all_candidates(bias_threshold=bias_threshold)

    if not flagged_df.empty:
        st.markdown(f"Showing candidates with bias score > **{bias_threshold}**")
        st.dataframe(
            flagged_df[["id", "resume_name", "candidate_name", "bias_score", "ats_score", "domain", "timestamp"]]
            .sort_values(by="bias_score", key=lambda x: pd.to_numeric(x, errors="coerce"), ascending=False),
            use_container_width=True
        )
    else:
        st.success("✅ No flagged candidates found above the selected threshold.")


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
