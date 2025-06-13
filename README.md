# 💼 LEXIBOT – Smart Resume AI 🔍🚀  
**Transform your career with AI-powered resume analysis, inclusive rewriting, job insights, and more.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![LLM Powered](https://img.shields.io/badge/LLM-Groq%20%2B%20LLaMA3-blueviolet.svg)

---

### 🔥 Live Preview (Optional)
[🧪 Try Demo](#) • [🛠 GitHub Repo](#) • [📽 Pitch Video](#)  
> Replace `#` with actual URLs when available.

---

## 🚀 Project Summary

**LEXIBOT** is a powerful, ethically-aware resume analysis tool designed using cutting-edge LLMs. It helps users:

- Analyze resumes against job descriptions (ATS scoring)
- Detect gender-coded bias in language
- Rewrite resumes to be bias-free and professional
- Recommend jobs, courses, and improvements
- Build resumes interactively
- Visualize resume metrics through an interactive dashboard
- Admins can view logs, trends, and user activity

> Ideal for students, job seekers, and hiring teams.

---

## 🧠 Key Features

| Category | Details |
|---------|---------|
| 🧾 **Resume Analyzer** | ATS-style scoring, formatting tips, missing keyword suggestions |
| 🟣 **Bias Detection** | Gender-coded word detection (masculine/feminine) with highlights |
| ✨ **Inclusive Rewriting** | LLM rewrites resume using neutral, professional language |
| 🎯 **Custom Scoring** | Adjustable weights for Education, Experience, Skills, Language, and Keywords |
| 📊 **Visual Dashboard** | Charts for bias score, word usage, ATS breakdown |
| 🧑‍💼 **Resume Builder** | Build eye-catching resumes with photo and key sections |
| 🔍 **Job & Course Finder** | LinkedIn/Naukri/FoundIt job links, and role-based upskilling courses |
| 🔐 **User & Admin Views** | Login, session logs, resume database, admin activity tracker |

---

## 🌐 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python, SQLite
- **AI/LLM**: Groq API + LLaMA 3.3
- **OCR Support**: EasyOCR
- **Vector Search**: FAISS + HuggingFace Embeddings
- **PDF Parsing**: PyMuPDF, pdf2image
- **Charts**: Matplotlib, Streamlit built-ins
- **NLP**: NLTK, Regex-based gender detection

---

## 🖼 Screenshots

### 🔒 Login + Live Stats
![Login Page](docs/login_stats.png)

### 📊 Resume Dashboard
![Dashboard](docs/dashboard.png)

### ⚖️ Bias Detection + Rewriting
![Bias Detection](docs/bias_analysis.png)

> Place these images in a `docs/` folder inside your repo.

---

## 📦 Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/your-username/lexibot.git
cd lexibot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run tia.py

