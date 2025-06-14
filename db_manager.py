import sqlite3
import pandas as pd
from datetime import datetime
import pytz
from collections import defaultdict

# 🔌 Connect to the SQLite database
conn = sqlite3.connect("resume_data.db", check_same_thread=False)
cursor = conn.cursor()

# 🗂️ Create the candidates table
cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_name TEXT,
    candidate_name TEXT,
    ats_score INTEGER,
    edu_score INTEGER,
    exp_score INTEGER,
    skills_score INTEGER,
    lang_score INTEGER,
    keyword_score INTEGER,
    bias_score REAL,
    domain TEXT,
    timestamp DATETIME
)
""")
conn.commit()

# ✅ Insert a new candidate entry with local timestamp
def insert_candidate(data: tuple):
    local_tz = pytz.timezone("Asia/Kolkata")
    local_time = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO candidates (
            resume_name, candidate_name, ats_score, edu_score, exp_score,
            skills_score, lang_score, keyword_score, bias_score, domain, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data + (local_time,))
    conn.commit()

# 📊 Top domains by ATS
def get_top_domains_by_score(limit: int = 5) -> list:
    cursor.execute("""
        SELECT domain, ROUND(AVG(ats_score), 2) AS avg_score, COUNT(*) AS count
        FROM candidates
        GROUP BY domain
        ORDER BY avg_score DESC
        LIMIT ?
    """, (limit,))
    return cursor.fetchall()

# 📈 Resume count by date (for trend analysis)
def get_resume_count_by_day():
    query = """
    SELECT DATE(timestamp) AS day, COUNT(*) AS count
    FROM candidates
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp) DESC
    """
    return pd.read_sql_query(query, conn)

# 📊 Average ATS score by domain
def get_average_ats_by_domain():
    query = """
    SELECT domain, ROUND(AVG(ats_score), 2) AS avg_ats_score
    FROM candidates
    GROUP BY domain
    ORDER BY avg_ats_score DESC
    """
    return pd.read_sql_query(query, conn)

# 🥧 Resume distribution by domain
def get_domain_distribution():
    query = """
    SELECT domain, COUNT(*) as count
    FROM candidates
    GROUP BY domain
    """
    return pd.read_sql_query(query, conn)

# 📅 Filter candidates by date
def filter_candidates_by_date(start: str, end: str):
    query = """
    SELECT * FROM candidates
    WHERE DATE(timestamp) BETWEEN DATE(?) AND DATE(?)
    ORDER BY timestamp DESC
    """
    return pd.read_sql_query(query, conn, params=(start, end))

# 🗑️ Delete candidate by ID
def delete_candidate_by_id(candidate_id: int):
    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()

# 📄 Get all candidates with optional bias and ATS filters
def get_all_candidates(bias_threshold: float = None, min_ats: int = None):
    query = "SELECT * FROM candidates WHERE 1=1"
    params = []

    if bias_threshold is not None:
        query += " AND bias_score >= ?"
        params.append(bias_threshold)

    if min_ats is not None:
        query += " AND ats_score >= ?"
        params.append(min_ats)

    query += " ORDER BY timestamp DESC"
    return pd.read_sql_query(query, conn, params=params)

# 📤 Export all candidate data to CSV
def export_to_csv(filepath: str = "candidates_export.csv"):
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY timestamp DESC", conn)
    df.to_csv(filepath, index=False)

# 🔎 Get a specific candidate by ID (for safe delete preview)
def get_candidate_by_id(candidate_id: int):
    query = "SELECT * FROM candidates WHERE id = ?"
    return pd.read_sql_query(query, conn, params=(candidate_id,))

# 🧠 Get bias score distribution (Fair vs Biased resumes)
def get_bias_distribution(threshold: float = 0.6):
    query = f"""
    SELECT 
        CASE WHEN bias_score >= {threshold} THEN 'Biased' ELSE 'Fair' END AS bias_category,
        COUNT(*) AS count
    FROM candidates
    GROUP BY bias_category
    """
    return pd.read_sql_query(query, conn)

# 📈 ATS score trend over time
def get_daily_ats_stats():
    query = """
    SELECT DATE(timestamp) AS date, ROUND(AVG(ats_score), 2) AS avg_ats
    FROM candidates
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp)
    """
    return pd.read_sql_query(query, conn)

# 🧠 Domain Detection from Job Title + Description
def detect_domain_from_title_and_description(job_title, job_description):
    title = job_title.lower().strip()
    jd = job_description.lower().strip()
    combined = f"{title} {jd}"

    domain_scores = defaultdict(int)

    WEIGHTS = {
        "Data Science": 3,
        "AI / Machine Learning": 3,
        "UI/UX Design": 3,
        "Mobile Development": 3,
        "Frontend Development": 2,
        "Backend Development": 2,
        "Full Stack Development": 4,
        "Cybersecurity": 4,
        "Cloud Engineering": 3,
        "DevOps / Infrastructure": 3,
        "General Software Engineering": 1,
    }

    keywords = {
        "Data Science": [
            "data analyst", "data scientist", "data science", "eda", "pandas", "numpy",
            "data analysis", "statistics", "data visualization", "matplotlib", "seaborn",
            "power bi", "tableau", "looker", "kpi", "sql", "excel", "dashboards",
            "insights", "hypothesis testing", "a/b testing", "business intelligence", "data wrangling"
        ],
        "AI / Machine Learning": [
            "machine learning", "ml engineer", "deep learning", "neural network",
            "nlp", "computer vision", "ai engineer", "scikit-learn", "tensorflow", "pytorch",
            "llm", "huggingface", "xgboost", "lightgbm", "classification", "regression",
            "reinforcement learning", "transfer learning", "model training", "bert", "gpt"
        ],
        "UI/UX Design": [
            "ui", "ux", "figma", "designer", "user interface", "user experience",
            "adobe xd", "sketch", "wireframe", "prototyping", "interaction design",
            "user research", "usability", "design system", "visual design", "accessibility",
            "human-centered design", "affinity diagram"
        ],
        "Mobile Development": [
            "android", "ios", "flutter", "kotlin", "swift", "mobile app", "react native",
            "mobile application", "play store", "app store", "firebase", "mobile sdk",
            "xcode", "android studio", "cross-platform", "native mobile"
        ],
        "Frontend Development": [
            "frontend", "html", "css", "javascript", "react", "angular", "vue",
            "typescript", "next.js", "webpack", "bootstrap", "tailwind", "sass", "es6",
            "responsive design", "web accessibility", "dom", "jquery", "redux"
        ],
        "Backend Development": [
            "backend", "node.js", "django", "flask", "express", "api development",
            "sql", "nosql", "server-side", "mysql", "postgresql", "mongodb", "rest api",
            "graphql", "java", "spring boot", "authentication", "authorization", "mvc",
            "business logic", "orm", "database schema"
        ],
        "Full Stack Development": [
            "full stack", "fullstack", "mern", "mean", "mevn",
            "frontend and backend", "end-to-end development", "full stack developer",
            "api integration", "react + node", "monolith", "microservices", "integrated app"
        ],
        "Cybersecurity": [
            "cybersecurity", "security analyst", "penetration testing", "ethical hacking",
            "owasp", "vulnerability", "threat analysis", "infosec", "red team", "blue team",
            "incident response", "firewall", "ids", "ips", "malware", "encryption",
            "cyber threat", "security operations", "siem", "zero-day", "cyber attack"
        ],
        "Cloud Engineering": [
            "cloud", "aws", "azure", "gcp", "cloud engineer", "cloud computing",
            "cloud infrastructure", "cloud security", "s3", "ec2", "cloud formation",
            "load balancer", "auto scaling", "cloud storage", "cloud native", "cloud migration"
        ],
        "DevOps / Infrastructure": [
            "devops", "docker", "kubernetes", "ci/cd", "jenkins", "ansible",
            "infrastructure as code", "terraform", "monitoring", "prometheus", "grafana",
            "deployment", "automation", "pipeline", "build and release", "scripting",
            "bash", "shell script", "site reliability"
        ],
        "General Software Engineering": [
            "software engineer", "web developer", "developer", "programmer",
            "object oriented", "design patterns", "agile", "scrum", "git", "version control",
            "unit testing", "integration testing", "debugging", "code review", "system design"
        ]
    }

    for domain, kws in keywords.items():
        matches = sum(1 for kw in kws if kw in combined)
        domain_scores[domain] += matches * WEIGHTS[domain]

    frontend_match = any(kw in combined for kw in keywords["Frontend Development"])
    backend_match = any(kw in combined for kw in keywords["Backend Development"])

    if "full stack" in combined or "fullstack" in combined:
        domain_scores["Full Stack Development"] += 2
    elif frontend_match and backend_match:
        domain_scores["Full Stack Development"] += WEIGHTS["Full Stack Development"]

    if domain_scores:
        top_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[top_domain] > 0:
            return top_domain

    return "General"
# 🚩 Get all flagged candidates (bias_score > threshold)
def get_flagged_candidates(threshold: float = 0.6):
    query = """
    SELECT resume_name, candidate_name, ats_score, bias_score, domain, timestamp
    FROM candidates
    WHERE bias_score > ?
    ORDER BY bias_score DESC
    """
    return pd.read_sql_query(query, conn, params=(threshold,))

