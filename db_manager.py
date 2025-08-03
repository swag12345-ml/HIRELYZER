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

# ✅ Normalize domain values
def normalize_domain(domain: str) -> str:
    mapping = {
        "AI / Machine Learning": "AI/ML",
        "Artificial Intelligence": "AI/ML",
        "Machine Learning": "AI/ML",
        "DevOps / Infrastructure": "DevOps & Infrastructure",
        "Cloud Engineering": "Cloud & DevOps",
        "General Software Engineering": "Software Engineering"
    }
    return mapping.get(domain.strip(), domain.strip())

# ✅ Insert a new candidate entry with normalized domain
def insert_candidate(data: tuple):
    # Extract domain and normalize
    domain = data[9]
    normalized_domain = normalize_domain(domain)

    # Replace original domain with normalized version
    updated_data = data[:9] + (data[9],) + (normalize_domain(data[9]),)

    # Add timestamp
    local_tz = pytz.timezone("Asia/Kolkata")
    local_time = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")

    # Final insert
    cursor.execute("""
        INSERT INTO candidates (
            resume_name, candidate_name, ats_score, edu_score, exp_score,
            skills_score, lang_score, keyword_score, bias_score, domain, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data[:9] + (normalized_domain, local_time))
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
from collections import defaultdict

from collections import defaultdict
def get_domain_similarity(resume_domain, job_domain):
    resume_domain = resume_domain.strip().lower()
    job_domain = job_domain.strip().lower()

    normalization = {
        "frontend": "frontend development",
        "backend": "backend development",
        "fullstack": "full stack development",
        "ui/ux": "ui/ux design",
        "ux/ui": "ui/ux design",
        "software developer": "software engineering",
        "mobile developer": "mobile development",
        "android developer": "mobile development",
        "ios developer": "mobile development",
        "ai": "ai/ml",
        "machine learning": "ai/ml",
        "ml": "ai/ml",
        "cloud": "cloud & devops",
        "cloud engineer": "cloud & devops",
        "devops": "devops & infrastructure",
        "cyber security": "cybersecurity",
        "cybersecurity engineer": "cybersecurity",
        "security analyst": "cybersecurity"
    }

    resume_domain = normalization.get(resume_domain, resume_domain)
    job_domain = normalization.get(job_domain, job_domain)

    similarity_map = {
        ("full stack development", "frontend development"): 0.8,
        ("full stack development", "backend development"): 0.8,
        ("full stack development", "ui/ux design"): 0.7,
        ("frontend development", "ui/ux design"): 0.9,
        ("backend development", "frontend development"): 0.6,
        ("data science", "ai/ml"): 0.9,
        ("cloud engineering", "devops & infrastructure"): 0.85,
        ("software engineering", "full stack development"): 0.75,
        ("software engineering", "frontend development"): 0.6,
        ("ai/ml", "data science"): 0.9,
        ("software engineering", "general"): 0.5,
        ("mobile development", "software engineering"): 0.6,
        ("mobile development", "full stack development"): 0.65,
        ("cybersecurity", "software engineering"): 0.5,
        ("cybersecurity", "devops & infrastructure"): 0.65,
        ("cloud & devops", "devops & infrastructure"): 0.8,
        ("cloud & devops", "software engineering"): 0.5
    }

    if resume_domain == job_domain:
        return 1.0

    return similarity_map.get((resume_domain, job_domain)) or \
           similarity_map.get((job_domain, resume_domain)) or 0.3

def detect_domain_from_title_and_description(job_title, job_description):
    title = job_title.lower().strip()
    jd = job_description.lower().strip()
    combined = f"{title} {jd}"

    domain_scores = defaultdict(int)
    domain_match_counts = defaultdict(int)

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
           "full stack", "fullstack", "mern", "mean", "mevn", "lamp", "jamstack",
           "frontend and backend", "end-to-end development", "full stack developer",
            "api integration", "rest api", "graphql", "react + node", "react.js + express",
            "monolith", "microservices", "serverless architecture", "integrated app",
            "web application", "cross-functional development", "component-based architecture",
            "database design", "middleware", "mvc", "mvvm", "authentication", "authorization",
            "session management", "cloud deployment", "responsive ui", "performance tuning",
            "state management", "redux", "context api", "axios", "fetch api",
            "typescript", "es6", "html5", "css3", "javascript", "react", "next.js", "angular",
            "node.js", "express.js", "spring boot", "java", "python", "django", "flask",
            "mysql", "postgresql", "mongodb", "sqlite", "nosql", "sql", "orm", "prisma",
            "docker", "ci/cd", "git", "github", "bitbucket", "testing", "jest", "mocha",
            "unit testing", "integration testing", "agile", "scrum", "devops"
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

    # Step 1: Count matches and assign weighted scores
    for domain, kws in keywords.items():
        match_count = sum(1 for kw in kws if kw in combined)
        domain_match_counts[domain] = match_count
        domain_scores[domain] = match_count * WEIGHTS[domain]

    # Step 2: Handle Full Stack more intelligently
    frontend_hits = domain_match_counts["Frontend Development"]
    backend_hits = domain_match_counts["Backend Development"]
    fullstack_mentioned = "full stack" in combined or "fullstack" in combined

    if fullstack_mentioned:
        domain_scores["Full Stack Development"] += 5  # Boost if explicitly mentioned

    if frontend_hits >= 5 and backend_hits >= 5:
        domain_scores["Full Stack Development"] += 8  # Boost for dual presence

    # Avoid false full stack assignment when one side dominates
    if frontend_hits > backend_hits * 2 and not fullstack_mentioned:
        domain_scores["Full Stack Development"] = 0
    elif backend_hits > frontend_hits * 2 and not fullstack_mentioned:
        domain_scores["Full Stack Development"] = 0

    # Step 3: Return highest scoring normalized domain
    if domain_scores:
        top_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[top_domain] > 0:
            normalization_map = {
                "AI / Machine Learning": "AI/ML",
                "Artificial Intelligence": "AI/ML",
                "Machine Learning": "AI/ML",
                "DevOps / Infrastructure": "DevOps & Infrastructure",
                "Cloud Engineering": "Cloud & DevOps",
                "General Software Engineering": "Software Engineering"
            }
            return normalization_map.get(top_domain, top_domain)

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



