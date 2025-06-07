import sqlite3
import pandas as pd
from datetime import datetime
import pytz  # ✅ For local timezone

# 🔌 Connect to the SQLite database
conn = sqlite3.connect("resume_data.db", check_same_thread=False)
cursor = conn.cursor()

# 🗂️ Create the candidates table (timestamp provided by Python with local time)
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

# 📊 Resume count by date (for trend analysis)
def get_resume_count_by_day():
    query = """
    SELECT DATE(timestamp) AS day, COUNT(*) AS count
    FROM candidates
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp) DESC
    """
    df = pd.read_sql_query(query, conn)
    return df

# 📊 Average ATS score by domain
def get_average_ats_by_domain():
    query = """
    SELECT domain, ROUND(AVG(ats_score), 2) AS avg_ats_score
    FROM candidates
    GROUP BY domain
    ORDER BY avg_ats_score DESC
    """
    df = pd.read_sql_query(query, conn)
    return df

# 🥧 Resume distribution by domain
def get_domain_distribution():
    query = """
    SELECT domain, COUNT(*) as count
    FROM candidates
    GROUP BY domain
    """
    df = pd.read_sql_query(query, conn)
    return df

# 📅 Filter candidates by date
def filter_candidates_by_date(start: str, end: str):
    query = """
    SELECT * FROM candidates
    WHERE DATE(timestamp) BETWEEN DATE(?) AND DATE(?)
    ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn, params=(start, end))
    return df

# 🗑️ Delete candidate by ID
def delete_candidate_by_id(candidate_id: int):
    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()

# 📄 Get all candidates (if needed separately)
def get_all_candidates() -> list:
    cursor.execute("SELECT * FROM candidates ORDER BY timestamp DESC")
    return cursor.fetchall()

# 📤 Export database to CSV
def export_to_csv(filepath: str = "candidates_export.csv"):
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY timestamp DESC", conn)
    df.to_csv(filepath, index=False)
