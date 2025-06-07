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

# ✅ Insert a new candidate entry (timestamp added manually with local time)
def insert_candidate(data: tuple):
    """
    Insert a candidate's evaluation into the database with local timestamp.
    :param data: Tuple of 10 elements (resume_name, candidate_name, ats_score,
                 edu_score, exp_score, skills_score, lang_score,
                 keyword_score, bias_score, domain)
    """
    # Use local timezone (e.g., Asia/Kolkata)
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

# 📄 All candidates (latest first)
def get_all_candidates() -> list:
    cursor.execute("SELECT * FROM candidates ORDER BY timestamp DESC")
    return cursor.fetchall()

# 🗑️ Delete candidate by ID
def delete_candidate_by_id(candidate_id: int):
    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()

# 📤 Export database to CSV
def export_to_csv(filepath: str = "candidates_export.csv"):
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY timestamp DESC", conn)
    df.to_csv(filepath, index=False)
