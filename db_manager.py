import sqlite3
import pandas as pd
from datetime import datetime

# 🔌 Connect to the SQLite database
conn = sqlite3.connect("resume_data.db", check_same_thread=False)
cursor = conn.cursor()

# 🗂️ Create the candidates table (with manual timestamp field)
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

# ✅ Insert a new candidate entry
def insert_candidate(data: tuple):
    """
    Insert a candidate's evaluation into the database with timestamp.

    :param data: Tuple of 11 elements excluding timestamp:
        (resume_name, candidate_name, ats_score, edu_score, exp_score,
         skills_score, lang_score, keyword_score, bias_score, domain)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO candidates (
            resume_name, candidate_name, ats_score, edu_score, exp_score,
            skills_score, lang_score, keyword_score, bias_score, domain, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data + (timestamp,))
    conn.commit()

# 📊 Top domains by ATS
def get_top_domains_by_score(limit: int = 5) -> list:
    """
    Return top domains sorted by average ATS score.

    :param limit: Max number of top domains to return.
    :return: List of tuples (domain, average_score, resume_count)
    """
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
    """
    Get all candidate rows from the database, most recent first.

    :return: List of tuples (each row).
    """
    cursor.execute("SELECT * FROM candidates ORDER BY timestamp DESC")
    return cursor.fetchall()

# 🗑️ Delete candidate by ID
def delete_candidate_by_id(candidate_id: int):
    """
    Delete a specific candidate row.

    :param candidate_id: The primary key (ID) of the row to delete.
    """
    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()

# 📤 Export database to CSV
def export_to_csv(filepath: str = "candidates_export.csv"):
    """
    Export all candidate rows into a CSV file.

    :param filepath: Name of the output file.
    """
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY timestamp DESC", conn)
    df.to_csv(filepath, index=False)
