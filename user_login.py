import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime
import pytz

DB_NAME = "resume_data.db"

# ------------------ Utility: Get IST Time ------------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

st.write("🕒 Current IST Time:", get_ist_time().strftime("%Y-%m-%d %H:%M:%S"))

# ------------------ Create Tables ------------------
def create_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Create users table with optional name/email
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            name TEXT
        )
    ''')

    # Add missing columns if needed (for upgrades)
    try:
        c.execute('ALTER TABLE users ADD COLUMN email TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN name TEXT')
    except sqlite3.OperationalError:
        pass

    c.execute('''
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

# ------------------ Add Traditional User ------------------
def add_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                  (username, hashed_password.decode('utf-8')))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# ------------------ Verify Traditional User ------------------
def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8'))
    return False

# ------------------ Log User Action ------------------
def log_user_action(username, action):
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO user_logs (username, action, timestamp) VALUES (?, ?, ?)', 
              (username, action, timestamp))
    conn.commit()
    conn.close()

# ------------------ Get Total Users ------------------
def get_total_registered_users():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    count = c.fetchone()[0]
    conn.close()
    return count

# ------------------ Get Today's Logins ------------------
def get_logins_today():
    today = get_ist_time().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*) FROM user_logs
        WHERE action = 'login'
          AND DATE(timestamp) = ?
    """, (today,))
    count = c.fetchone()[0]
    conn.close()
    return count

# ------------------ Get Logs ------------------
def get_all_user_logs():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, action, timestamp FROM user_logs ORDER BY timestamp DESC")
    logs = c.fetchall()
    conn.close()
    return logs

# ------------------ Add Google User ------------------
def add_google_user(email, name="Google User"):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Add user with default password and optional name/email
        c.execute('INSERT OR IGNORE INTO users (username, password, email, name) VALUES (?, ?, ?, ?)', 
                  (email, 'google_oauth', email, name))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

# ------------------ Check if User Exists ------------------
def user_exists(username_or_email):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ? OR email = ?", 
              (username_or_email, username_or_email))
    exists = c.fetchone() is not None
    conn.close()
    return exists
