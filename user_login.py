# user_login.py

import sqlite3
import bcrypt
from datetime import datetime
from datetime import datetime, timedelta

DB_NAME = "resume_data.db"

# ------------------ Create Tables ------------------
def create_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # User logs table
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

# ------------------ Add User ------------------
def add_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                  (username, hashed_password.decode('utf-8')))
        conn.commit()
        conn.close()
        return True  # Successfully registered
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Username already exists



# ------------------ Verify User ------------------
def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()

    if result:
        stored_hashed = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_hashed.encode('utf-8'))
    return False

# ------------------ Log User Action ------------------
def log_user_action(username, action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO user_logs (username, action, timestamp) VALUES (?, ?, ?)', 
              (username, action, timestamp))
    conn.commit()
    conn.close()

# ------------------ Get Active User Count ------------------
def get_total_registered_users():
    conn = sqlite3.connect("resume_data.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    count = c.fetchone()[0]
    conn.close()
    return count

from datetime import datetime, timedelta

def get_recent_active_users(window_minutes=10):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Get unique users with their most recent action
    c.execute("""
        SELECT username, action, MAX(timestamp) 
        FROM user_logs 
        GROUP BY username
    """)
    rows = c.fetchall()
    
    active_users = 0
    now = datetime.now()
    for username, action, ts in rows:
        last_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        if action == 'login' and (now - last_time).total_seconds() <= window_minutes * 60:
            active_users += 1

    conn.close()
    return active_users


def get_all_user_logs():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, action, timestamp FROM user_logs ORDER BY timestamp DESC")
    logs = c.fetchall()
    conn.close()
    return logs
