import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime
import pytz
import re
import smtplib
from email.mime.text import MIMEText
import random

DB_NAME = "resume_data.db"
OTP_STORE = {}  # Temporary store for OTPs {email: otp}

# ------------------ Utility: Get IST Time ------------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

st.write("🕒 Current IST Time:", get_ist_time().strftime("%Y-%m-%d %H:%M:%S"))

# ------------------ Email Sender ------------------
def send_otp_email(to_email):
    otp = str(random.randint(100000, 999999))
    OTP_STORE[to_email] = otp

    from_email = st.secrets["EMAIL_USER"]
    password = st.secrets["EMAIL_PASS"]

    msg = MIMEText(f"Your password reset OTP is: {otp}")
    msg["Subject"] = "Password Reset OTP"
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, password)
        server.send_message(msg)

    return otp

# ------------------ Password Strength Validator ------------------
def is_strong_password(password):
    return (
        len(password) >= 8 and
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'[0-9]', password) and
        re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
    )

# ------------------ Create Tables ------------------
def create_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            groq_api_key TEXT
        )
    ''')
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
def add_user(username, password, email):
    if not is_strong_password(password):
        return False, "⚠️ Weak password."

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', 
                  (username, hashed_password.decode('utf-8'), email))
        conn.commit()
        return True, "✅ Registered!"
    except sqlite3.IntegrityError:
        return False, "🚫 Username or Email exists."
    finally:
        conn.close()

# ------------------ Verify User ------------------
def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT password, groq_api_key FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result:
        stored_hashed, stored_key = result
        if bcrypt.checkpw(password.encode('utf-8'), stored_hashed.encode('utf-8')):
            st.session_state.username = username
            st.session_state.user_groq_key = stored_key or ""
            return True, stored_key
    return False, None

# ------------------ Reset Password ------------------
def reset_password(email, new_password):
    if not is_strong_password(new_password):
        return False, "⚠️ Weak password."

    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET password = ? WHERE email = ?", 
              (hashed_password.decode('utf-8'), email))
    conn.commit()
    conn.close()
    return True, "✅ Password reset successful."

# ------------------ Save API Key ------------------
def save_user_api_key(username, api_key):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET groq_api_key = ? WHERE username = ?", (api_key, username))
    conn.commit()
    conn.close()
    st.session_state.user_groq_key = api_key

# ------------------ Logs & Stats ------------------
def log_user_action(username, action):
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO user_logs (username, action, timestamp) VALUES (?, ?, ?)', 
              (username, action, timestamp))
    conn.commit()
    conn.close()

def get_total_registered_users():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    count = c.fetchone()[0]
    conn.close()
    return count

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

def get_all_user_logs():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, action, timestamp FROM user_logs ORDER BY timestamp DESC")
    logs = c.fetchall()
    conn.close()
    return logs
