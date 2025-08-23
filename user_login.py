import sqlite3
import streamlit as st
import requests as req
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from datetime import datetime
import pytz
import os

DB_NAME = "resume_data.db"

# Google OAuth credentials (Move these to Streamlit Secrets in production)
GOOGLE_CLIENT_ID = "89576252093-4bfhnj3of7lh3rck0fodis863i0d4qp5.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
REDIRECT_URI = "https://lexibot-zmxdulncckdrifhe8pad5g.streamlit.app/"

# ------------------ Utility: Get IST Time ------------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

# ------------------ Create Tables ------------------
def create_tables():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
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

# ------------------ Add User If Not Exists ------------------
def add_google_user(email):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (email,))
    if not c.fetchone():
        c.execute("INSERT INTO users (username, email) VALUES (?, ?)", (email, email))
        conn.commit()
    conn.close()

# ------------------ Save or Update Groq API Key ------------------
def save_user_api_key(username, api_key):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET groq_api_key = ? WHERE username = ?", (api_key, username))
    conn.commit()
    conn.close()
    st.session_state.user_groq_key = api_key

def get_user_api_key(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT groq_api_key FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result and result[0] else None

# ------------------ Log User Action ------------------
def log_user_action(username, action):
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO user_logs (username, action, timestamp) VALUES (?, ?, ?)',
              (username, action, timestamp))
    conn.commit()
    conn.close()

# ------------------ Metrics ------------------
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
        WHERE action = 'google_login'
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

# ------------------ Google OAuth Logic ------------------
def google_login_ui():
    AUTH_URL = (
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?response_type=code"
        f"&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=openid%20email%20profile"
    )

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.markdown(f"[👉 Login with Google]({AUTH_URL})")

        if "code" in st.query_params:
            code = st.query_params["code"]
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code"
            }
            response = req.post(token_url, data=data)
            tokens = response.json()

            try:
                idinfo = id_token.verify_oauth2_token(tokens["id_token"], grequests.Request(), GOOGLE_CLIENT_ID)
                email = idinfo.get("email")
                st.session_state.username = email
                st.session_state.logged_in = True
                add_google_user(email)
                log_user_action(email, "google_login")
                st.experimental_rerun()
            except ValueError:
                st.error("Login failed. Please try again.")
    else:
        st.success(f"✅ Welcome, {st.session_state.username}!")
        st.write("Your Email:", st.session_state.username)

        # API Key Section
        user_api_key = get_user_api_key(st.session_state.username) or ""
        api_key_input = st.text_input("Your Groq API Key", value=user_api_key)
        if st.button("Save API Key"):
            save_user_api_key(st.session_state.username, api_key_input)
            st.success("API Key saved!")

        if st.button("Logout"):
            log_user_action(st.session_state.username, "logout")
            st.session_state.logged_in = False
            st.experimental_rerun()

# ------------------ Run App ------------------
create_tables()
st.title("🔐 Google Login System with Tracking")
google_login_ui()

st.write("📊 **Total Registered Users:**", get_total_registered_users())
st.write("📆 **Logins Today:**", get_logins_today())
