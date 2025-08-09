import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime
import pytz
import re
import smtplib
import random
from email.mime.text import MIMEText

DB_NAME = "resume_data.db"
OTP_STORE = {}  # Temporary store for password reset OTPs {email: otp}

# ------------------ Utility: Get IST Time ------------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

# Show IST Time in UI
st.write("🕒 Current IST Time:", get_ist_time().strftime("%Y-%m-%d %H:%M:%S"))

# ------------------ Password Strength Validator ------------------
def is_strong_password(password):
    return (
        len(password) >= 8 and
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'[0-9]', password) and
        re.search(r'[!@#$%^&*(),.?\":{}|<>]', password)
    )

# ------------------ Check if Username Already Exists ------------------
def username_exists(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

# ------------------ Create Tables ------------------
def create_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            groq_api_key TEXT
        )
    ''')
    try:
        c.execute('ALTER TABLE users ADD COLUMN email TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN groq_api_key TEXT')
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

# ------------------ Add User ------------------
def add_user(username, password):
    if not is_strong_password(password):
        return False, "⚠️ Password must be at least 8 characters long and include uppercase, lowercase, number, and special character."
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                  (username, hashed_password.decode('utf-8')))
        conn.commit()
        return True, "✅ Registered! You can now login."
    except sqlite3.IntegrityError:
        return False, "🚫 Username already exists."
    finally:
        conn.close()

# ------------------ Verify User & Load Saved API Key ------------------
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

# ------------------ Save or Update User's Groq API Key ------------------
def save_user_api_key(username, api_key):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET groq_api_key = ? WHERE username = ?", (api_key, username))
    conn.commit()
    conn.close()
    st.session_state.user_groq_key = api_key

# ------------------ Get User's Saved API Key ------------------
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

# ------------------ Get Total Registered Users ------------------
def get_total_registered_users():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    count = c.fetchone()[0]
    conn.close()
    return count

# ------------------ Get Today's Logins (based on IST) ------------------
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

# ------------------ Get All User Logs ------------------
def get_all_user_logs():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, action, timestamp FROM user_logs ORDER BY timestamp DESC")
    logs = c.fetchall()
    conn.close()
    return logs

# ------------------ Email Sending for OTP ------------------
def send_email_otp(to_email, otp):
    sender_email = st.secrets["EMAIL_USER"]
    sender_pass = st.secrets["EMAIL_PASS"]
    msg = MIMEText(f"Your OTP for password reset is: {otp}")
    msg["Subject"] = "Password Reset OTP"
    msg["From"] = sender_email
    msg["To"] = to_email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.sendmail(sender_email, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# ------------------ Forgot Password Flow ------------------
def forgot_password_flow():
    st.subheader("🔑 Forgot Password")
    step = st.session_state.get("forgot_step", "email")

    if step == "email":
        email = st.text_input("Enter your registered email")
        if st.button("Send OTP"):
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT username FROM users WHERE email = ?", (email,))
            user = c.fetchone()
            conn.close()
            if user:
                otp = str(random.randint(100000, 999999))
                OTP_STORE[email] = otp
                if send_email_otp(email, otp):
                    st.session_state.forgot_step = "otp"
                    st.session_state.forgot_email = email
                    st.success("✅ OTP sent to your email!")
            else:
                st.error("❌ Email not found.")

    elif step == "otp":
        otp_input = st.text_input("Enter OTP")
        if st.button("Verify OTP"):
            if otp_input == OTP_STORE.get(st.session_state.forgot_email):
                st.session_state.forgot_step = "reset"
                st.success("✅ OTP Verified! You can now reset your password.")
            else:
                st.error("❌ Incorrect OTP.")

    elif step == "reset":
        new_password = st.text_input("Enter new password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        if st.button("Reset Password"):
            if new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif not is_strong_password(new_password):
                st.error("⚠️ Weak password. Must be 8+ chars, with uppercase, lowercase, number & symbol.")
            else:
                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, st.session_state.forgot_email))
                conn.commit()
                conn.close()
                st.success("✅ Password reset successful!")
                st.session_state.forgot_step = "email"
