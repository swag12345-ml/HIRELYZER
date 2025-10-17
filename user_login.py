import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime, timedelta
import pytz
import re
import os
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Persistent storage path for Streamlit Cloud
os.makedirs(".streamlit_storage", exist_ok=True)
DB_NAME = os.path.join(".streamlit_storage", "resume_data.db")

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
        re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
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
        return False, "⚠ Password must be at least 8 characters long and include uppercase, lowercase, number, and special character."

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
            # Store username in session
            st.session_state.username = username
            # Save key in session (if exists)
            st.session_state.user_groq_key = stored_key or ""
            return True, stored_key  # ✅ still returns tuple
    return False, None  # ✅ matches expected unpacking

# ------------------ Save or Update User's Groq API Key ------------------
def save_user_api_key(username, api_key):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET groq_api_key = ? WHERE username = ?", (api_key, username))
    conn.commit()
    conn.close()
    # Also update in session so it's immediately available
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

# ------------------ Forgot Password Functions ------------------

def generate_otp():
    """Generate a random 6-digit OTP as a string."""
    return str(random.randint(100000, 999999))

def send_email_otp(to_email, otp):
    """
    Send OTP via Gmail SMTP using credentials from st.secrets.
    Returns True if successful, False otherwise.
    """
    try:
        # Get email credentials from secrets
        sender_email = st.secrets["email_address"]
        sender_password = st.secrets["email_password"]

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = "Password Reset OTP"

        # Email body
        body = f"""
        Hello,

        Your OTP for password reset is: {otp}

        This OTP will expire in 3 minutes.

        If you did not request this password reset, please ignore this email.

        Best regards,
        Resume App Team
        """

        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

        # Send email
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()

        return True

    except smtplib.SMTPException as e:
        st.error(f"SMTP Error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def get_user_by_email(email):
    """
    Check if an email exists in the users table.
    Returns the username if found, None otherwise.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def update_password_by_email(email, new_password):
    """
    Update the user's password (bcrypt-hashed) for the given email.
    Returns True if successful, False otherwise.
    """
    # Validate password strength
    if not is_strong_password(new_password):
        st.error("Password must be at least 8 characters long and include uppercase, lowercase, number, and special character.")
        return False

    # Hash the new password
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET password = ? WHERE email = ?",
                  (hashed_password.decode('utf-8'), email))
        conn.commit()

        # Check if any row was updated
        if c.rowcount > 0:
            conn.close()
            return True
        else:
            conn.close()
            return False
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        conn.close()
        return False

# ------------------ Database Backup & Download UI ------------------
st.divider()
st.subheader("📦 Database Backup & Download")

if os.path.exists(DB_NAME):
    with open(DB_NAME, "rb") as f:
        st.download_button(
            "⬇️ Download resume_data.db",
            data=f,
            file_name="resume_data_backup.db",
            mime="application/octet-stream"
        )
else:
    st.warning("⚠️ No database file found yet.")
