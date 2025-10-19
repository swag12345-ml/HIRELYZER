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
import dns.resolver

# Persistent storage path for Streamlit Cloud
os.makedirs(".streamlit_storage", exist_ok=True)
DB_NAME = os.path.join(".streamlit_storage", "resume_data.db")

# ------------------ Utility: Get IST Time ------------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

# Show IST Time in UI


# ------------------ Password Strength Validator ------------------
def is_strong_password(password):
    return (
        len(password) >= 8 and
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'[0-9]', password) and
        re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
    )

# ------------------ Email Validation ------------------
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# ------------------ Check Email Domain MX Record ------------------
def domain_has_mx_record(email):
    """
    Check if the email domain has valid MX records.
    Returns True if MX records exist, False otherwise.
    """
    try:
        domain = email.split('@')[1]
        dns.resolver.resolve(domain, 'MX')
        return True
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, IndexError):
        return False
    except Exception:
        return True

# ------------------ Check if Username Already Exists ------------------
def username_exists(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

# ------------------ Check if Email Already Exists ------------------
def email_exists(email):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE email = ?", (email,))
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
            email TEXT UNIQUE,
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
    try:
        c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_email ON users(email)')
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

# ------------------ Add User (with OTP Verification) ------------------
def add_user(username, password, email=None):
    """
    Validate user registration details and send OTP for email verification.
    Does NOT insert user into database yet - that happens in complete_registration().
    Returns (success, message) tuple.
    """
    if not is_strong_password(password):
        return False, "⚠ Password must be at least 8 characters long and include uppercase, lowercase, number, and special character."

    if not email:
        return False, "⚠ Email is required for registration."

    if not is_valid_email(email):
        return False, "⚠ Invalid email format. Please provide a valid email address."

    if not domain_has_mx_record(email):
        return False, "⚠ Email domain does not exist or has no valid mail server."

    if email_exists(email):
        return False, "🚫 Email already exists. Please use a different email."

    if username_exists(username):
        return False, "🚫 Username already exists."

    otp = generate_otp()

    if not send_registration_otp(email, otp):
        return False, "❌ Failed to send OTP email. Please check your email address and try again."

    st.session_state.pending_registration = {
        'username': username,
        'password': password,
        'email': email,
        'otp': otp,
        'timestamp': get_ist_time()
    }

    return True, "📧 Verification email sent! Please check your inbox for OTP."

def send_registration_otp(to_email, otp):
    """
    Send OTP for registration verification via Gmail SMTP.
    Returns True if successful, False otherwise.
    """
    try:
        sender_email = st.secrets["email_address"]
        sender_password = st.secrets["email_password"]

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = "Email Verification OTP"

        body = f"""
        Hello,

        Welcome! Your verification OTP for registration is: {otp}

        This OTP will expire in 3 minutes.

        If you did not request this registration, please ignore this email.

        Best regards,
        Resume App Team
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

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

def complete_registration(entered_otp):
    """
    Verify OTP and complete user registration by inserting into database.
    Returns (success, message) tuple.
    """
    if 'pending_registration' not in st.session_state:
        return False, "⚠ No pending registration found. Please start registration again."

    pending = st.session_state.pending_registration
    stored_otp = pending['otp']
    timestamp = pending['timestamp']

    time_elapsed = (get_ist_time() - timestamp).total_seconds()
    if time_elapsed > 180:
        del st.session_state.pending_registration
        return False, "⏱ OTP has expired. Please register again."

    if entered_otp != stored_otp:
        return False, "❌ Invalid OTP. Please try again."

    username = pending['username']
    password = pending['password']
    email = pending['email']

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                  (username, hashed_password.decode('utf-8'), email))
        conn.commit()
        del st.session_state.pending_registration
        return True, "✅ Registration completed! You can now login."
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return False, "🚫 Username already exists."
        elif 'email' in str(e):
            return False, "🚫 Email already exists."
        else:
            return False, "🚫 Registration failed. Username or email already exists."
    except Exception as e:
        return False, f"❌ Database error: {str(e)}"
    finally:
        conn.close()

# ------------------ Verify User & Load Saved API Key ------------------
def verify_user(username_or_email, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    if '@' in username_or_email:
        c.execute('SELECT username, password, groq_api_key FROM users WHERE email = ?', (username_or_email,))
    else:
        c.execute('SELECT username, password, groq_api_key FROM users WHERE username = ?', (username_or_email,))

    result = c.fetchone()
    conn.close()

    if result:
        if '@' in username_or_email:
            actual_username, stored_hashed, stored_key = result
        else:
            actual_username = username_or_email
            stored_hashed, stored_key = result[1], result[2]

        if bcrypt.checkpw(password.encode('utf-8'), stored_hashed.encode('utf-8')):
            st.session_state.username = actual_username
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
