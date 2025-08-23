import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime
import pytz
import re
import json
from google.oauth2 import id_token
from google.auth.transport import requests
import secrets

DB_NAME = "resume_data.db"

# Google OAuth Configuration
GOOGLE_CLIENT_ID = "89576252093-4bfhnj3of7lh3rck0fodis863i0d4qp5.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_REDIRECT_URI = "https://lexibot-zmxdulncckdrifhe8pad5g.streamlit.app/"

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

# ------------------ Check if Google User Already Exists ------------------
def google_user_exists(google_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE google_id = ?", (google_id,))
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
            password TEXT,
            email TEXT,
            groq_api_key TEXT,
            google_id TEXT UNIQUE,
            profile_picture TEXT,
            auth_method TEXT DEFAULT 'password'
        )
    ''')
    
    # Add new columns if they don't exist
    try:
        c.execute('ALTER TABLE users ADD COLUMN email TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN groq_api_key TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN google_id TEXT UNIQUE')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN profile_picture TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN auth_method TEXT DEFAULT "password"')
    except sqlite3.OperationalError:
        pass

    c.execute('''
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            auth_method TEXT DEFAULT 'password'
        )
    ''')

    # Add auth_method column to user_logs if it doesn't exist
    try:
        c.execute('ALTER TABLE user_logs ADD COLUMN auth_method TEXT DEFAULT "password"')
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

# ------------------ Add User (Traditional) ------------------
def add_user(username, password):
    if not is_strong_password(password):
        return False, "⚠ Password must be at least 8 characters long and include uppercase, lowercase, number, and special character."

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, auth_method) VALUES (?, ?, ?)', 
                  (username, hashed_password.decode('utf-8'), 'password'))
        conn.commit()
        return True, "✅ Registered! You can now login."
    except sqlite3.IntegrityError:
        return False, "🚫 Username already exists."
    finally:
        conn.close()

# ------------------ Add Google User ------------------
def add_google_user(google_id, email, name, picture_url):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Use email as username for Google users, or generate unique username if email exists
        username = email.split('@')[0]
        counter = 1
        original_username = username
        
        while username_exists(username):
            username = f"{original_username}_{counter}"
            counter += 1
        
        c.execute('''INSERT INTO users (username, email, google_id, profile_picture, auth_method) 
                     VALUES (?, ?, ?, ?, ?)''', 
                  (username, email, google_id, picture_url, 'google'))
        conn.commit()
        return True, username, "✅ Google account registered successfully!"
    except sqlite3.IntegrityError:
        return False, None, "🚫 Google account already exists."
    finally:
        conn.close()

# ------------------ Verify User & Load Saved API Key ------------------
def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT password, groq_api_key FROM users WHERE username = ? AND auth_method = ?', (username, 'password'))
    result = c.fetchone()
    conn.close()

    if result:
        stored_hashed, stored_key = result
        if stored_hashed and bcrypt.checkpw(password.encode('utf-8'), stored_hashed.encode('utf-8')):
            # Store username in session
            st.session_state.username = username
            st.session_state.auth_method = 'password'
            # Save key in session (if exists)
            st.session_state.user_groq_key = stored_key or ""
            return True, stored_key
    return False, None

# ------------------ Verify Google User ------------------
def verify_google_user(google_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT username, email, groq_api_key, profile_picture FROM users WHERE google_id = ?', (google_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        username, email, stored_key, profile_picture = result
        # Store user info in session
        st.session_state.username = username
        st.session_state.user_email = email
        st.session_state.profile_picture = profile_picture
        st.session_state.auth_method = 'google'
        st.session_state.user_groq_key = stored_key or ""
        return True, username, stored_key
    return False, None, None

# ------------------ Generate Google OAuth URL ------------------
def generate_google_oauth_url():
    state = secrets.token_urlsafe(32)
    st.session_state.oauth_state = state
    
    oauth_url = (
        f"https://accounts.google.com/o/oauth2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={GOOGLE_REDIRECT_URI}&"
        f"scope=openid email profile&"
        f"response_type=code&"
        f"state={state}"
    )
    return oauth_url

# ------------------ Handle Google OAuth Callback ------------------
def handle_google_callback(code, state):
    # Verify state parameter
    if state != st.session_state.get('oauth_state'):
        return False, "Invalid state parameter"
    
    try:
        # Exchange code for tokens
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': GOOGLE_REDIRECT_URI,
        }
        
        import requests as req
        token_response = req.post(token_url, data=token_data)
        token_json = token_response.json()
        
        if 'id_token' not in token_json:
            return False, "Failed to get ID token"
        
        # Verify and decode ID token
        idinfo = id_token.verify_oauth2_token(
            token_json['id_token'], 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        
        google_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo.get('name', email.split('@')[0])
        picture = idinfo.get('picture', '')
        
        # Check if user exists
        if google_user_exists(google_id):
            success, username, stored_key = verify_google_user(google_id)
            if success:
                return True, f"Welcome back, {username}!"
        else:
            # Create new Google user
            success, username, message = add_google_user(google_id, email, name, picture)
            if success:
                # Set session for new user
                st.session_state.username = username
                st.session_state.user_email = email
                st.session_state.profile_picture = picture
                st.session_state.auth_method = 'google'
                st.session_state.user_groq_key = ""
                return True, message
            else:
                return False, message
        
        return True, "Google login successful!"
        
    except Exception as e:
        return False, f"OAuth error: {str(e)}"

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
def log_user_action(username, action, auth_method='password'):
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO user_logs (username, action, timestamp, auth_method) VALUES (?, ?, ?, ?)', 
              (username, action, timestamp, auth_method))
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
    c.execute("SELECT username, action, timestamp, auth_method FROM user_logs ORDER BY timestamp DESC")
    logs = c.fetchall()
    conn.close()
    return logs

# ------------------ Get Authentication Stats ------------------
def get_auth_stats():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Get user counts by auth method
    c.execute("SELECT auth_method, COUNT(*) FROM users GROUP BY auth_method")
    auth_counts = dict(c.fetchall())
    
    # Get login counts by auth method for today
    today = get_ist_time().strftime('%Y-%m-%d')
    c.execute("""
        SELECT auth_method, COUNT(*) FROM user_logs
        WHERE action = 'login' AND DATE(timestamp) = ?
        GROUP BY auth_method
    """, (today,))
    login_counts = dict(c.fetchall())
    
    conn.close()
    return auth_counts, login_counts
