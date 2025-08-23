import sqlite3
import bcrypt
import streamlit as st
import datetime   # ✅ add this line
from datetime import datetime
import pytz
import re
import requests
import json
import os
from urllib.parse import urlencode
import secrets

DB_NAME = "resume_data.db"

# ------------------ Google OAuth Configuration ------------------
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your-google-client-secret")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/auth/callback")

# Google OAuth URLs
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

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
            password TEXT,
            email TEXT UNIQUE,
            groq_api_key TEXT,
            google_id TEXT UNIQUE,
            profile_picture TEXT,
            auth_provider TEXT DEFAULT 'local',
            created_at TEXT,
            last_login TEXT
        )
    ''')
    
    # Add new columns if they don't exist
    columns_to_add = [
        ('email', 'TEXT UNIQUE'),
        ('groq_api_key', 'TEXT'),
        ('google_id', 'TEXT UNIQUE'),
        ('profile_picture', 'TEXT'),
        ('auth_provider', 'TEXT DEFAULT "local"'),
        ('created_at', 'TEXT'),
        ('last_login', 'TEXT')
    ]
    
    for column_name, column_type in columns_to_add:
        try:
            c.execute(f'ALTER TABLE users ADD COLUMN {column_name} {column_type}')
        except sqlite3.OperationalError:
            pass

    # Enhanced user logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            auth_provider TEXT DEFAULT 'local'
        )
    ''')
    
    # Add new columns to user_logs if they don't exist
    log_columns_to_add = [
        ('ip_address', 'TEXT'),
        ('user_agent', 'TEXT'),
        ('auth_provider', 'TEXT DEFAULT "local"')
    ]
    
    for column_name, column_type in log_columns_to_add:
        try:
            c.execute(f'ALTER TABLE user_logs ADD COLUMN {column_name} {column_type}')
        except sqlite3.OperationalError:
            pass

    # OAuth state table for security
    c.execute('''
        CREATE TABLE IF NOT EXISTS oauth_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

# ------------------ Google OAuth Functions ------------------
def generate_oauth_state():
    """Generate a secure random state for OAuth"""
    state = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + datetime.timedelta(minutes=10)).isoformat()
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO oauth_states (state, created_at, expires_at) VALUES (?, ?, ?)',
              (state, datetime.now().isoformat(), expires_at))
    conn.commit()
    conn.close()
    
    return state

def verify_oauth_state(state):
    """Verify OAuth state and clean up expired states"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Clean up expired states
    c.execute('DELETE FROM oauth_states WHERE expires_at < ?', (datetime.now().isoformat(),))
    
    # Check if state exists and is valid
    c.execute('SELECT 1 FROM oauth_states WHERE state = ?', (state,))
    valid = c.fetchone() is not None
    
    if valid:
        # Remove used state
        c.execute('DELETE FROM oauth_states WHERE state = ?', (state,))
    
    conn.commit()
    conn.close()
    return valid

def get_google_auth_url():
    """Generate Google OAuth authorization URL"""
    state = generate_oauth_state()
    st.session_state.oauth_state = state
    
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': GOOGLE_REDIRECT_URI,
        'scope': 'openid email profile',
        'response_type': 'code',
        'state': state,
        'access_type': 'offline',
        'prompt': 'consent'
    }
    
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

def exchange_code_for_token(code, state):
    """Exchange authorization code for access token"""
    if not verify_oauth_state(state):
        return None, "Invalid or expired OAuth state"
    
    data = {
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': GOOGLE_REDIRECT_URI,
    }
    
    try:
        response = requests.post(GOOGLE_TOKEN_URL, data=data)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as e:
        return None, f"Token exchange failed: {str(e)}"

def get_google_user_info(access_token):
    """Get user information from Google"""
    headers = {'Authorization': f'Bearer {access_token}'}
    
    try:
        response = requests.get(GOOGLE_USER_INFO_URL, headers=headers)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as e:
        return None, f"Failed to get user info: {str(e)}"

def create_or_update_google_user(user_info):
    """Create or update user from Google OAuth"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    google_id = user_info.get('id')
    email = user_info.get('email')
    name = user_info.get('name', email.split('@')[0])
    picture = user_info.get('picture')
    
    current_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if user exists by Google ID or email
    c.execute('SELECT username, id FROM users WHERE google_id = ? OR email = ?', (google_id, email))
    existing_user = c.fetchone()
    
    if existing_user:
        username, user_id = existing_user
        # Update existing user
        c.execute('''UPDATE users SET 
                     google_id = ?, email = ?, profile_picture = ?, 
                     auth_provider = 'google', last_login = ?
                     WHERE id = ?''',
                  (google_id, email, picture, current_time, user_id))
    else:
        # Create new user
        username = name.lower().replace(' ', '_')
        # Ensure username is unique
        base_username = username
        counter = 1
        while username_exists(username):
            username = f"{base_username}_{counter}"
            counter += 1
        
        c.execute('''INSERT INTO users 
                     (username, email, google_id, profile_picture, auth_provider, created_at, last_login)
                     VALUES (?, ?, ?, ?, 'google', ?, ?)''',
                  (username, email, google_id, picture, current_time, current_time))
        
        # Log registration
        log_user_action(username, 'register', auth_provider='google')
    
    conn.commit()
    conn.close()
    
    return username

# ------------------ Add User (Enhanced) ------------------
def add_user(username, password, email=None):
    if not is_strong_password(password):
        return False, "⚠ Password must be at least 8 characters long and include uppercase, lowercase, number, and special character."

    if email and email_exists(email):
        return False, "🚫 Email already exists."

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    current_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO users (username, password, email, auth_provider, created_at) 
                     VALUES (?, ?, ?, 'local', ?)''', 
                  (username, hashed_password.decode('utf-8'), email, current_time))
        conn.commit()
        
        # Log registration
        log_user_action(username, 'register', auth_provider='local')
        
        return True, "✅ Registered! You can now login."
    except sqlite3.IntegrityError:
        return False, "🚫 Username already exists."
    finally:
        conn.close()

# ------------------ Verify User & Load Saved API Key (Enhanced) ------------------
def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT password, groq_api_key FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    
    if result:
        stored_hashed, stored_key = result
        if stored_hashed and bcrypt.checkpw(password.encode('utf-8'), stored_hashed.encode('utf-8')):
            # Update last login
            current_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
            c.execute('UPDATE users SET last_login = ? WHERE username = ?', (current_time, username))
            conn.commit()
            
            # Store username in session
            st.session_state.username = username
            # Save key in session (if exists)
            st.session_state.user_groq_key = stored_key or ""
            
            # Log login
            log_user_action(username, 'login', auth_provider='local')
            
            conn.close()
            return True, stored_key
    
    conn.close()
    return False, None

# ------------------ Google Login Handler ------------------
def handle_google_login():
    """Handle Google OAuth login process"""
    # Check for authorization code in URL parameters
    query_params = st.experimental_get_query_params()
    
    if 'code' in query_params and 'state' in query_params:
        code = query_params['code'][0]
        state = query_params['state'][0]
        
        # Exchange code for token
        token_data, error = exchange_code_for_token(code, state)
        if error:
            st.error(f"Authentication failed: {error}")
            return False
        
        # Get user info
        access_token = token_data.get('access_token')
        user_info, error = get_google_user_info(access_token)
        if error:
            st.error(f"Failed to get user information: {error}")
            return False
        
        # Create or update user
        username = create_or_update_google_user(user_info)
        
        # Set session state
        st.session_state.username = username
        st.session_state.user_groq_key = get_user_api_key(username) or ""
        
        # Log login
        log_user_action(username, 'login', auth_provider='google')
        
        # Clear URL parameters
        st.experimental_set_query_params()
        
        st.success(f"✅ Successfully logged in with Google! Welcome, {username}")
        st.experimental_rerun()
        return True
    
    return False

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

# ------------------ Log User Action (Enhanced) ------------------
def log_user_action(username, action, auth_provider='local', ip_address=None, user_agent=None):
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''INSERT INTO user_logs 
                 (username, action, timestamp, ip_address, user_agent, auth_provider) 
                 VALUES (?, ?, ?, ?, ?, ?)''', 
              (username, action, timestamp, ip_address, user_agent, auth_provider))
    conn.commit()
    conn.close()

# ------------------ Logout Function ------------------
def logout_user():
    """Handle user logout"""
    if 'username' in st.session_state:
        username = st.session_state.username
        log_user_action(username, 'logout')
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.success("✅ Successfully logged out!")
        st.experimental_rerun()

# ------------------ Get User Profile ------------------
def get_user_profile(username):
    """Get complete user profile information"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''SELECT username, email, profile_picture, auth_provider, 
                        created_at, last_login FROM users WHERE username = ?''', (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return {
            'username': result[0],
            'email': result[1],
            'profile_picture': result[2],
            'auth_provider': result[3],
            'created_at': result[4],
            'last_login': result[5]
        }
    return None

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

# ------------------ Get User Statistics ------------------
def get_user_statistics():
    """Get comprehensive user statistics"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Total users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    
    # Users by auth provider
    c.execute("SELECT auth_provider, COUNT(*) FROM users GROUP BY auth_provider")
    auth_stats = dict(c.fetchall())
    
    # Today's activity
    today = get_ist_time().strftime('%Y-%m-%d')
    c.execute("""
        SELECT action, COUNT(*) FROM user_logs
        WHERE DATE(timestamp) = ?
        GROUP BY action
    """, (today,))
    today_activity = dict(c.fetchall())
    
    # Recent registrations (last 7 days)
    week_ago = (get_ist_time() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    c.execute("""
        SELECT COUNT(*) FROM users
        WHERE DATE(created_at) >= ?
    """, (week_ago,))
    recent_registrations = c.fetchone()[0]
    
    conn.close()
    
    return {
        'total_users': total_users,
        'auth_providers': auth_stats,
        'today_activity': today_activity,
        'recent_registrations': recent_registrations
    }

# ------------------ Get All User Logs (Enhanced) ------------------
def get_all_user_logs(limit=100):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT username, action, timestamp, auth_provider 
        FROM user_logs 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    logs = c.fetchall()
    conn.close()
    return logs

# ------------------ Get User Activity History ------------------
def get_user_activity_history(username, limit=50):
    """Get activity history for a specific user"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT action, timestamp, auth_provider 
        FROM user_logs 
        WHERE username = ?
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (username, limit))
    logs = c.fetchall()
    conn.close()
    return logs

