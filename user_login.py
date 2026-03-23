import psycopg2
import psycopg2.extras
import bcrypt
import streamlit as st
from datetime import datetime, timedelta
import pytz
import re
import os
import random
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import dns.resolver


# ── Cached PostgreSQL connection (shares the same singleton as db_manager) ───
@st.cache_resource
def _get_user_pg_connection():
    """
    Dedicated cached connection for user_login operations.
    @st.cache_resource means it is created once per Streamlit worker process
    and reused on every rerun — no reconnection on every click.
    """
    conn = psycopg2.connect(
        host=st.secrets["SUPABASE_HOST"],
        dbname=st.secrets["SUPABASE_DB"],
        user=st.secrets["SUPABASE_USER"],
        password=st.secrets["SUPABASE_PASSWORD"],
        port=st.secrets["SUPABASE_PORT"],
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    conn.autocommit = False
    return conn


def _conn():
    """Return the cached connection, reconnecting if the socket was dropped."""
    conn = _get_user_pg_connection()
    try:
        conn.isolation_level  # lightweight liveness check
    except Exception:
        st.cache_resource.clear()
        conn = _get_user_pg_connection()
    return conn


def _execute(sql: str, params=None, fetch: str = "none"):
    """
    Run a SQL statement inside an implicit transaction.
    fetch: 'one' | 'all' | 'none'
    Commits on success, rolls back on error.
    """
    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            result = None
            if fetch == "one":
                result = cur.fetchone()
            elif fetch == "all":
                result = cur.fetchall()
            conn.commit()
            return result
    except Exception:
        conn.rollback()
        raise


# ── Utility ──────────────────────────────────────────────────────────────────

def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)


def is_strong_password(password):
    return (
        len(password) >= 8 and
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'[0-9]', password) and
        re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
    )


def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None


def domain_has_mx_record(email):
    try:
        domain = email.split('@')[1]
        dns.resolver.resolve(domain, 'MX')
        return True
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
            dns.resolver.NoNameservers, IndexError):
        return False
    except Exception:
        return True


# ── Existence checks ─────────────────────────────────────────────────────────

def username_exists(username):
    row = _execute(
        "SELECT 1 FROM users WHERE username = %s", (username,), fetch="one"
    )
    return row is not None


def email_exists(email):
    row = _execute(
        "SELECT 1 FROM users WHERE email = %s", (email,), fetch="one"
    )
    return row is not None


# ── Table creation ───────────────────────────────────────────────────────────

def create_user_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        id           SERIAL PRIMARY KEY,
        username     TEXT UNIQUE NOT NULL,
        password     TEXT NOT NULL,
        email        TEXT UNIQUE,
        groq_api_key TEXT
    );
    CREATE TABLE IF NOT EXISTS user_logs (
        id        SERIAL PRIMARY KEY,
        username  TEXT NOT NULL,
        action    TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    """
    try:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    except Exception as e:
        _conn().rollback()
        st.error(f"Error creating tables: {e}")
    # Ensure the login confirmation tokens table also exists
    create_login_tokens_table()


# ── Login confirmation tokens ────────────────────────────────────────────────

def create_login_tokens_table():
    """
    Create the login_tokens table if it doesn't exist.
    Called once at app startup alongside create_user_table().
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS login_tokens (
        id         SERIAL PRIMARY KEY,
        username   TEXT NOT NULL,
        token      TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL,
        used       BOOLEAN NOT NULL DEFAULT FALSE
    );
    """
    try:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    except Exception as e:
        _conn().rollback()
        st.error(f"Error creating login_tokens table: {e}")


def create_login_token(username: str) -> str:
    """
    Generate a cryptographically secure 64-char hex token, store it in
    login_tokens, and return the raw token string.
    Any previous unused tokens for this user are invalidated first.
    """
    # Invalidate stale tokens for this user so only one link is live at a time
    _execute(
        "UPDATE login_tokens SET used = TRUE WHERE username = %s AND used = FALSE",
        (username,),
    )
    token = secrets.token_hex(32)          # 64 hex chars, 256 bits of entropy
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    _execute(
        "INSERT INTO login_tokens (username, token, created_at, used) VALUES (%s, %s, %s, FALSE)",
        (username, token, timestamp),
    )
    return token


def verify_login_token(token: str):
    """
    Validate a login confirmation token.

    Returns (username: str, groq_api_key: str | None) on success.
    Returns (None, None) on failure (expired, used, or not found).

    A token is valid for 10 minutes and can only be used once.
    """
    row = _execute(
        """
        SELECT lt.username, lt.created_at, lt.used, u.groq_api_key
        FROM login_tokens lt
        JOIN users u ON u.username = lt.username
        WHERE lt.token = %s
        """,
        (token,),
        fetch="one",
    )
    if not row:
        return None, None
    if row["used"]:
        return None, None

    # Check 10-minute expiry against IST
    ist = pytz.timezone("Asia/Kolkata")
    created_at = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
    created_at = ist.localize(created_at)
    elapsed = (get_ist_time() - created_at).total_seconds()
    if elapsed > 600:          # 10 minutes
        return None, None

    # Mark token as consumed
    _execute(
        "UPDATE login_tokens SET used = TRUE WHERE token = %s",
        (token,),
    )
    return row["username"], row["groq_api_key"]


def poll_login_confirmed(username: str) -> bool:
    """
    Lightweight poll: returns True if the most recent login token for
    `username` has been consumed (used = TRUE).  Called every few seconds
    by the 'awaiting_email' screen so the laptop session auto-detects when
    the user clicked the confirmation link on another device.
    """
    row = _execute(
        """
        SELECT used FROM login_tokens
        WHERE username = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (username,),
        fetch="one",
    )
    return bool(row and row["used"])


def send_login_confirmation_email(to_email: str, username: str, token: str) -> bool:
    """
    Send a 'Yes, it's me' confirmation link to the user's registered email.
    The app_url is read from st.secrets["APP_URL"] (e.g. https://yourapp.streamlit.app).
    """
    try:
        app_url = st.secrets.get("APP_URL", "http://localhost:8501")
        confirm_url = f"{app_url}?login_token={token}"

        body = f"""Hello {username},

Someone just tried to sign in to your HIRELYZER account.

If this was you, please confirm your login by clicking the link below:

  {confirm_url}

This link expires in 10 minutes and can only be used once.

If you did NOT attempt to log in, you can safely ignore this email — your account remains secure.

Best regards,
HIRELYZER Team
"""
        return _send_email(to_email, "Confirm your HIRELYZER login", body)
    except Exception as e:
        st.error(f"Error sending login confirmation email: {e}")
        return False


def get_email_by_username(username: str):
    """Return the registered email address for a given username."""
    row = _execute(
        "SELECT email FROM users WHERE username = %s", (username,), fetch="one"
    )
    return row["email"] if row else None


# ── OTP helpers ───────────────────────────────────────────────────────────────

def generate_otp():
    return str(random.randint(100000, 999999))


def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Internal SMTP helper used by both registration and password reset."""
    try:
        sender_email = st.secrets["email_address"]
        sender_password = st.secrets["email_password"]

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        return True
    except smtplib.SMTPException as e:
        st.error(f"SMTP Error: {e}")
        return False
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False


def send_registration_otp(to_email: str, otp: str) -> bool:
    body = f"""Hello,

Welcome! Your verification OTP for registration is: {otp}

This OTP will expire in 3 minutes.

If you did not request this registration, please ignore this email.

Best regards,
Resume App Team
"""
    return _send_email(to_email, "Email Verification OTP", body)


def send_email_otp(to_email: str, otp: str) -> bool:
    body = f"""Hello,

Your OTP for password reset is: {otp}

This OTP will expire in 3 minutes.

If you did not request this password reset, please ignore this email.

Best regards,
Resume App Team
"""
    return _send_email(to_email, "Password Reset OTP", body)


# ── Registration ─────────────────────────────────────────────────────────────

def add_user(username, password, email=None):
    """
    Validate details and send OTP.  Does NOT write to DB yet.
    Returns (success: bool, message: str).
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
        'timestamp': get_ist_time(),
    }
    return True, "📧 Verification email sent! Please check your inbox for OTP."


def complete_registration(entered_otp):
    """
    Verify OTP and insert the new user into Supabase.
    Returns (success: bool, message: str).
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
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        _execute(
            "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
            (username, hashed_password, email),
        )
        del st.session_state.pending_registration
        return True, "✅ Registration completed! You can now login."
    except psycopg2.errors.UniqueViolation as e:
        err = str(e)
        if 'username' in err:
            return False, "🚫 Username already exists."
        elif 'email' in err:
            return False, "🚫 Email already exists."
        return False, "🚫 Registration failed. Username or email already exists."
    except Exception as e:
        return False, f"❌ Database error: {e}"


# ── Authentication ────────────────────────────────────────────────────────────

def verify_user(username_or_email, password):
    if '@' in username_or_email:
        sql = "SELECT username, password, groq_api_key FROM users WHERE email = %s"
    else:
        sql = "SELECT username, password, groq_api_key FROM users WHERE username = %s"

    row = _execute(sql, (username_or_email,), fetch="one")
    if row:
        actual_username = row["username"]
        stored_hashed = row["password"]
        stored_key = row["groq_api_key"]

        if bcrypt.checkpw(password.encode('utf-8'), stored_hashed.encode('utf-8')):
            st.session_state.username = actual_username
            st.session_state.user_groq_key = stored_key or ""
            return True, stored_key
    return False, None


# ── API key management ────────────────────────────────────────────────────────

def save_user_api_key(username, api_key):
    _execute(
        "UPDATE users SET groq_api_key = %s WHERE username = %s",
        (api_key, username),
    )
    st.session_state.user_groq_key = api_key


def get_user_api_key(username):
    row = _execute(
        "SELECT groq_api_key FROM users WHERE username = %s", (username,), fetch="one"
    )
    return row["groq_api_key"] if row and row["groq_api_key"] else None


# ── Logging ───────────────────────────────────────────────────────────────────

def log_user_action(username, action):
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    _execute(
        "INSERT INTO user_logs (username, action, timestamp) VALUES (%s, %s, %s)",
        (username, action, timestamp),
    )


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_total_registered_users():
    row = _execute("SELECT COUNT(*) AS cnt FROM users", fetch="one")
    return row["cnt"] if row else 0


def get_logins_today():
    today = get_ist_time().strftime('%Y-%m-%d')
    row = _execute(
        """
        SELECT COUNT(*) AS cnt FROM user_logs
        WHERE action = 'login' AND DATE(timestamp) = %s
        """,
        (today,),
        fetch="one",
    )
    return row["cnt"] if row else 0


def get_all_user_logs():
    rows = _execute(
        "SELECT username, action, timestamp FROM user_logs ORDER BY timestamp DESC",
        fetch="all",
    )
    return [(r["username"], r["action"], r["timestamp"]) for r in (rows or [])]


# ── Forgot password ───────────────────────────────────────────────────────────

def get_user_by_email(email):
    row = _execute(
        "SELECT username FROM users WHERE email = %s", (email,), fetch="one"
    )
    return row["username"] if row else None


def update_password_by_email(email, new_password):
    if not is_strong_password(new_password):
        st.error("Password must be at least 8 characters long and include uppercase, lowercase, number, and special character.")
        return False

    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password = %s WHERE email = %s",
                (hashed_password, email),
            )
            updated = cur.rowcount
        conn.commit()
        return updated > 0
    except Exception as e:
        _conn().rollback()
        st.error(f"Database error: {e}")
        return False
