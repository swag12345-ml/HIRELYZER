import hashlib
import os
import sqlite3
from datetime import datetime, timedelta
from langchain_groq import ChatGroq

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.sqlite")
CACHE_EXPIRY_HOURS = 24
FAILURE_COOLDOWN_MINUTES = 5  # Wait before retrying a failed key

# ---- Initialize SQLite DB ----
def init_db():
    with sqlite3.connect(CACHE_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS key_usage (
                api_key TEXT PRIMARY KEY,
                last_used DATETIME,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                last_error TEXT
            )
        """)
    print("✅ SQLite cache & key usage tables initialized.")

init_db()

# ---- Load Admin API Keys ----
def load_groq_api_keys():
    try:
        import streamlit as st
        secret_keys = st.secrets.get("GROQ_API_KEYS", "")
        if secret_keys:
            return [k.strip() for k in secret_keys.split(",") if k.strip()]
    except Exception as e:
        print(f"⚠️ Could not load st.secrets: {e}")

    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        return [k.strip() for k in env_keys.split(",") if k.strip()]

    raise ValueError("❌ No Groq API keys found. Use st.secrets or the GROQ_API_KEYS environment variable.")

# ---- Prompt Hashing ----
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

# ---- Cache Functions ----
def get_cached_response(prompt: str):
    key = hash_prompt(prompt)
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)

    with sqlite3.connect(CACHE_FILE) as conn:
        cur = conn.execute("SELECT response, timestamp FROM llm_cache WHERE prompt_hash = ?", (key,))
        row = cur.fetchone()

    if row:
        response, ts_str = row
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        if ts >= cutoff:
            return response
    return None

def set_cached_response(prompt: str, response: str):
    key = hash_prompt(prompt)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(CACHE_FILE) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO llm_cache (prompt_hash, response, timestamp)
            VALUES (?, ?, ?)
        """, (key, response, ts))

# ---- Key Usage Tracking ----
def update_key_usage(api_key, success=True, error_msg=None):
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(CACHE_FILE) as conn:
        if success:
            conn.execute("""
                INSERT INTO key_usage (api_key, last_used, success_count, fail_count, last_error)
                VALUES (?, ?, 1, 0, NULL)
                ON CONFLICT(api_key) DO UPDATE SET
                    last_used = excluded.last_used,
                    success_count = success_count + 1,
                    last_error = NULL
            """, (api_key, now_str))
        else:
            conn.execute("""
                INSERT INTO key_usage (api_key, last_used, success_count, fail_count, last_error)
                VALUES (?, ?, 0, 1, ?)
                ON CONFLICT(api_key) DO UPDATE SET
                    last_used = excluded.last_used,
                    fail_count = fail_count + 1,
                    last_error = excluded.last_error
            """, (api_key, now_str, error_msg))

def get_healthy_keys(admin_keys):
    """Return keys that are not in cooldown from last failure."""
    now = datetime.utcnow()
    healthy_keys = []
    with sqlite3.connect(CACHE_FILE) as conn:
        for key in admin_keys:
            cur = conn.execute("SELECT last_used, fail_count FROM key_usage WHERE api_key = ?", (key,))
            row = cur.fetchone()
            if row:
                last_used_str, fail_count = row
                if last_used_str:
                    last_used = datetime.strptime(last_used_str, "%Y-%m-%d %H:%M:%S")
                    if fail_count > 0 and (now - last_used).total_seconds() < FAILURE_COOLDOWN_MINUTES * 60:
                        continue  # still cooling down
            healthy_keys.append(key)
    return healthy_keys

# ---- Internal LLM Call ----
def try_call_llm(prompt, api_key, model, temperature):
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content

# ---- Main Call ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    cached = get_cached_response(prompt)
    if cached:
        return cached

    # ✅ Normalize user key
    user_key = session.get("user_groq_key")
    user_key = user_key.strip() if isinstance(user_key, str) else ""
    last_error = None

    # ✅ Load and filter admin keys
    admin_keys = load_groq_api_keys()
    admin_keys = [k for k in admin_keys if k]
    admin_keys = get_healthy_keys(admin_keys)  # filter out cooling down keys

    # 1️⃣ Try user key first
    if user_key:
        try:
            print("🔑 Trying user API key")
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, response)
            update_key_usage(user_key, success=True)
            return response
        except Exception as e:
            print(f"❌ User API key failed: {e}")
            update_key_usage(user_key, success=False, error_msg=str(e))
            last_error = e

    # 2️⃣ Smart rotate through admin keys
    if admin_keys:
        start_idx = session.get("key_index", 0)
        for i in range(len(admin_keys)):
            idx = (start_idx + i) % len(admin_keys)
            key = admin_keys[idx]
            try:
                print(f"🔁 Trying admin API key {idx + 1} of {len(admin_keys)}")
                response = try_call_llm(prompt, key, model, temperature)
                session["key_index"] = (idx + 1) % len(admin_keys)
                set_cached_response(prompt, response)
                update_key_usage(key, success=True)
                return response
            except Exception as e:
                print(f"❌ Admin key {idx + 1} failed: {e}")
                update_key_usage(key, success=False, error_msg=str(e))
                last_error = e

    # ❌ Final failure
    raise RuntimeError(f"❌ All Groq API keys failed. Last error: {last_error}")
