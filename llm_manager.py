import hashlib
import os
import sqlite3
from datetime import datetime, timedelta
from langchain_groq import ChatGroq

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.sqlite")
CACHE_EXPIRY_HOURS = 24
FAILURE_COOLDOWN_MINUTES = 5

# ---- Initialize DB ----
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
                fail_count INTEGER DEFAULT 0
            )
        """)
    print("✅ Cache + minimal key_usage initialized.")

init_db()

# ---- Load API Keys from Streamlit secrets or env ----
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

    raise ValueError("❌ No Groq API keys found. Add them to st.secrets or env variable.")

# ---- Prompt Hashing ----
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

# ---- Cache Handling ----
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

# ---- Cooldown Tracking Only ----
def mark_key_failure(api_key):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(CACHE_FILE) as conn:
        conn.execute("""
            INSERT INTO key_usage (api_key, last_used, fail_count)
            VALUES (?, ?, 1)
            ON CONFLICT(api_key) DO UPDATE SET
                last_used = excluded.last_used,
                fail_count = fail_count + 1
        """, (api_key, now))

def clear_key_failure(api_key):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(CACHE_FILE) as conn:
        conn.execute("""
            INSERT INTO key_usage (api_key, last_used, fail_count)
            VALUES (?, ?, 0)
            ON CONFLICT(api_key) DO UPDATE SET
                last_used = excluded.last_used,
                fail_count = 0
        """, (api_key, now))

# ---- Filter Keys Not in Cooldown ----
def get_healthy_keys(api_keys):
    now = datetime.utcnow()
    healthy = []
    with sqlite3.connect(CACHE_FILE) as conn:
        for key in api_keys:
            cur = conn.execute("SELECT last_used, fail_count FROM key_usage WHERE api_key = ?", (key,))
            row = cur.fetchone()
            if row:
                last_used_str, fail_count = row
                if fail_count > 0 and last_used_str:
                    last_used = datetime.strptime(last_used_str, "%Y-%m-%d %H:%M:%S")
                    if (now - last_used).total_seconds() < FAILURE_COOLDOWN_MINUTES * 60:
                        continue  # cooling down
            healthy.append(key)
    return healthy

# ---- Call LLM ----
def try_call_llm(prompt, api_key, model, temperature):
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content

# ---- Public Main Entry ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    cached = get_cached_response(prompt)
    if cached:
        return cached

    user_key = session.get("user_groq_key", "").strip() if isinstance(session.get("user_groq_key"), str) else ""
    last_error = None

    # Try user key
    if user_key:
        try:
            print("🔑 Trying user key")
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, response)
            return response
        except Exception as e:
            print(f"❌ User key failed: {e}")
            mark_key_failure(user_key)
            last_error = e

    # Try admin pool
    admin_keys = get_healthy_keys(load_groq_api_keys())
    if admin_keys:
        start = session.get("key_index", 0)
        for i in range(len(admin_keys)):
            idx = (start + i) % len(admin_keys)
            key = admin_keys[idx]
            try:
                print(f"🔁 Trying admin key {idx + 1} of {len(admin_keys)}")
                response = try_call_llm(prompt, key, model, temperature)
                session["key_index"] = (idx + 1) % len(admin_keys)
                set_cached_response(prompt, response)
                clear_key_failure(key)
                return response
            except Exception as e:
                print(f"❌ Admin key {idx + 1} failed: {e}")
                mark_key_failure(key)
                last_error = e

    raise RuntimeError(f"❌ All keys failed. Last error: {last_error}")
