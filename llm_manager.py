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

# In-memory cooldown tracker: {api_key: datetime_of_last_failure}
_failed_keys = {}

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
    print("✅ Cache initialized (in-memory cooldown tracking).")

init_db()

# ---- Load API Keys ----
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

# ---- Cooldown Tracking (in-memory) ----
def mark_key_failure(api_key):
    _failed_keys[api_key] = datetime.utcnow()

def clear_key_failure(api_key):
    if api_key in _failed_keys:
        del _failed_keys[api_key]

def get_healthy_keys(api_keys):
    now = datetime.utcnow()
    healthy = []
    for key in api_keys:
        last_fail = _failed_keys.get(key)
        if last_fail and (now - last_fail).total_seconds() < FAILURE_COOLDOWN_MINUTES * 60:
            continue  # still cooling down
        healthy.append(key)
    return healthy

# ---- Call LLM ----
def try_call_llm(prompt, api_key, model, temperature):
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content

# ---- Main Entry ----
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
