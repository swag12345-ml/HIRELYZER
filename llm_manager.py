import hashlib
import os
import random
import sqlite3
from datetime import datetime, timedelta
from langchain_groq import ChatGroq

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(WORKING_DIR, "llm_data.sqlite")
CACHE_EXPIRY_HOURS = 24
FAILURE_COOLDOWN_MINUTES = 5       # Temporary error cooldown
QUOTA_COOLDOWN_MINUTES = 60        # Quota exhaustion cooldown

# ---- Initialize DB ----
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS key_failures (
                api_key TEXT PRIMARY KEY,
                fail_time DATETIME,
                reason TEXT
            )
        """)
    print("✅ DB initialized (cache + cooldown persistence).")

init_db()

# ---- Load API Keys ----
def load_groq_api_keys():
    try:
        import streamlit as st
        secret_keys = st.secrets.get("GROQ_API_KEYS", "")
        if secret_keys:
            keys = [k.strip() for k in secret_keys.split(",") if k.strip()]
            random.shuffle(keys)
            return keys
    except Exception as e:
        print(f"⚠️ Could not load st.secrets: {e}")

    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        random.shuffle(keys)
        return keys

    raise ValueError("❌ No Groq API keys found. Add them to st.secrets or env variable.")

# ---- Prompt Hashing ----
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

# ---- Cache Handling ----
def get_cached_response(prompt: str):
    key = hash_prompt(prompt)
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
    with sqlite3.connect(DB_FILE) as conn:
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
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO llm_cache (prompt_hash, response, timestamp)
            VALUES (?, ?, ?)
        """, (key, response, ts))

# ---- Cooldown Tracking ----
def mark_key_failure(api_key, reason="error"):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO key_failures (api_key, fail_time, reason)
            VALUES (?, ?, ?)
        """, (api_key, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), reason))

def clear_key_failure(api_key):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM key_failures WHERE api_key = ?", (api_key,))

def get_healthy_keys(api_keys):
    now = datetime.utcnow()
    healthy = []
    with sqlite3.connect(DB_FILE) as conn:
        for key in api_keys:
            row = conn.execute("SELECT fail_time, reason FROM key_failures WHERE api_key = ?", (key,)).fetchone()
            if row:
                fail_time, reason = row
                fail_dt = datetime.strptime(fail_time, "%Y-%m-%d %H:%M:%S")
                cooldown = QUOTA_COOLDOWN_MINUTES if reason == "quota" else FAILURE_COOLDOWN_MINUTES
                if (now - fail_dt).total_seconds() < cooldown * 60:
                    continue
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

    if "key_index" not in session:
        session["key_index"] = random.randint(0, len(load_groq_api_keys()) - 1)

    user_key = session.get("user_groq_key", "").strip() if isinstance(session.get("user_groq_key"), str) else ""
    last_error = None

    # Try user key first
    if user_key:
        try:
            print(f"🔑 Trying user key: {user_key[:6]}...{user_key[-4:]}")
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, response)
            return response
        except Exception as e:
            reason = "quota" if any(w in str(e).lower() for w in ["quota", "rate limit", "429"]) else "error"
            print(f"❌ User key failed ({reason}): {e}")
            mark_key_failure(user_key, reason)
            last_error = e

    # Try admin pool
    admin_keys = get_healthy_keys(load_groq_api_keys())
    if admin_keys:
        start = session["key_index"]
        for i in range(len(admin_keys)):
            idx = (start + i) % len(admin_keys)
            key = admin_keys[idx]
            try:
                print(f"🔁 Trying admin key {idx+1}/{len(admin_keys)}: {key[:6]}...{key[-4:]}")
                response = try_call_llm(prompt, key, model, temperature)
                session["key_index"] = (idx + 1) % len(admin_keys)
                set_cached_response(prompt, response)
                clear_key_failure(key)
                return response
            except Exception as e:
                reason = "quota" if any(w in str(e).lower() for w in ["quota", "rate limit", "429"]) else "error"
                print(f"❌ Admin key {idx+1} failed ({reason}): {e}")
                mark_key_failure(key, reason)
                last_error = e

    print(f"⚠️ All LLM keys failed. Last error: {last_error}")
    return "LLM unavailable: using fallback response"
