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
FAILURE_COOLDOWN_MINUTES = 5
QUOTA_COOLDOWN_MINUTES = 60
DAILY_KEY_LIMIT = 800
DEAD_KEY_REMOVE_DAYS = 3  # auto-remove permanently dead keys after X days

# ---- DB Init ----
def init_db():
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS key_usage (
                api_key TEXT PRIMARY KEY,
                usage_count INTEGER,
                last_reset DATE
            )
        """)
init_db()

# ---- Auto-clean expired cache ----
def cleanup_cache():
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        conn.execute("DELETE FROM llm_cache WHERE timestamp < ?", (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
    # Auto-remove dead keys older than DEAD_KEY_REMOVE_DAYS
    cutoff_dead = datetime.utcnow() - timedelta(days=DEAD_KEY_REMOVE_DAYS)
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        conn.execute("DELETE FROM key_failures WHERE fail_time < ?", (cutoff_dead.strftime("%Y-%m-%d %H:%M:%S"),))

# ---- Load API Keys ----
def load_groq_api_keys():
    try:
        import streamlit as st
        secret_keys = st.secrets.get("GROQ_API_KEYS", "")
        if secret_keys:
            keys = [k.strip() for k in secret_keys.split(",") if k.strip()]
            random.shuffle(keys)
            return keys
    except:
        pass
    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        random.shuffle(keys)
        return keys
    raise ValueError("❌ No Groq API keys found.")

# ---- Hash Prompt ----
def hash_prompt(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()

# ---- Cache Handling ----
def get_cached_response(prompt: str, model: str):
    key = hash_prompt(prompt, model)
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        row = conn.execute("SELECT response, timestamp FROM llm_cache WHERE prompt_hash = ?", (key,)).fetchone()
    if row:
        response, ts_str = row
        if datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") >= cutoff:
            return response
    return None

def set_cached_response(prompt: str, model: str, response: str):
    key = hash_prompt(prompt, model)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO llm_cache (prompt_hash, response, timestamp)
            VALUES (?, ?, ?)
        """, (key, response, ts))

# ---- Key Tracking ----
def increment_key_usage(api_key):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        row = conn.execute("SELECT usage_count, last_reset FROM key_usage WHERE api_key=?", (api_key,)).fetchone()
        if row:
            usage_count, last_reset = row
            if last_reset != today:
                conn.execute("UPDATE key_usage SET usage_count=1, last_reset=? WHERE api_key=?", (today, api_key))
            else:
                conn.execute("UPDATE key_usage SET usage_count=usage_count+1 WHERE api_key=?", (api_key,))
        else:
            conn.execute("INSERT INTO key_usage (api_key, usage_count, last_reset) VALUES (?, ?, ?)",
                         (api_key, 1, today))

def mark_key_failure(api_key, reason="error"):
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO key_failures (api_key, fail_time, reason)
            VALUES (?, ?, ?)
        """, (api_key, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), reason))

def clear_key_failure(api_key):
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        conn.execute("DELETE FROM key_failures WHERE api_key = ?", (api_key,))

def get_healthy_keys(api_keys, session=None):
    now = datetime.utcnow()
    healthy = []
    bad_in_session = session.get("bad_keys", set()) if session else set()
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        for key in api_keys:
            if key in bad_in_session:
                continue
            row = conn.execute("SELECT fail_time, reason FROM key_failures WHERE api_key = ?", (key,)).fetchone()
            if row:
                fail_time, reason = row
                fail_dt = datetime.strptime(fail_time, "%Y-%m-%d %H:%M:%S")
                cooldown = QUOTA_COOLDOWN_MINUTES if reason == "quota" else FAILURE_COOLDOWN_MINUTES
                if (now - fail_dt).total_seconds() < cooldown * 60:
                    continue
            usage = conn.execute("SELECT usage_count, last_reset FROM key_usage WHERE api_key=?", (key,)).fetchone()
            if usage:
                usage_count, last_reset = usage
                if last_reset != now.strftime("%Y-%m-%d"):
                    usage_count = 0
                if usage_count >= DAILY_KEY_LIMIT:
                    mark_key_failure(key, "quota")
                    continue
            healthy.append(key)
    return healthy

# ---- LLM Call ----
def try_call_llm(prompt, api_key, model, temperature):
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content

# ---- Main ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    cleanup_cache()  # auto-clean on each call
    cached = get_cached_response(prompt, model)
    if cached:
        return cached

    if "key_index" not in session:
        session["key_index"] = 0
    if "bad_keys" not in session:
        session["bad_keys"] = set()

    user_key = session.get("user_groq_key", "").strip() if isinstance(session.get("user_groq_key"), str) else ""
    last_error = None

    # Try user key first
    if user_key:
        try:
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, model, response)
            increment_key_usage(user_key)
            return response
        except Exception as e:
            reason = "quota" if any(w in str(e).lower() for w in ["quota", "rate limit", "429"]) else "error"
            mark_key_failure(user_key, reason)
            session["bad_keys"].add(user_key)
            last_error = e

    # Try admin keys
    admin_keys = get_healthy_keys(load_groq_api_keys(), session)
    if admin_keys:
        start = session["key_index"]
        for i in range(len(admin_keys)):
            idx = (start + i) % len(admin_keys)
            key = admin_keys[idx]
            try:
                response = try_call_llm(prompt, key, model, temperature)
                set_cached_response(prompt, model, response)
                increment_key_usage(key)
                clear_key_failure(key)
                session["key_index"] = (idx + 1) % len(admin_keys)
                return response
            except Exception as e:
                reason = "quota" if any(w in str(e).lower() for w in ["quota", "rate limit", "429"]) else "error"
                mark_key_failure(key, reason)
                session["bad_keys"].add(key)
                last_error = e

    return f"LLM unavailable: {last_error}"



