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
MAX_BAD_KEYS_PER_SESSION = 50  # prevent memory bloat

# ---- DB Init ----
def init_db():
    try:
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10) as conn:
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
    except Exception:
        # If DB init fails, continue without crashing
        pass

init_db()

# ---- Auto-clean expired cache ----
def cleanup_cache():
    try:
        cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=5) as conn:
            conn.execute("DELETE FROM llm_cache WHERE timestamp < ?", (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
        # Auto-remove dead keys older than DEAD_KEY_REMOVE_DAYS
        cutoff_dead = datetime.utcnow() - timedelta(days=DEAD_KEY_REMOVE_DAYS)
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=5) as conn:
            conn.execute("DELETE FROM key_failures WHERE fail_time < ?", (cutoff_dead.strftime("%Y-%m-%d %H:%M:%S"),))
    except Exception:
        # Continue if cleanup fails
        pass

# ---- Load API Keys ----
def load_groq_api_keys():
    keys = []
    
    # Try Streamlit secrets first
    try:
        import streamlit as st
        secret_keys = st.secrets.get("GROQ_API_KEYS", "")
        if isinstance(secret_keys, str) and secret_keys.strip():
            keys = [k.strip() for k in secret_keys.split(",") if k.strip() and len(k.strip()) > 20]
    except Exception:
        pass
    
    # Try environment variable if no keys from secrets
    if not keys:
        try:
            env_keys = os.getenv("GROQ_API_KEYS", "")
            if env_keys:
                keys = [k.strip() for k in env_keys.split(",") if k.strip() and len(k.strip()) > 20]
        except Exception:
            pass
    
    if not keys:
        return []
    
    # Shuffle and validate keys
    try:
        random.shuffle(keys)
    except Exception:
        pass
    
    return keys[:70]  # Limit to prevent memory issues

# ---- Hash Prompt ----
def hash_prompt(prompt: str, model: str) -> str:
    try:
        safe_prompt = str(prompt)[:10000]  # Limit prompt length
        safe_model = str(model)[:100]
        return hashlib.sha256(f"{safe_model}|{safe_prompt}".encode("utf-8")).hexdigest()
    except Exception:
        # Fallback hash
        return hashlib.sha256(f"default|{len(str(prompt))}".encode("utf-8")).hexdigest()

# ---- Cache Handling ----
def get_cached_response(prompt: str, model: str):
    if not prompt or not model:
        return None
    
    try:
        key = hash_prompt(prompt, model)
        cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=3) as conn:
            row = conn.execute("SELECT response, timestamp FROM llm_cache WHERE prompt_hash = ?", (key,)).fetchone()
        
        if row and len(row) == 2:
            response, ts_str = row
            try:
                if datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") >= cutoff:
                    return response
            except ValueError:
                pass
    except Exception:
        pass
    
    return None

def set_cached_response(prompt: str, model: str, response: str):
    if not prompt or not model or not response:
        return
    
    try:
        key = hash_prompt(prompt, model)
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        safe_response = str(response)[:50000]  # Limit response size
        
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=3) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO llm_cache (prompt_hash, response, timestamp)
                VALUES (?, ?, ?)
            """, (key, safe_response, ts))
    except Exception:
        pass

# ---- Key Tracking ----
def increment_key_usage(api_key):
    if not api_key or len(str(api_key)) < 10:
        return
    
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=3) as conn:
            row = conn.execute("SELECT usage_count, last_reset FROM key_usage WHERE api_key=?", (api_key,)).fetchone()
            if row and len(row) == 2:
                usage_count, last_reset = row
                if last_reset != today:
                    conn.execute("UPDATE key_usage SET usage_count=1, last_reset=? WHERE api_key=?", (today, api_key))
                else:
                    conn.execute("UPDATE key_usage SET usage_count=usage_count+1 WHERE api_key=?", (api_key,))
            else:
                conn.execute("INSERT INTO key_usage (api_key, usage_count, last_reset) VALUES (?, ?, ?)",
                             (api_key, 1, today))
    except Exception:
        pass

def mark_key_failure(api_key, reason="error"):
    if not api_key:
        return
    
    try:
        safe_reason = str(reason)[:50]
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=3) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO key_failures (api_key, fail_time, reason)
                VALUES (?, ?, ?)
            """, (api_key, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), safe_reason))
    except Exception:
        pass

def clear_key_failure(api_key):
    if not api_key:
        return
    
    try:
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=3) as conn:
            conn.execute("DELETE FROM key_failures WHERE api_key = ?", (api_key,))
    except Exception:
        pass

def get_healthy_keys(api_keys, session=None):
    if not api_keys:
        return []
    
    try:
        now = datetime.utcnow()
        healthy = []
        bad_in_session = set()
        
        if session and isinstance(session, dict):
            bad_keys_raw = session.get("bad_keys", set())
            if isinstance(bad_keys_raw, (set, list)):
                bad_in_session = set(list(bad_keys_raw)[:MAX_BAD_KEYS_PER_SESSION])
        
        with sqlite3.connect(DB_FILE, check_same_thread=False, timeout=5) as conn:
            for key in api_keys:
                if not key or key in bad_in_session:
                    continue
                    
                try:
                    row = conn.execute("SELECT fail_time, reason FROM key_failures WHERE api_key = ?", (key,)).fetchone()
                    if row and len(row) == 2:
                        fail_time, reason = row
                        try:
                            fail_dt = datetime.strptime(fail_time, "%Y-%m-%d %H:%M:%S")
                            cooldown = QUOTA_COOLDOWN_MINUTES if reason == "quota" else FAILURE_COOLDOWN_MINUTES
                            if (now - fail_dt).total_seconds() < cooldown * 60:
                                continue
                        except ValueError:
                            continue
                    
                    usage = conn.execute("SELECT usage_count, last_reset FROM key_usage WHERE api_key=?", (key,)).fetchone()
                    if usage and len(usage) == 2:
                        usage_count, last_reset = usage
                        if last_reset != now.strftime("%Y-%m-%d"):
                            usage_count = 0
                        if usage_count >= DAILY_KEY_LIMIT:
                            mark_key_failure(key, "quota")
                            continue
                    
                    healthy.append(key)
                except Exception:
                    continue
        
        return healthy
    except Exception:
        return list(api_keys)[:10]  # Fallback with limited keys

# ---- LLM Call ----
def try_call_llm(prompt, api_key, model, temperature):
    if not prompt or not api_key or not model:
        raise ValueError("Invalid parameters")
    
    try:
        # Validate and sanitize inputs
        safe_prompt = str(prompt)[:8000]  # Limit prompt length
        safe_model = str(model)[:100]
        safe_temp = max(0, min(1, float(temperature or 0)))
        
        llm = ChatGroq(model=safe_model, temperature=safe_temp, groq_api_key=api_key)
        result = llm.invoke(safe_prompt)
        
        if hasattr(result, 'content') and result.content:
            return str(result.content)
        else:
            raise ValueError("Empty response")
            
    except Exception as e:
        raise e

# ---- Main ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    # Validate inputs
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        return "Error: Invalid prompt"
    
    if not session or not isinstance(session, dict):
        session = {}
    
    # Sanitize inputs
    safe_prompt = str(prompt).strip()[:8000]
    safe_model = str(model or "llama-3.3-70b-versatile")[:100]
    safe_temperature = 0
    try:
        safe_temperature = max(0, min(1, float(temperature or 0)))
    except (ValueError, TypeError):
        safe_temperature = 0
    
    # Clean cache periodically (not every call to reduce load)
    try:
        if random.random() < 0.1:  # 10% chance
            cleanup_cache()
    except Exception:
        pass
    
    # Try cache first
    try:
        cached = get_cached_response(safe_prompt, safe_model)
        if cached:
            return cached
    except Exception:
        pass
    
    # Initialize session state safely
    try:
        if "key_index" not in session or not isinstance(session["key_index"], int):
            session["key_index"] = 0
        if "bad_keys" not in session or not isinstance(session["bad_keys"], (set, list)):
            session["bad_keys"] = set()
        else:
            # Limit bad_keys size to prevent memory bloat
            bad_keys = session["bad_keys"]
            if isinstance(bad_keys, list):
                bad_keys = set(bad_keys)
            if len(bad_keys) > MAX_BAD_KEYS_PER_SESSION:
                session["bad_keys"] = set(list(bad_keys)[:MAX_BAD_KEYS_PER_SESSION])
    except Exception:
        session["key_index"] = 0
        session["bad_keys"] = set()

    user_key = None
    try:
        user_key_raw = session.get("user_groq_key", "")
        if isinstance(user_key_raw, str) and len(user_key_raw.strip()) > 20:
            user_key = user_key_raw.strip()
    except Exception:
        pass

    last_error = "No working API keys available"

    # Try user key first
    if user_key:
        try:
            response = try_call_llm(safe_prompt, user_key, safe_model, safe_temperature)
            if response:
                set_cached_response(safe_prompt, safe_model, response)
                increment_key_usage(user_key)
                return response
        except Exception as e:
            try:
                error_msg = str(e).lower()
                reason = "quota" if any(w in error_msg for w in ["quota", "rate limit", "429", "exceeded"]) else "error"
                mark_key_failure(user_key, reason)
                session["bad_keys"].add(user_key)
                last_error = str(e)
            except Exception:
                last_error = "User key failed"

    # Try admin keys
    try:
        admin_keys = load_groq_api_keys()
        if admin_keys:
            healthy_keys = get_healthy_keys(admin_keys, session)
            if healthy_keys:
                start = session.get("key_index", 0)
                if not isinstance(start, int) or start < 0:
                    start = 0
                
                for i in range(len(healthy_keys)):
                    try:
                        idx = (start + i) % len(healthy_keys)
                        key = healthy_keys[idx]
                        
                        if not key:
                            continue
                            
                        response = try_call_llm(safe_prompt, key, safe_model, safe_temperature)
                        if response:
                            set_cached_response(safe_prompt, safe_model, response)
                            increment_key_usage(key)
                            clear_key_failure(key)
                            session["key_index"] = (idx + 1) % len(healthy_keys)
                            return response
                            
                    except Exception as e:
                        try:
                            error_msg = str(e).lower()
                            reason = "quota" if any(w in error_msg for w in ["quota", "rate limit", "429", "exceeded"]) else "error"
                            mark_key_failure(key, reason)
                            session["bad_keys"].add(key)
                            last_error = str(e)
                        except Exception:
                            last_error = "Admin key failed"
                            continue
        else:
            last_error = "No API keys configured"
    except Exception as e:
        last_error = f"System error: {str(e)}"

    return f"⚠️ LLM temporarily unavailable. Please try again in a few minutes. ({last_error[:100]})"
