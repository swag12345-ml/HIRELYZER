import hashlib
import os
import random
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from langchain_groq import ChatGroq

# ---- CONFIG ----
CACHE_EXPIRY_HOURS = 24
FAILURE_COOLDOWN_MINUTES = 5
QUOTA_COOLDOWN_MINUTES = 60
DAILY_KEY_LIMIT = 800
DEAD_KEY_REMOVE_DAYS = 3


# ---- Cached Connection (own singleton, no import from db_manager) ----
# Uses the same @st.cache_resource pattern as db_manager to avoid
# opening a new connection on every call, without causing a circular import.

def _get_llm_connection():
    """
    Returns a single cached psycopg2 connection for llm_manager.
    @st.cache_resource ensures it is NOT recreated on every Streamlit rerun.
    Kept separate from db_manager's connection to avoid circular imports,
    but uses the same pattern and credentials.
    """
    import streamlit as st
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

# Wrap with cache_resource at module level
try:
    import streamlit as st
    _get_llm_connection = st.cache_resource(_get_llm_connection)
except Exception:
    pass  # Running outside Streamlit (tests/scripts) — no caching needed


def get_conn():
    """
    Returns the cached connection.
    Reconnects automatically if the connection was dropped.
    """
    conn = _get_llm_connection()
    try:
        conn.isolation_level  # lightweight liveness check
    except Exception:
        try:
            import streamlit as st
            st.cache_resource.clear()
        except Exception:
            pass
        conn = _get_llm_connection()
    return conn


# ---- Table Setup ----
# Run once via init_tables() OR paste into Supabase SQL Editor:
#
# CREATE TABLE IF NOT EXISTS llm_cache (
#     prompt_hash TEXT PRIMARY KEY,
#     response    TEXT,
#     timestamp   TIMESTAMPTZ DEFAULT NOW()
# );
# CREATE TABLE IF NOT EXISTS key_failures (
#     api_key   TEXT PRIMARY KEY,
#     fail_time TIMESTAMPTZ,
#     reason    TEXT
# );
# CREATE TABLE IF NOT EXISTS key_usage (
#     api_key     TEXT PRIMARY KEY,
#     usage_count INTEGER DEFAULT 0,
#     last_reset  DATE
# );

def init_tables():
    """Create the 3 llm_manager tables in Supabase if they don't exist."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    response    TEXT,
                    timestamp   TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS key_failures (
                    api_key   TEXT PRIMARY KEY,
                    fail_time TIMESTAMPTZ,
                    reason    TEXT
                );
                CREATE TABLE IF NOT EXISTS key_usage (
                    api_key     TEXT PRIMARY KEY,
                    usage_count INTEGER DEFAULT 0,
                    last_reset  DATE
                );
            """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e


# ---- Auto-clean expired cache ----
def cleanup_cache():
    """Delete expired cache rows and permanently dead key records."""
    cutoff      = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
    cutoff_dead = datetime.utcnow() - timedelta(days=DEAD_KEY_REMOVE_DAYS)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM llm_cache    WHERE timestamp  < %s", (cutoff,))
            cur.execute("DELETE FROM key_failures WHERE fail_time  < %s", (cutoff_dead,))
        conn.commit()
    except Exception:
        conn.rollback()


# ---- Load API Keys ----
def load_groq_api_keys():
    try:
        import streamlit as st
        secret_keys = st.secrets.get("GROQ_API_KEYS", "")
        if secret_keys:
            keys = [k.strip() for k in secret_keys.split(",") if k.strip()]
            random.shuffle(keys)
            return keys
    except Exception:
        pass
    env_keys = os.getenv("GROQ_API_KEYS", "")
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
    """Return a valid cached LLM response, or None if not found / expired."""
    key    = hash_prompt(prompt, model)
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
    conn   = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT response FROM llm_cache WHERE prompt_hash = %s AND timestamp >= %s",
                (key, cutoff)
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def set_cached_response(prompt: str, model: str, response: str):
    """Upsert LLM response into the cache table."""
    key  = hash_prompt(prompt, model)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO llm_cache (prompt_hash, response, timestamp)
                VALUES (%s, %s, NOW())
                ON CONFLICT (prompt_hash) DO UPDATE
                    SET response  = EXCLUDED.response,
                        timestamp = NOW()
            """, (key, response))
        conn.commit()
    except Exception:
        conn.rollback()


# ---- Key Tracking ----
def increment_key_usage(api_key: str):
    """Increment daily usage for a key, resetting count if it's a new day."""
    today = datetime.utcnow().date()
    conn  = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT usage_count, last_reset FROM key_usage WHERE api_key = %s",
                (api_key,)
            )
            row = cur.fetchone()
            if row:
                usage_count, last_reset = row
                if isinstance(last_reset, str):
                    last_reset = datetime.strptime(last_reset, "%Y-%m-%d").date()
                if last_reset != today:
                    cur.execute(
                        "UPDATE key_usage SET usage_count = 1, last_reset = %s WHERE api_key = %s",
                        (today, api_key)
                    )
                else:
                    cur.execute(
                        "UPDATE key_usage SET usage_count = usage_count + 1 WHERE api_key = %s",
                        (api_key,)
                    )
            else:
                cur.execute(
                    "INSERT INTO key_usage (api_key, usage_count, last_reset) VALUES (%s, 1, %s)",
                    (api_key, today)
                )
        conn.commit()
    except Exception:
        conn.rollback()


def mark_key_failure(api_key: str, reason: str = "error"):
    """Record or update a key failure with timestamp and reason."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO key_failures (api_key, fail_time, reason)
                VALUES (%s, NOW(), %s)
                ON CONFLICT (api_key) DO UPDATE
                    SET fail_time = NOW(),
                        reason    = EXCLUDED.reason
            """, (api_key, reason))
        conn.commit()
    except Exception:
        conn.rollback()


def clear_key_failure(api_key: str):
    """Remove a key from the failures table after a successful call."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM key_failures WHERE api_key = %s", (api_key,))
        conn.commit()
    except Exception:
        conn.rollback()


def get_healthy_keys(api_keys: list) -> list:
    """
    Return keys not in active cooldown and under daily quota.
    Uses 2 bulk queries to avoid N+1 database calls.
    """
    now   = datetime.utcnow()
    today = now.date()
    conn  = get_conn()

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT api_key, fail_time, reason FROM key_failures")
            failures = {r["api_key"]: r for r in cur.fetchall()}

            cur.execute("SELECT api_key, usage_count, last_reset FROM key_usage")
            usage_map = {r["api_key"]: r for r in cur.fetchall()}
    except Exception:
        shuffled = list(api_keys)
        random.shuffle(shuffled)
        return shuffled

    healthy = []
    for key in api_keys:
        # --- Cooldown check ---
        if key in failures:
            fail_time = failures[key]["fail_time"]
            reason    = failures[key]["reason"]
            if isinstance(fail_time, str):
                fail_time = datetime.fromisoformat(fail_time)
            if fail_time.tzinfo is not None:
                fail_time = fail_time.replace(tzinfo=None)
            cooldown_mins = QUOTA_COOLDOWN_MINUTES if reason == "quota" else FAILURE_COOLDOWN_MINUTES
            if (now - fail_time).total_seconds() < cooldown_mins * 60:
                continue

        # --- Daily quota check ---
        if key in usage_map:
            usage_count = usage_map[key]["usage_count"]
            last_reset  = usage_map[key]["last_reset"]
            if isinstance(last_reset, str):
                last_reset = datetime.strptime(last_reset, "%Y-%m-%d").date()
            if last_reset == today and usage_count >= DAILY_KEY_LIMIT:
                mark_key_failure(key, "quota")
                continue

        healthy.append(key)

    random.shuffle(healthy)
    return healthy


# ---- LLM Call ----
def try_call_llm(prompt: str, api_key: str, model: str, temperature: float) -> str:
    """Make a single LLM call via Groq."""
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content


# ---- Main Entry Point ----
def call_llm(prompt: str, session, model: str = "llama-3.3-70b-versatile", temperature: float = 0) -> str:
    """
    Main LLM call with:
    1. Cache check       (Supabase llm_cache table)
    2. User key attempt  (from session["user_groq_key"])
    3. Admin key rotation with per-key cooldown + daily quota tracking
    """
    # Step 1: Cleanup once per session
    if not session.get("_cache_cleaned"):
        cleanup_cache()
        session["_cache_cleaned"] = True

    # Step 2: Cache lookup
    cached = get_cached_response(prompt, model)
    if cached:
        return cached

    if "key_index" not in session:
        session["key_index"] = 0

    user_key   = session.get("user_groq_key", "")
    user_key   = user_key.strip() if isinstance(user_key, str) else ""
    last_error = None

    # Step 3: Try user-provided key first
    if user_key:
        try:
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, model, response)
            increment_key_usage(user_key)
            return response
        except Exception as e:
            reason = "quota" if any(w in str(e).lower() for w in ["quota", "rate limit", "429"]) else "error"
            mark_key_failure(user_key, reason)
            last_error = e

    # Step 4: Rotate through admin keys
    admin_keys = get_healthy_keys(load_groq_api_keys())
    if admin_keys:
        start_index = session["key_index"] % len(admin_keys)
        for offset in range(len(admin_keys)):
            idx = (start_index + offset) % len(admin_keys)
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
                last_error = e

    return f"❌ LLM unavailable: {last_error or 'No healthy API keys available'}"
