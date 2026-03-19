"""
LLM Manager — Supabase PostgreSQL backend
Migrated from SQLite to psycopg2, using the same @st.cache_resource singleton
pattern as db_manager.py and user_login.py.
"""

import hashlib
import os
import random
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras
import pytz
import streamlit as st
from langchain_groq import ChatGroq

# ── CONFIG ────────────────────────────────────────────────────────────────────
CACHE_EXPIRY_HOURS      = 24
FAILURE_COOLDOWN_MINUTES = 5
QUOTA_COOLDOWN_MINUTES  = 60
DAILY_KEY_LIMIT         = 800
DEAD_KEY_REMOVE_DAYS    = 3   # auto-remove permanently dead keys after X days


# ── Timezone helper (matches user_login.py) ───────────────────────────────────
def get_ist_time() -> datetime:
    """Return current datetime in Asia/Kolkata (IST) timezone."""
    return datetime.now(pytz.timezone("Asia/Kolkata"))


# ── Cached Supabase connection (one per Streamlit worker) ─────────────────────
@st.cache_resource
def _get_llm_pg_connection():
    """
    Dedicated cached psycopg2 connection for llm_manager operations.
    Created once per Streamlit worker process — never recreated on reruns.
    Mirrors the pattern used in db_manager.py and user_login.py.
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
    """Return the cached connection, reconnecting silently if the socket dropped."""
    conn = _get_llm_pg_connection()
    try:
        conn.isolation_level  # lightweight liveness check
    except Exception:
        st.cache_resource.clear()
        conn = _get_llm_pg_connection()
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


# ── Schema initialisation ─────────────────────────────────────────────────────
def init_db():
    """Create llm_manager tables in Supabase if they don't already exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS llm_cache (
        prompt_hash TEXT PRIMARY KEY,
        response    TEXT        NOT NULL,
        timestamp   TIMESTAMP   NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS key_failures (
        api_key   TEXT PRIMARY KEY,
        fail_time TIMESTAMP NOT NULL,
        reason    TEXT      NOT NULL DEFAULT 'error'
    );

    CREATE TABLE IF NOT EXISTS key_usage (
        api_key     TEXT PRIMARY KEY,
        usage_count INTEGER  NOT NULL DEFAULT 0,
        last_reset  DATE     NOT NULL DEFAULT CURRENT_DATE
    );
    """
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

init_db()


# ── Cache cleanup ─────────────────────────────────────────────────────────────
def cleanup_cache():
    """Delete expired cache rows and permanently dead keys."""
    cutoff_cache = get_ist_time() - timedelta(hours=CACHE_EXPIRY_HOURS)
    cutoff_dead  = get_ist_time() - timedelta(days=DEAD_KEY_REMOVE_DAYS)

    _execute(
        "DELETE FROM llm_cache WHERE timestamp < %s",
        (cutoff_cache,),
    )
    _execute(
        "DELETE FROM key_failures WHERE fail_time < %s",
        (cutoff_dead,),
    )


# ── API key loader ────────────────────────────────────────────────────────────
def load_groq_api_keys():
    """Load Groq keys from Streamlit secrets (preferred) or environment."""
    try:
        secret_keys = st.secrets.get("GROQ_API_KEYS", "")
        if secret_keys:
            keys = [k.strip() for k in secret_keys.split(",") if k.strip()]
            random.shuffle(keys)
            return keys
    except Exception:
        pass

    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        random.shuffle(keys)
        return keys

    raise ValueError("❌ No Groq API keys found in secrets or environment.")


# ── Prompt hashing ────────────────────────────────────────────────────────────
def hash_prompt(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()


# ── Cache R/W ─────────────────────────────────────────────────────────────────
def get_cached_response(prompt: str, model: str):
    """Return cached LLM response if still within CACHE_EXPIRY_HOURS, else None."""
    key    = hash_prompt(prompt, model)
    cutoff = get_ist_time() - timedelta(hours=CACHE_EXPIRY_HOURS)

    row = _execute(
        "SELECT response, timestamp FROM llm_cache WHERE prompt_hash = %s",
        (key,),
        fetch="one",
    )
    if row:
        ts = row["timestamp"]
        # psycopg2 returns datetime objects for TIMESTAMP columns
        if isinstance(ts, str):
            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        # Make naive datetimes timezone-aware (IST) for comparison
        if ts.tzinfo is None:
            ts = pytz.timezone("Asia/Kolkata").localize(ts)
        if ts >= cutoff:
            return row["response"]
    return None


def set_cached_response(prompt: str, model: str, response: str):
    """Upsert a response into the LLM cache."""
    key = hash_prompt(prompt, model)
    _execute(
        """
        INSERT INTO llm_cache (prompt_hash, response, timestamp)
        VALUES (%s, %s, NOW())
        ON CONFLICT (prompt_hash)
        DO UPDATE SET response = EXCLUDED.response,
                      timestamp = EXCLUDED.timestamp
        """,
        (key, response),
    )


# ── Key tracking ──────────────────────────────────────────────────────────────
def increment_key_usage(api_key: str):
    """Increment daily usage counter for a key, resetting if the date changed."""
    _execute(
        """
        INSERT INTO key_usage (api_key, usage_count, last_reset)
        VALUES (%s, 1, CURRENT_DATE)
        ON CONFLICT (api_key) DO UPDATE
            SET usage_count = CASE
                    WHEN key_usage.last_reset = CURRENT_DATE
                    THEN key_usage.usage_count + 1
                    ELSE 1
                END,
                last_reset = CURRENT_DATE
        """,
        (api_key,),
    )


def mark_key_failure(api_key: str, reason: str = "error"):
    """Record (or update) a key failure with a timestamp and reason."""
    _execute(
        """
        INSERT INTO key_failures (api_key, fail_time, reason)
        VALUES (%s, NOW(), %s)
        ON CONFLICT (api_key) DO UPDATE
            SET fail_time = EXCLUDED.fail_time,
                reason    = EXCLUDED.reason
        """,
        (api_key, reason),
    )


def clear_key_failure(api_key: str):
    """Remove a key from the failure table (marks it healthy again)."""
    _execute(
        "DELETE FROM key_failures WHERE api_key = %s",
        (api_key,),
    )


def get_healthy_keys(api_keys: list) -> list:
    """
    Return the subset of api_keys that are:
    - not in cooldown (FAILURE_COOLDOWN_MINUTES / QUOTA_COOLDOWN_MINUTES)
    - below DAILY_KEY_LIMIT
    Result is shuffled for load-balancing.
    """
    now     = get_ist_time()
    today   = now.strftime("%Y-%m-%d")
    healthy = []

    # Pull all relevant rows in two queries instead of N queries
    failures_rows = _execute(
        "SELECT api_key, fail_time, reason FROM key_failures WHERE api_key = ANY(%s)",
        (api_keys,),
        fetch="all",
    ) or []
    usage_rows = _execute(
        "SELECT api_key, usage_count, last_reset FROM key_usage WHERE api_key = ANY(%s)",
        (api_keys,),
        fetch="all",
    ) or []

    # Index into dicts for O(1) lookup
    failures = {r["api_key"]: r for r in failures_rows}
    usages   = {r["api_key"]: r for r in usage_rows}

    for key in api_keys:
        # ── cooldown check ────────────────────────────────────────────────────
        if key in failures:
            f        = failures[key]
            fail_dt  = f["fail_time"]
            if isinstance(fail_dt, str):
                fail_dt = datetime.strptime(fail_dt, "%Y-%m-%d %H:%M:%S")
            # Make naive datetimes timezone-aware (IST) for comparison
            if fail_dt.tzinfo is None:
                fail_dt = pytz.timezone("Asia/Kolkata").localize(fail_dt)
            cooldown = (
                QUOTA_COOLDOWN_MINUTES
                if f["reason"] == "quota"
                else FAILURE_COOLDOWN_MINUTES
            )
            if (now - fail_dt).total_seconds() < cooldown * 60:
                continue  # still in cooldown

        # ── daily quota check ─────────────────────────────────────────────────
        if key in usages:
            u           = usages[key]
            last_reset  = u["last_reset"]
            if isinstance(last_reset, datetime):
                last_reset = last_reset.strftime("%Y-%m-%d")
            elif hasattr(last_reset, "isoformat"):      # date object
                last_reset = last_reset.isoformat()
            usage_count = u["usage_count"] if last_reset == today else 0
            if usage_count >= DAILY_KEY_LIMIT:
                mark_key_failure(key, "quota")
                continue

        healthy.append(key)

    random.shuffle(healthy)
    return healthy


# ── Single LLM call ───────────────────────────────────────────────────────────
def try_call_llm(prompt: str, api_key: str, model: str, temperature: float) -> str:
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content


# ── Main entry point ──────────────────────────────────────────────────────────
def call_llm(
    prompt: str,
    session,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0,
) -> str:
    """
    1. Check Supabase cache — return immediately on hit.
    2. Try user-provided Groq key (if set).
    3. Rotate through healthy admin keys.
    """
    # Step 1 — cache
    cleanup_cache()
    cached = get_cached_response(prompt, model)
    if cached:
        return cached

    if "key_index" not in session:
        session["key_index"] = 0

    user_key = (
        session.get("user_groq_key", "").strip()
        if isinstance(session.get("user_groq_key"), str)
        else ""
    )
    last_error = None

    # Step 2 — user key
    if user_key:
        try:
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, model, response)
            increment_key_usage(user_key)
            return response
        except Exception as e:
            reason = (
                "quota"
                if any(w in str(e).lower() for w in ["quota", "rate limit", "429"])
                else "error"
            )
            mark_key_failure(user_key, reason)
            last_error = e

    # Step 3 — admin key rotation
    admin_keys = get_healthy_keys(load_groq_api_keys())
    if admin_keys:
        start = session["key_index"] % len(admin_keys)
        for offset in range(len(admin_keys)):
            idx = (start + offset) % len(admin_keys)
            key = admin_keys[idx]
            try:
                response = try_call_llm(prompt, key, model, temperature)
                set_cached_response(prompt, model, response)
                increment_key_usage(key)
                clear_key_failure(key)
                session["key_index"] = (idx + 1) % len(admin_keys)
                return response
            except Exception as e:
                reason = (
                    "quota"
                    if any(w in str(e).lower() for w in ["quota", "rate limit", "429"])
                    else "error"
                )
                mark_key_failure(key, reason)
                last_error = e

    return f"❌ LLM unavailable: {last_error or 'No healthy API keys available'}"
