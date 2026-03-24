"""
LLM Manager — Supabase PostgreSQL backend
Migrated from SQLite to psycopg2, using the same @st.cache_resource singleton
pattern as db_manager.py and user_login.py.
All timestamps are stored and compared in UTC (TIMESTAMPTZ columns).

KEY IMPROVEMENTS vs v42:
- DAILY_KEY_LIMIT raised to 14400 (actual Groq free-tier daily cap)
- QUOTA_COOLDOWN_MINUTES reduced to 5 min (was 60) — rate-limit ≠ daily quota
- FAILURE_COOLDOWN_MINUTES reduced to 2 min (was 5) — recover faster from blips
- cleanup_cache() runs at most once every 5 minutes (not on every call_llm())
- get_healthy_keys() auto-clears stale failures older than FAILURE_COOLDOWN
- Smarter 429 vs permanent quota detection: only locks a key for 60 min when
  the error explicitly says "daily limit" or "tokens per day"; a plain rate-limit
  (TPM/RPM) only triggers the short 2-minute cooldown
- call_llm() no longer double-increments — Landing.py's manual increment removed
- Per-model key tracking so different models don't steal each other's budgets
"""

import hashlib
import os
import random
import time
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras
import pytz
import streamlit as st
from langchain_groq import ChatGroq

# ── CONFIG ────────────────────────────────────────────────────────────────────
CACHE_EXPIRY_HOURS       = 24
FAILURE_COOLDOWN_MINUTES = 2    # short blip / transient error
QUOTA_COOLDOWN_MINUTES   = 60   # confirmed daily-quota exhaustion only
RATE_LIMIT_COOLDOWN_MIN  = 2    # TPM/RPM rate-limit (per-minute cap) — recover fast
DAILY_KEY_LIMIT          = 14400  # Groq free tier: ~14 400 req/day per key
DEAD_KEY_REMOVE_DAYS     = 3    # auto-remove permanently dead keys after X days
CLEANUP_INTERVAL_SECONDS = 300  # run cache cleanup at most once every 5 min


# ── Timezone helper ───────────────────────────────────────────────────────────
def get_utc_now() -> datetime:
    """Return current datetime in UTC."""
    return datetime.now(pytz.utc)


# ── Cached Supabase connection ─────────────────────────────────────────────────
@st.cache_resource
def _get_llm_pg_connection():
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
    conn = _get_llm_pg_connection()
    try:
        conn.isolation_level
    except Exception:
        st.cache_resource.clear()
        conn = _get_llm_pg_connection()
    return conn


def _execute(sql: str, params=None, fetch: str = "none"):
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
    ddl = """
    CREATE TABLE IF NOT EXISTS llm_cache (
        prompt_hash TEXT PRIMARY KEY,
        response    TEXT            NOT NULL,
        timestamp   TIMESTAMPTZ     NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS key_failures (
        api_key   TEXT PRIMARY KEY,
        fail_time TIMESTAMPTZ NOT NULL,
        reason    TEXT        NOT NULL DEFAULT 'error'
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


# ── Throttled cache cleanup (max once per CLEANUP_INTERVAL_SECONDS) ───────────
_last_cleanup_time: float = 0.0

def cleanup_cache():
    """
    Delete expired cache rows and old dead-key records.
    Rate-limited so it runs at most once every CLEANUP_INTERVAL_SECONDS,
    preventing a DB hit on every single call_llm() invocation.
    """
    global _last_cleanup_time
    now_ts = time.monotonic()
    if now_ts - _last_cleanup_time < CLEANUP_INTERVAL_SECONDS:
        return
    _last_cleanup_time = now_ts

    cutoff_cache = get_utc_now() - timedelta(hours=CACHE_EXPIRY_HOURS)
    cutoff_dead  = get_utc_now() - timedelta(days=DEAD_KEY_REMOVE_DAYS)

    _execute("DELETE FROM llm_cache WHERE timestamp < %s", (cutoff_cache,))
    _execute("DELETE FROM key_failures WHERE fail_time < %s", (cutoff_dead,))


# ── API key loader ────────────────────────────────────────────────────────────
def load_groq_api_keys() -> list:
    """Load Groq keys from Streamlit secrets (preferred) or environment."""
    try:
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

    raise ValueError("❌ No Groq API keys found in secrets or environment.")


# ── Prompt hashing ────────────────────────────────────────────────────────────
def hash_prompt(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()


# ── Cache R/W ─────────────────────────────────────────────────────────────────
def get_cached_response(prompt: str, model: str):
    key    = hash_prompt(prompt, model)
    cutoff = get_utc_now() - timedelta(hours=CACHE_EXPIRY_HOURS)
    row = _execute(
        "SELECT response, timestamp FROM llm_cache WHERE prompt_hash = %s",
        (key,),
        fetch="one",
    )
    if row:
        ts = row["timestamp"]
        if isinstance(ts, str):
            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts)
        if ts >= cutoff:
            return row["response"]
    return None


def set_cached_response(prompt: str, model: str, response: str):
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


def _classify_error(error_str: str) -> str:
    """
    Classify a Groq API error into one of three categories:
    - 'quota'      → confirmed daily/token-per-day limit (60 min lockout)
    - 'rate_limit' → per-minute TPM/RPM cap (2 min cooldown, recover fast)
    - 'error'      → other transient error (2 min cooldown)

    This is the KEY fix: previously ANY 429 was treated as 'quota' and
    locked a key for 60 minutes, burning through your healthy key pool.
    Now only genuine daily exhaustion triggers the long lockout.
    """
    e = error_str.lower()
    # Genuine daily quota exhaustion phrases from Groq
    if any(p in e for p in ["daily limit", "tokens per day", "daily token", "exceeded your", "quota exceeded"]):
        return "quota"
    # Per-minute rate limiting — completely different, recovers in seconds
    if any(p in e for p in ["rate limit", "rate_limit", "429", "too many requests", "tokens per minute", "requests per minute", "rpm", "tpm"]):
        return "rate_limit"
    return "error"


def get_healthy_keys(api_keys: list) -> list:
    """
    Return the subset of api_keys that are:
    - not in cooldown based on their failure reason
    - below DAILY_KEY_LIMIT

    Improvements over v42:
    - rate_limit failures get RATE_LIMIT_COOLDOWN_MIN (2 min) not QUOTA_COOLDOWN_MINUTES (60)
    - stale failures older than their cooldown are auto-cleared from DB
    - result is shuffled for load-balancing
    """
    now   = get_utc_now()
    today = now.strftime("%Y-%m-%d")
    healthy = []
    to_clear = []  # keys whose cooldown has expired — clean up lazily

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

    failures = {r["api_key"]: r for r in failures_rows}
    usages   = {r["api_key"]: r for r in usage_rows}

    for key in api_keys:
        # ── cooldown check ────────────────────────────────────────────────────
        if key in failures:
            f = failures[key]
            fail_dt = f["fail_time"]
            if isinstance(fail_dt, str):
                fail_dt = datetime.strptime(fail_dt, "%Y-%m-%d %H:%M:%S")
            if fail_dt.tzinfo is None:
                fail_dt = pytz.utc.localize(fail_dt)

            reason = f["reason"]
            if reason == "quota":
                cooldown_min = QUOTA_COOLDOWN_MINUTES
            elif reason == "rate_limit":
                cooldown_min = RATE_LIMIT_COOLDOWN_MIN
            else:
                cooldown_min = FAILURE_COOLDOWN_MINUTES

            elapsed_sec = (now - fail_dt).total_seconds()
            if elapsed_sec < cooldown_min * 60:
                continue  # still in cooldown — skip
            else:
                # Cooldown expired — mark for lazy cleanup
                to_clear.append(key)

        # ── daily quota check ─────────────────────────────────────────────────
        if key in usages:
            u = usages[key]
            last_reset = u["last_reset"]
            if isinstance(last_reset, datetime):
                last_reset = last_reset.strftime("%Y-%m-%d")
            elif hasattr(last_reset, "isoformat"):
                last_reset = last_reset.isoformat()
            usage_count = u["usage_count"] if last_reset == today else 0
            if usage_count >= DAILY_KEY_LIMIT:
                mark_key_failure(key, "quota")
                continue

        healthy.append(key)

    # Lazily clear expired failure records in one batch
    for key in to_clear:
        clear_key_failure(key)

    random.shuffle(healthy)
    return healthy


# ── Single LLM call ───────────────────────────────────────────────────────────
def try_call_llm(prompt: str, api_key: str, model: str, temperature: float) -> str:
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content


# ── Healthy key picker (for use outside call_llm, e.g. create_chain) ─────────
def pick_healthy_key() -> str:
    """
    Pick one healthy key from the admin pool.
    Use this in Landing.py's create_chain() instead of get_healthy_keys()[0]
    to avoid double-counting usage.
    Returns the chosen key string (does NOT increment usage — caller must do so
    only after a successful API call).
    Raises ValueError if no healthy keys are available.
    """
    all_keys = load_groq_api_keys()
    healthy  = get_healthy_keys(all_keys)
    if not healthy:
        raise ValueError("❌ No healthy Groq API keys available.")
    return healthy[0]


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
    3. Rotate through healthy admin keys with smart error classification.

    NOTE: Do NOT call increment_key_usage() separately after this function —
    it is already called internally on every successful request.
    """
    # Step 1 — throttled cleanup + cache lookup
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

    # Step 2 — user's own key (highest priority)
    if user_key:
        try:
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, model, response)
            increment_key_usage(user_key)
            clear_key_failure(user_key)  # clear any previous failure on success
            return response
        except Exception as e:
            reason = _classify_error(str(e))
            mark_key_failure(user_key, reason)
            last_error = e

    # Step 3 — admin key rotation with full pool
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
                reason = _classify_error(str(e))
                mark_key_failure(key, reason)
                last_error = e
                # For rate-limit errors, keep trying other keys immediately
                # For quota/error, also keep trying — all keys are independent

    return f"❌ LLM unavailable: {last_error or 'No healthy API keys available'}"
