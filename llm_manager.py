# llm_manager.py
# llm_manager.py
import hashlib, os, shelve, json
from langchain_groq import ChatGroq
import streamlit as st  # ✅ Required for secrets on Streamlit Cloud

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.db")

# ---- Load Admin API Keys from Streamlit secrets ----
def load_groq_api_keys():
    keys_string = st.secrets.get("GROQ_API_KEYS", "")
    if not keys_string:
        raise ValueError("❌ No Groq API keys found in st.secrets. Set GROQ_API_KEYS in secrets.")
    return [k.strip() for k in keys_string.split(",") if k.strip()]

# ---- Select API Key (prioritize based on admin vs user) ----
def get_next_groq_key(session, tried_keys=None):
    tried_keys = tried_keys or set()
    bad_keys = session.get("bad_keys", set())

    user_key = session.get("user_groq_key")
    username = session.get("username", "")
    is_admin = username.lower() == "admin"

    keys_to_try = []

    if is_admin:
        keys_to_try.extend(load_groq_api_keys())
        if user_key:
            keys_to_try.append(user_key)
    else:
        if user_key:
            keys_to_try.append(user_key)
        keys_to_try.extend(load_groq_api_keys())

    # Remove already tried or failed keys
    keys_to_try = [k for k in keys_to_try if k not in tried_keys and k not in bad_keys]

    if not keys_to_try:
        raise RuntimeError("❌ All Groq API keys have failed or been exhausted.")

    idx = session.get("key_index", 0)
    key = keys_to_try[idx % len(keys_to_try)]
    session["key_index"] = idx + 1
    return key

# ---- Prompt Caching ----
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def get_cached_response(prompt: str):
    with shelve.open(CACHE_FILE) as db:
        return db.get(hash_prompt(prompt))

def set_cached_response(prompt: str, response: str):
    with shelve.open(CACHE_FILE) as db:
        db[hash_prompt(prompt)] = response

# ---- Main LLM Call with Retry ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    cached = get_cached_response(prompt)
    if cached:
        return cached

    max_attempts = 5
    tried_keys = set()
    last_error = None

    for _ in range(max_attempts):
        try:
            key = get_next_groq_key(session, tried_keys)
            tried_keys.add(key)

            llm = ChatGroq(model=model, temperature=temperature, groq_api_key=key)
            response = llm.invoke(prompt).content

            set_cached_response(prompt, response)
            return response

        except Exception as e:
            print(f"⚠️ Key failed: {key} | Error: {e}")
            session.setdefault("bad_keys", set()).add(key)
            last_error = e

    # Final failure after all attempts
    raise RuntimeError(f"❌ All Groq API key attempts failed. Last error: {last_error}")
