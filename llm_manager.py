# llm_manager.py
import hashlib, os, shelve, json
from langchain_groq import ChatGroq

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.db")
CONFIG_PATH = os.path.join(WORKING_DIR, "config.json")

# ---- Load Admin API Keys from Environment or config.json ----
def load_groq_api_keys():
    # 1. Try environment variable first (for deployment)
    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        return env_keys.split(",")  # comma-separated keys

    # 2. Fallback to local config.json (for development)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        return data.get("GROQ_API_KEYS", [])

    # 3. Neither found — fail gracefully
    raise ValueError("❌ No Groq API keys found. Set GROQ_API_KEYS env var or use config.json.")

# ---- Select API Key (User → Admin Fallback) ----
def get_next_groq_key(session):
    # ✅ 1. Use user-supplied key if present
    if session.get("user_groq_key"):
        return session["user_groq_key"]

    # ✅ 2. Fallback to rotating among admin keys
    keys = load_groq_api_keys()
    if not keys:
        raise ValueError("❌ No Groq API keys available.")

    idx = session.get("key_index", 0)
    key = keys[idx % len(keys)]
    session["key_index"] = idx + 1
    return key

# ---- Prompt Hashing for Response Caching ----
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def get_cached_response(prompt: str):
    key = hash_prompt(prompt)
    with shelve.open(CACHE_FILE) as db:
        return db.get(key)

def set_cached_response(prompt: str, response: str):
    key = hash_prompt(prompt)
    with shelve.open(CACHE_FILE) as db:
        db[key] = response

# ---- Main LLM Call with Caching & API Key Selection ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    # ✅ Check cache first
    cached = get_cached_response(prompt)
    if cached:
        return cached

    # ✅ Choose proper Groq API key
    key = get_next_groq_key(session)

    # ✅ Call Groq LLM
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=key)
    response = llm.invoke(prompt).content

    # ✅ Cache and return result
    set_cached_response(prompt, response)
    return response
