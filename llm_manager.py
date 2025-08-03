# llm_manager.py
import hashlib, os, shelve, json
from langchain_groq import ChatGroq

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.db")
CONFIG_PATH = os.path.join(WORKING_DIR, "config.json")

# ---- Load Admin API Keys from Environment or config.json ----
def load_groq_api_keys():
    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        return env_keys.split(",")
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        return data.get("GROQ_API_KEYS", [])
    raise ValueError("❌ No Groq API keys found.")

# ---- Select API Key (prioritize based on admin vs user) ----
def get_next_groq_key(session, tried_keys=None):
    tried_keys = tried_keys or set()
    bad_keys = session.get("bad_keys", set())

    user_key = session.get("user_groq_key")
    username = session.get("username", "")
    is_admin = username.lower() == "admin"

    keys_to_try = []

    # Admins try admin keys first
    if is_admin:
        keys_to_try = load_groq_api_keys()
        if user_key:
            keys_to_try.append(user_key)
    else:
        if user_key:
            keys_to_try.append(user_key)
        keys_to_try.extend(load_groq_api_keys())

    # Filter already tried or bad keys
    keys_to_try = [k for k in keys_to_try if k not in tried_keys and k not in bad_keys]
    if not keys_to_try:
        raise RuntimeError("❌ All Groq API keys have failed or been exhausted.")

    idx = session.get("key_index", 0)
    key = keys_to_try[idx % len(keys_to_try)]
    session["key_index"] = idx + 1
    return key

# ---- Caching ----
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def get_cached_response(prompt: str):
    with shelve.open(CACHE_FILE) as db:
        return db.get(hash_prompt(prompt))

def set_cached_response(prompt: str, response: str):
    with shelve.open(CACHE_FILE) as db:
        db[hash_prompt(prompt)] = response

# ---- Main LLM Call ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    # Check cache
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

    raise RuntimeError(f"❌ All Groq API key attempts failed. Last error: {last_error}")
