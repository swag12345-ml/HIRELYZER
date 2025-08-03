import hashlib, os, shelve, json
from langchain_groq import ChatGroq
from langchain_core.exceptions import OutputParserException

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.db")
CONFIG_PATH = os.path.join(WORKING_DIR, "config.json")

# ---- Load Admin API Keys from env or config.json ----
def load_groq_api_keys():
    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        return env_keys.split(",")  # Comma-separated
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        return data.get("GROQ_API_KEYS", [])
    return []

# ---- Hash Prompt for Caching ----
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

# ---- Get Next Admin Key (Rotating Index) ----
def get_next_admin_key(session, tried_keys):
    admin_keys = load_groq_api_keys()
    if not admin_keys:
        return None
    index = session.get("key_index", 0)
    for i in range(len(admin_keys)):
        idx = (index + i) % len(admin_keys)
        key = admin_keys[idx]
        if key not in tried_keys:
            session["key_index"] = idx + 1  # rotate
            return key
    return None

# ---- Main LLM Call with Retry and Fallback ----
def call_llm(prompt: str, session, model="llama-3-3-70b-8192", temperature=0):
    # ✅ Check cache
    cached = get_cached_response(prompt)
    if cached:
        return cached

    tried_keys = set()
    last_error = None

    # ✅ Try user key first
    user_key = session.get("user_groq_key")
    if user_key:
        try:
            llm = ChatGroq(model=model, temperature=temperature, groq_api_key=user_key)
            response = llm.invoke(prompt).content
            set_cached_response(prompt, response)
            return response
        except Exception as e:
            last_error = f"User key failed: {e}"
            tried_keys.add(user_key)

    # ✅ Try all admin keys (rotating, skip bad)
    while True:
        admin_key = get_next_admin_key(session, tried_keys)
        if not admin_key:
            break  # No more unused keys
        tried_keys.add(admin_key)
        try:
            llm = ChatGroq(model=model, temperature=temperature, groq_api_key=admin_key)
            response = llm.invoke(prompt).content
            set_cached_response(prompt, response)
            return response
        except Exception as e:
            last_error = f"Admin key failed: {e}"

    # ❌ Final fallback
    raise RuntimeError(f"❌ All Groq API key attempts failed. Last error: {last_error}")
