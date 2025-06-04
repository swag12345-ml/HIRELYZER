# llm_manager.py
import hashlib, os, shelve, json
from langchain_groq import ChatGroq

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.db")
CONFIG_PATH = os.path.join(WORKING_DIR, "config.json")

# ---- API Key Manager ----
def load_groq_api_keys():
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
        return data.get("GROQ_API_KEYS", [])

def get_next_groq_key(session):
    keys = load_groq_api_keys()
    idx = session.get("key_index", 0)
    key = keys[idx % len(keys)]
    session["key_index"] = idx + 1
    return key

# ---- Caching Mechanism ----
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

# ---- Wrapper for Groq LLM with caching ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    cached = get_cached_response(prompt)
    if cached:
        return cached

    key = get_next_groq_key(session)
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=key)
    response = llm.invoke(prompt).content

    set_cached_response(prompt, response)
    return response
