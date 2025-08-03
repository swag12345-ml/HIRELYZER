import hashlib, os, shelve
from langchain_groq import ChatGroq

# ---- CONFIG ----
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORKING_DIR, "llm_cache.db")

# ---- Load Admin API Keys from st.secrets or env ----
def load_groq_api_keys():
    try:
        import streamlit as st
        secret_keys = st.secrets.get("GROQ_API_KEYS", "")
        if secret_keys:
            return [k.strip() for k in secret_keys.split(",")]
    except Exception as e:
        print(f"⚠️ Could not load st.secrets: {e}")

    env_keys = os.getenv("GROQ_API_KEYS")
    if env_keys:
        return [k.strip() for k in env_keys.split(",")]

    raise ValueError("❌ No Groq API keys found. Use st.secrets or the GROQ_API_KEYS environment variable.")

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

# ---- Internal: Call LLM with one key ----
def try_call_llm(prompt, api_key, model, temperature):
    llm = ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    return llm.invoke(prompt).content

# ---- Main LLM Call with Dual Key Logic ----
def call_llm(prompt: str, session, model="llama-3.3-70b-versatile", temperature=0):
    # ✅ Cache check
    cached = get_cached_response(prompt)
    if cached:
        return cached

    # ✅ Safe handling of user key
    user_key = session.get("user_groq_key")
    user_key = user_key.strip() if isinstance(user_key, str) else ""

    admin_keys = load_groq_api_keys()
    admin_keys = [k for k in admin_keys if k]
    last_error = None

    # 1️⃣ Try user key (first)
    if user_key:
        try:
            print("🔑 Trying user API key (1st attempt)")
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, response)
            return response
        except Exception as e:
            print(f"❌ User API key failed (1st): {e}")
            last_error = e

    # 2️⃣ Try rotating admin keys
    if admin_keys:
        start_idx = session.get("key_index", 0)
        for i in range(len(admin_keys)):
            idx = (start_idx + i) % len(admin_keys)
            key = admin_keys[idx]
            try:
                print(f"🔁 Trying admin API key {idx + 1} of {len(admin_keys)}")
                response = try_call_llm(prompt, key, model, temperature)
                session["key_index"] = (idx + 1) % len(admin_keys)
                set_cached_response(prompt, response)
                return response
            except Exception as e:
                print(f"❌ Admin key {idx + 1} failed: {e}")
                last_error = e

    # 3️⃣ Retry user key again
    if user_key:
        try:
            print("🔁 Retrying user API key (2nd attempt)")
            response = try_call_llm(prompt, user_key, model, temperature)
            set_cached_response(prompt, response)
            return response
        except Exception as e:
            print(f"❌ User API key failed (2nd): {e}")
            last_error = e

    # ❌ Final failure
    raise RuntimeError(f"❌ All Groq API keys failed. Last error: {last_error}")
