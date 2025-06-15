import streamlit as st
import sqlite3
from datetime import datetime

# --- DB Setup ---
def create_confession_table():
    conn = sqlite3.connect("resume_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS confessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            confession TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_confession(username, confession):
    conn = sqlite3.connect("resume_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO confessions (username, confession) VALUES (?, ?)", (username, confession))
    conn.commit()
    conn.close()

def get_all_confessions():
    conn = sqlite3.connect("resume_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT username, confession, timestamp FROM confessions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

# --- Initialization ---
create_confession_table()

# --- Confession UI ---
st.title("💌 Anonymous Confessions")

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login to post or view confessions.")
    st.stop()

# --- Confession Submission ---
with st.form("confess_form"):
    confession = st.text_area("Write your confession anonymously here...", max_chars=1000, height=150)
    submit = st.form_submit_button("Submit Confession")

if submit and confession.strip():
    add_confession(st.session_state.username, confession.strip())
    st.success("✅ Your confession has been posted anonymously.")

# --- Display Confessions ---
st.markdown("## 🔍 Recent Confessions")
for _, confession_text, time in get_all_confessions():
    st.markdown(f"🕊️ **Anonymous** confessed:  \n> *{confession_text}*  \n<small style='color:gray;'>🕒 {time}</small>", unsafe_allow_html=True)

# --- Admin View ---
if st.session_state.username == "admin":
    st.markdown("---")
    st.markdown("## 🔐 Admin View (Real Names)")
    for user, confession_text, time in get_all_confessions():
        st.markdown(f"👤 **{user}** confessed:  \n> *{confession_text}*  \n<small style='color:gray;'>🕒 {time}</small>", unsafe_allow_html=True)
