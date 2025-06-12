import streamlit as st
from user_login import get_all_user_logs

st.set_page_config(page_title="Admin Logs", layout="wide")

st.markdown("<h2 style='color:#00BFFF;'>📊 User Activity Logs</h2>", unsafe_allow_html=True)

logs = get_all_user_logs()

if logs:
    st.write(f"Total Logs: {len(logs)}")
    st.dataframe(
        {
            "Username": [log[0] for log in logs],
            "Action": [log[1] for log in logs],
            "Timestamp": [log[2] for log in logs]
        },
        use_container_width=True
    )
else:
    st.warning("No user logs found.")
