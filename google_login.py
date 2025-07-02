import streamlit as st
from authlib.integrations.requests_client import OAuth2Session
import os

# ✅ Load credentials securely
GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", os.getenv("GOOGLE_CLIENT_ID"))
GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET", os.getenv("GOOGLE_CLIENT_SECRET"))
REDIRECT_URI = st.secrets.get("REDIRECT_URI", os.getenv("REDIRECT_URI"))

AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def fetch_token_from_url():
    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        code = query_params["code"][0]
        oauth = OAuth2Session(
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="openid email profile"
        )
        try:
            token = oauth.fetch_token(
                TOKEN_URL,
                code=code,
                redirect_uri=REDIRECT_URI,
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                include_client_id=True
            )
            if token and "access_token" in token:
                st.session_state.google_token = token
                st.write("✅ Token fetched:", token)  # TEMP debug
            else:
                st.error("🚫 Token fetched but missing access_token.")
                st.stop()
        except Exception as e:
            st.error(f"🚫 Failed to fetch token. Error: {e}")
            st.session_state.pop("google_token", None)
            st.stop()


def login_via_google():
    if "google_token" not in st.session_state:
        oauth = OAuth2Session(
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="openid email profile"
        )
        auth_url, state = oauth.create_authorization_url(
            AUTH_URL,
            redirect_uri=REDIRECT_URI
        )
        st.session_state.oauth_state = state
        st.markdown(f"[🔐 Login with Google]({auth_url})", unsafe_allow_html=True)
        st.stop()
    else:
        token = st.session_state.get("google_token")
        if not token or not isinstance(token, dict) or "access_token" not in token:
            st.error("⚠️ Invalid or missing token. Please try logging in again.")
            st.session_state.pop("google_token", None)
            st.stop()

        # ✅ FIX: Attach token directly into the session
        oauth = OAuth2Session(
            client_id=GOOGLE_CLIENT_ID,
            token=token  # ✅ Correct method
        )

        try:
            resp = oauth.get(USERINFO_URL)
            user_data = resp.json()
            return {
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "picture": user_data.get("picture")
            }
        except Exception as e:
            st.error(f"🚫 Failed to get user info. Error: {e}")
            st.session_state.pop("google_token", None)
            st.stop()

