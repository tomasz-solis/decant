import streamlit as st
from supabase import Client, create_client


def _normalize_secret_string(raw_value: object, secret_name: str) -> str:
    """Normalize and validate secret strings from Streamlit secrets."""
    if raw_value is None:
        raise ValueError(f"{secret_name} is missing")

    value = str(raw_value).strip()
    quote_pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    ]
    for left_quote, right_quote in quote_pairs:
        if value.startswith(left_quote) and value.endswith(right_quote) and len(value) >= 2:
            value = value[1:-1].strip()
            break

    if not value:
        raise ValueError(f"{secret_name} is empty")
    return value


def _get_supabase_credentials(username: str) -> tuple[str, str]:
    """Map Streamlit username to Supabase email/password from secrets."""
    users = st.secrets["supabase_users"]

    credential_map = {
        "tomasz": ("tomasz_email", "tomasz_password"),
        "karolina": ("karolina_email", "karolina_password"),
    }

    if username not in credential_map:
        raise ValueError(f"Unknown Streamlit user '{username}' for Supabase auth")

    email_key, password_key = credential_map[username]
    return (
        _normalize_secret_string(users[email_key], f"supabase_users.{email_key}"),
        _normalize_secret_string(users[password_key], f"supabase_users.{password_key}"),
    )


def get_supabase_client() -> Client:
    """
    Return a Supabase client authenticated as the current Streamlit user.

    Assumes Streamlit authentication already ran and `st.session_state["username"]` exists.
    """
    username = st.session_state.get("username")
    if not username:
        raise RuntimeError("Streamlit auth must run before Supabase initialization")

    cached_client = st.session_state.get("supabase_client")
    cached_username = st.session_state.get("supabase_client_username")
    if cached_client is not None and cached_username == username:
        return cached_client

    supabase_url = _normalize_secret_string(st.secrets["SUPABASE_URL"], "SUPABASE_URL")
    supabase_key = _normalize_secret_string(st.secrets["SUPABASE_KEY"], "SUPABASE_KEY")
    sb = create_client(supabase_url, supabase_key)
    email, password = _get_supabase_credentials(username)

    auth_response = sb.auth.sign_in_with_password(
        {"email": email, "password": password}
    )
    session = getattr(auth_response, "session", None)
    if not session or not session.access_token or not session.refresh_token:
        raise RuntimeError("Supabase password login did not return a valid session")

    sb.auth.set_session(session.access_token, session.refresh_token)

    st.session_state["supabase_client"] = sb
    st.session_state["supabase_client_username"] = username
    return sb


def get_user_supabase() -> Client:
    """Backward-compatible alias."""
    return get_supabase_client()
