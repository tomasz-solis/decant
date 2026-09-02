#!/usr/bin/env python3
"""Reset a Decant user's Supabase password from the command line.

The app has no in-app reset flow (see `src/decant/ui/auth_form.py`), so
this script talks to the Supabase Admin API directly.

Usage:
    export SUPABASE_SERVICE_ROLE_KEY="eyJ..."     # never commit this
    uv run scripts/reset_password.py user@example.com
    uv run scripts/reset_password.py user@example.com --password 'my-new-pass'

Without --password a random one is generated and printed. `SUPABASE_URL`
is read from `.streamlit/secrets.toml`, or from the env var of the same
name if the file is missing.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import tomllib
from pathlib import Path

import requests

SECRETS_PATH = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"


def supabase_url() -> str:
    """Return the project URL from secrets.toml, falling back to the env var."""
    if SECRETS_PATH.exists():
        with SECRETS_PATH.open("rb") as fh:
            url = tomllib.load(fh).get("SUPABASE_URL")
        if url:
            return str(url).strip().strip('"').rstrip("/")
    url = os.environ.get("SUPABASE_URL")
    if not url:
        sys.exit(f"SUPABASE_URL not found in {SECRETS_PATH} or the environment.")
    return url.strip().rstrip("/")


def find_user_id(url: str, headers: dict[str, str], email: str) -> str:
    """Return the Supabase user UUID for `email`, or exit if there is no match."""
    # ponytail: single unpaginated page. This is a two-person household app;
    # add page walking if the user list ever exceeds 1000.
    resp = requests.get(
        f"{url}/auth/v1/admin/users",
        headers=headers,
        params={"per_page": 1000},
        timeout=30,
    )
    resp.raise_for_status()
    users = resp.json().get("users", [])
    for user in users:
        if (user.get("email") or "").lower() == email.lower():
            return user["id"]
    known = ", ".join(sorted(u.get("email", "?") for u in users)) or "(none)"
    sys.exit(f"No user with email {email}. Known users: {known}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset a Decant user's password.")
    parser.add_argument("email", help="email address of the account to reset")
    parser.add_argument(
        "--password",
        help="new password; a random 16-character one is generated if omitted",
    )
    args = parser.parse_args()

    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not service_key:
        sys.exit(
            "SUPABASE_SERVICE_ROLE_KEY is not set. Copy the service_role key "
            "from Supabase > Project Settings > API and export it for this "
            "shell only - do not put it in secrets.toml."
        )

    new_password = args.password or secrets.token_urlsafe(12)
    url = supabase_url()
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}

    user_id = find_user_id(url, headers, args.email)
    resp = requests.put(
        f"{url}/auth/v1/admin/users/{user_id}",
        headers=headers,
        json={"password": new_password},
        timeout=30,
    )
    if not resp.ok:
        sys.exit(f"Reset failed ({resp.status_code}): {resp.text}")

    print(f"Password reset for {args.email} (id {user_id})")
    if not args.password:
        print(f"New password: {new_password}")


if __name__ == "__main__":
    main()
