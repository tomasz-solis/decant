# Get Supabase API Keys

You need the Supabase project URL and anon key to run the app locally — the same values that go into `.streamlit/secrets.toml`.

## Step 1: Find the values in the Supabase dashboard

1. Open your project at https://supabase.com/dashboard
2. Click **Project Settings** → **API**
3. Copy:
   - **Project URL** — looks like `https://YOURPROJECT.supabase.co`
   - **anon / public key** — a long JWT starting with `eyJ...`

The `service_role` key on the same page is different — don't use it for the app. It bypasses Row-Level Security and should only be used in trusted admin scripts.

## Step 2: Add to `.streamlit/secrets.toml`

```toml
SUPABASE_URL = "https://YOURPROJECT.supabase.co"
SUPABASE_KEY = "eyJ..."
```

The full set of secrets the app expects is in the [README](../README.md#setup).

## Step 3: Verify

```bash
uv run streamlit run app.py
```

The app should boot without "missing required secret" errors. If you see RLS failures (the gallery shows nothing for a signed-in user, for example), the keys are loading but the `CELLAR_ID` may not match an actual row — check `sql/` for the RLS policies and the wines table schema.

---

**Don't commit secrets to git.** `.streamlit/secrets.toml` should be in `.gitignore` already; double-check before adding the file.
