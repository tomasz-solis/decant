# Get Supabase API Keys

You need to add your Supabase API keys to enable photo storage.

## Step 1: Get Your Keys

1. Go to: https://supabase.com/dashboard/project/rageghyliafgxublfljb/settings/api

2. Copy these two values:
   - **Project URL** (e.g., `https://rageghyliafgxublfljb.supabase.co`)
   - **anon/public key** (starts with `eyJ...`)

## Step 2: Add to .streamlit/secrets.toml

Add these lines to your `.streamlit/secrets.toml`:

```toml
# Supabase Storage (for wine images)
SUPABASE_URL = "https://rageghyliafgxublfljb.supabase.co"
SUPABASE_KEY = "paste-your-anon-key-here"
```

## Step 3: Test the Connection

```bash
python scripts/migrate_images_to_supabase.py
```

This will:
- Create the 'wine-images' bucket in Supabase Storage
- Upload all 15 existing wine photos
- Give you public URLs for each photo

---

**Don't commit these keys to git!** The `.streamlit/secrets.toml` file is already in `.gitignore`.
