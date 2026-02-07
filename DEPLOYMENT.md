# 🚀 Streamlit Cloud Deployment Guide

> Documentation created with Claude AI assistance

This guide will walk you through deploying Decant to Streamlit Cloud with PostgreSQL persistence.

## ✅ Pre-Deployment Checklist

All these steps have been completed for you:

- [x] `app.py` - Main application entry point with PostgreSQL integration
- [x] `requirements.txt` - All Python dependencies including psycopg[binary]
- [x] `packages.txt` - System dependencies for Streamlit Cloud
- [x] `.gitignore` - Configured to exclude secrets and sensitive files
- [x] PostgreSQL database - 15 wines migrated to Supabase
- [x] Supabase Storage - 15 wine photos uploaded to cloud storage
- [x] Authentication - Password protection with streamlit-authenticator
- [x] API key handling - Uses `st.secrets` for Streamlit Cloud
- [x] Graceful degradation - Falls back to CSV if database unavailable

## 🛠️ Step 1: Prepare Your Repository

### Files Ready for Deployment:

1. **app.py** - Your main Streamlit application
2. **requirements.txt** - All dependencies (streamlit, openai, pandas, plotly, etc.)
3. **.gitignore** - Protects secrets:
   - `.env` (local development)
   - `.streamlit/secrets.toml` (local secrets)
   - Most CSV files (except essential ones for app functionality)

### What's Been Updated:

The app now uses **dual authentication** for flexibility:
- **Streamlit Cloud**: Reads from `st.secrets["OPENAI_API_KEY"]`
- **Local Development**: Falls back to `.env` file

```python
# Code snippet from app.py
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.getenv("OPENAI_API_KEY")
```

## 🔐 Step 2: Set Up Secrets in Streamlit Cloud

**IMPORTANT:** Never commit your actual API keys or database credentials to GitHub!

### In Streamlit Cloud:

1. Go to https://share.streamlit.io/
2. Deploy your app (connect to GitHub repo)
3. Once deployed, go to: **Your App → ⚙️ Settings → Secrets**
4. Add your secrets in TOML format:

```toml
# Supabase PostgreSQL Database (Session Pooler - IPv4 compatible)
DATABASE_URL = "postgresql://postgres.rageghyliafgxublfljb:rfq9eam2qpy.VDG6mwq@aws-1-eu-central-1.pooler.supabase.com:5432/postgres"

# Supabase Storage (for wine images)
SUPABASE_URL = "https://rageghyliafgxublfljb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2VnaHlsaWFmZ3h1YmxmbGpiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA0MDU5NTIsImV4cCI6MjA4NTk4MTk1Mn0.gZCO0j8XXW3AX-im8khx6YSotcfBAnr6gpIdBAesPrM"

# OpenAI API Key
OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"

# Authentication
[passwords]
# CHANGE THIS PASSWORD AFTER FIRST LOGIN!
# Generate new hash: python -m decant.auth YOUR_NEW_PASSWORD
# Default: username="admin", password="wine123"
admin = "$2b$12$PtPlOBLFwQEPoXR6/V2To.sJRnrnpOi2BSjlwuR7pkD.jE6WJueUC"

[cookie]
name = "decant_auth"
key = "decant_secure_signature_key_8x7b2m9q"  # Random key for session cookies
expiry_days = 30
```

5. Click **Save**
6. The app will automatically restart with the new secrets

### Reference File:

See `.streamlit/secrets.toml` (excluded from git) for the exact structure.

## 📦 Step 3: Initialize Git Repository (If Not Already Done)

```bash
# Initialize git (if needed)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Decant app ready for Streamlit Cloud"

# Create GitHub repository and push
gh repo create decant --public --source=. --remote=origin --push
# OR manually:
# git remote add origin https://github.com/YOUR-USERNAME/decant.git
# git branch -M main
# git push -u origin main
```

## 🌐 Step 4: Deploy to Streamlit Cloud

1. **Go to https://share.streamlit.io/**
2. Click **"New app"**
3. **Connect your GitHub repository:**
   - Repository: `your-username/decant`
   - Branch: `main` (or `master`)
   - Main file path: `app.py`
4. Click **"Deploy"**
5. Wait 2-3 minutes for initial build
6. Your app will be live at: `https://your-app-name.streamlit.app`

## 🔧 Step 5: Configure Secrets (Critical!)

Immediately after deployment:

1. Click **⚙️ Settings** in your Streamlit Cloud dashboard
2. Navigate to **Secrets**
3. Paste your OpenAI API key:

```toml
OPENAI_API_KEY = "sk-proj-..."
```

4. Click **Save**
5. App will restart automatically

## 🧪 Step 6: Test Your Deployed App

Test these features:

- [ ] App loads without errors
- [ ] Wine image upload works
- [ ] OpenAI Vision API extracts wine data from images
- [ ] Wine Gallery displays existing wines
- [ ] Palate Maps render correctly
- [ ] Score/price updates work
- [ ] Liked/disliked toggle works

## 📊 Data Persistence

### PostgreSQL Database (Supabase) ✅

**Good news**: Your app now uses PostgreSQL for persistent storage!

- **Database**: Supabase (rageghyliafgxublfljb.supabase.co)
- **Region**: EU (eu-central-1)
- **Current Data**: 15 wines migrated successfully
- **Connection**: Session Pooler (IPv4 compatible)

### How It Works:

1. **New wines** saved through the app go directly to PostgreSQL
2. **Changes persist** across app restarts and deployments
3. **No git commits needed** for data updates
4. **CSV fallback** available if database is unavailable

### Database Backup (Optional):

To backup your PostgreSQL data to CSV:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from decant.database import get_all_wines
df = get_all_wines()
df.to_csv('data/backup.csv', index=False)
print(f'Backed up {len(df)} wines')
"
```

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution:** Go to Settings → Secrets in Streamlit Cloud and add your key

### Issue: "Database unavailable, falling back to CSV"
**Solution:**
- Check DATABASE_URL is correctly set in Streamlit Cloud secrets
- Verify Supabase project is active (not paused)
- Ensure using Session Pooler: `aws-1-eu-central-1.pooler.supabase.com:5432`
- Check Supabase project status at https://supabase.com/dashboard

### Issue: "Tenant or user not found" (database error)
**Solution:**
- Verify username format: `postgres.rageghyliafgxublfljb`
- Check password is correct in DATABASE_URL
- Ensure using correct pooler endpoint (`aws-1-` not `aws-0-`)

### Issue: "No module named 'psycopg'"
**Solution:** Ensure `requirements.txt` includes `psycopg[binary]==3.3.2`

### Issue: App is slow/timing out
**Solution:**
- Check OpenAI API rate limits
- Consider caching with `@st.cache_data`
- Reduce image size before upload
- Monitor Supabase connection pooling limits

## 🔄 Updating Your Deployed App

Every time you push to GitHub, Streamlit Cloud will automatically rebuild:

```bash
# Make changes locally
git add .
git commit -m "Your update message"
git push

# Streamlit Cloud will auto-detect and redeploy
```

## 🎉 You're Live!

Your app is now accessible to anyone with the link. Share it with wine enthusiasts!

**Example URL:** `https://decant-yourname.streamlit.app`

## 🔐 Authentication Setup

**Status:** ✅ Authentication is now configured!

The app requires login before access:

- **Default username:** `admin`
- **Default password:** `wine123`
- **⚠️ IMPORTANT:** Change the password immediately after first login!

### Changing the Password:

```bash
# Generate a new password hash
python -m decant.auth YOUR_NEW_PASSWORD

# Copy the hash output, then update .streamlit/secrets.toml:
[passwords]
admin = "PASTE_NEW_HASH_HERE"
```

### Adding More Users:

```toml
[passwords]
admin = "$2b$12$..."      # First user
partner = "$2b$12$..."    # Second user (generate hash first)
```

**Note:** All users see the same wines - this provides password protection but does NOT separate data by user.

## 📸 Supabase Storage Setup

**Status:** ✅ All 15 wine photos migrated successfully!

- **Bucket:** `wine-images` (public, 2MB file size limit)
- **Location:** `wines/` folder
- **RLS Policies:** 4 policies configured (SELECT, INSERT, UPDATE, DELETE)
- **Access:** Anonymous (anon) role can read/write images

### Photo Management:

Photos are automatically uploaded to Supabase when you add a wine through the app. The app uses these functions:

- `upload_wine_image()` - Upload new photos
- `get_wine_image_url()` - Retrieve public URLs
- `delete_wine_image()` - Remove photos

All images have sanitized filenames (Unicode → ASCII conversion) for compatibility.

## 📝 Next Steps (Optional)

- [ ] Add custom domain (Streamlit Cloud paid tier)
- [ ] Set up GitHub Actions for automated testing
- [x] Add user authentication (streamlit-authenticator) ✅ DONE
- [ ] Implement multi-user support with user profiles
- [ ] Add wine recommendation export (PDF/CSV)

---

## 🆘 Need Help?

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- Streamlit Forum: https://discuss.streamlit.io/
- OpenAI API Docs: https://platform.openai.com/docs

---

**Built with Claude AI** - This documentation and deployment setup were created with assistance from [Claude Code](https://claude.ai/code)
