# 🚀 Streamlit Cloud Deployment Guide

> Documentation created with Claude AI assistance

This guide will walk you through deploying Decant to Streamlit Cloud.

## ✅ Pre-Deployment Checklist

All these steps have been completed for you:

- [x] `app.py` - Main application entry point
- [x] `requirements.txt` - All Python dependencies listed
- [x] `.gitignore` - Configured to exclude secrets and sensitive files
- [x] API key handling - Uses `st.secrets` for Streamlit Cloud
- [x] Essential data files - Allowed in git for deployment

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

## 🔐 Step 2: Set Up OpenAI API Key in Streamlit Cloud

**IMPORTANT:** Never commit your actual API key to GitHub!

### In Streamlit Cloud:

1. Go to https://share.streamlit.io/
2. Deploy your app (connect to GitHub repo)
3. Once deployed, go to: **Your App → ⚙️ Settings → Secrets**
4. Add your secret in TOML format:

```toml
OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"
```

5. Click **Save**
6. The app will automatically restart with the new secret

### Reference File:

See `.streamlit/secrets.toml.example` for the exact structure.

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

### Important Notes:

- **Local changes will NOT sync to Streamlit Cloud automatically**
- The deployed app starts with the data committed to GitHub
- Each time you add wines locally, you need to commit and push to update cloud:

```bash
git add data/history.csv data/processed/wine_features.csv
git commit -m "Update wine collection"
git push
```

- Streamlit Cloud will auto-redeploy on push

### Alternative: External Database (Optional)

For true persistence without git commits, consider:
- Streamlit Cloud connection to PostgreSQL/MySQL
- Google Sheets integration
- Airtable API
- AWS S3 bucket

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution:** Go to Settings → Secrets in Streamlit Cloud and add your key

### Issue: "No module named 'decant'"
**Solution:** Ensure `requirements.txt` includes `-e .` to install local package

### Issue: "File not found: data/history.csv"
**Solution:** Make sure the file is committed to GitHub (check `.gitignore` exceptions)

### Issue: App is slow/timing out
**Solution:**
- Check OpenAI API rate limits
- Consider caching with `@st.cache_data`
- Reduce image size before upload

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

## 📝 Next Steps (Optional)

- [ ] Add custom domain (Streamlit Cloud paid tier)
- [ ] Set up GitHub Actions for automated testing
- [ ] Add user authentication (streamlit-authenticator)
- [ ] Implement multi-user support with user profiles
- [ ] Add wine recommendation export (PDF/CSV)

---

## 🆘 Need Help?

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- Streamlit Forum: https://discuss.streamlit.io/
- OpenAI API Docs: https://platform.openai.com/docs

---

**Built with Claude AI** - This documentation and deployment setup were created with assistance from [Claude Code](https://claude.ai/code)
