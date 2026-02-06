# 🍷 Decant Quick Reference

One-page cheat sheet for common tasks.

---

## 🚀 Starting the App

```bash
source venv/bin/activate
streamlit run app.py
```

Browser opens at: `http://localhost:8501`

---

## 📥 Adding Wines

### Method 1: Streamlit App (Easiest) ⭐

1. Open app
2. Tab 1: Paste notes OR Tab 2: Upload bottle photo
3. Click "🔍 Evaluate Wine"
4. Fill "📔 Log This Bottle" form
5. Click "💾 Save to History"

### Method 2: Bulk CSV Edit

```bash
# 1. Edit data/history.csv (add rows)
# 2. Run feature extraction
python scripts/extract_features.py
```

---

## 🗑️ Removing Wines

### Quick Remove

```bash
# List all wines
python scripts/manage_wines.py --list

# Remove wine by row number
python scripts/manage_wines.py --remove 3

# Remove by name
python scripts/manage_wines.py --remove-name "Rombauer"
```

### Manual Remove

```bash
# 1. Edit data/history.csv (delete rows)
# 2. Regenerate features
python scripts/extract_features.py
```

---

## 📊 Managing Data

```bash
# List all wines
python scripts/manage_wines.py --list

# Show CSV template
python scripts/manage_wines.py --template

# Regenerate features
python scripts/manage_wines.py --regenerate

# Migrate JSON to CSV
python scripts/migrate_to_csv.py
```

---

## 🧪 CLI Wine Evaluation

```bash
python scripts/recommend.py "Your tasting notes here"
```

Output: Match score + recommendation + feature analysis

---

## 📓 Jupyter Notebooks

```bash
jupyter notebook

# Then open:
# - notebooks/01_palate_exploration.ipynb
# - notebooks/02_interactive_inference.ipynb
```

---

## 🔧 Troubleshooting

### App won't start
```bash
# Check API key
cat .env | grep OPENAI_API_KEY

# Reinstall dependencies
pip install -r requirements.txt
```

### "No wine data found"
```bash
# Run migration
python scripts/migrate_to_csv.py

# Run feature extraction
python scripts/extract_features.py
```

### Sidebar not updating
```bash
# Restart Streamlit
# Ctrl+C to stop, then:
streamlit run app.py
```

### Price column error
Check CSV has either `price`, `price_usd`, or `price_eur` column

---

## 📁 File Structure

```
data/
├── history.json          # Source data (JSON)
├── history.csv           # Flat wine list (CSV)
└── processed/
    └── wine_features.csv # AI-extracted features

scripts/
├── migrate_to_csv.py     # JSON → CSV
├── extract_features.py   # Extract features with AI
├── manage_wines.py       # Add/remove wines
└── recommend.py          # CLI recommender

src/decant/
├── predictor.py          # ICL prediction engine
└── schema.py             # Pydantic models

app.py                    # Streamlit web app
```

---

## 🎯 Common Workflows

### Add & Evaluate New Wine
```
1. streamlit run app.py
2. Upload bottle photo
3. Click "Analyze Image"
4. Click "Evaluate Wine"
5. Review match score
6. Fill "Log This Bottle" form
7. Save to history
```

### Clean Up Database
```
1. python scripts/manage_wines.py --list
2. python scripts/manage_wines.py --remove 3,5,7
3. Restart Streamlit app
```

### Bulk Import
```
1. Edit data/history.csv (add multiple rows)
2. python scripts/extract_features.py
3. streamlit run app.py
```

### Export for Analysis
```
# CSV files ready for:
- Excel/Google Sheets
- Pandas/Jupyter
- Data visualization tools
```

---

## 🔑 Environment Variables

Required in `.env` file:
```bash
OPENAI_API_KEY=sk-proj-...
```

Get your API key: https://platform.openai.com/api-keys

---

## 💡 Pro Tips

- **Use bottle photos** - Vision API works great for label extraction
- **Log as you taste** - Don't wait, use Streamlit form immediately
- **Check sidebar** - Always shows your palate fingerprint
- **Backup before bulk changes** - `manage_wines.py` does this automatically
- **Price tracking** - Add wines you haven't bought yet, mark liked=False, update later
- **Export CSVs** - Share with friends or import into other tools

---

## 🆘 Help

```bash
# Script help
python scripts/manage_wines.py --help
python scripts/recommend.py --help

# View documentation
cat WINE_MANAGEMENT.md
cat RUN_APP.md
cat SECURITY.md
```

---

## 📞 Quick Commands Summary

| Task | Command |
|------|---------|
| **Start app** | `streamlit run app.py` |
| **List wines** | `python scripts/manage_wines.py --list` |
| **Remove wine** | `python scripts/manage_wines.py --remove 3` |
| **Add wines (bulk)** | Edit `data/history.csv` → `python scripts/extract_features.py` |
| **Evaluate wine** | `python scripts/recommend.py "notes"` |
| **Regenerate features** | `python scripts/extract_features.py` |
| **Show template** | `python scripts/manage_wines.py --template` |
| **Start Jupyter** | `jupyter notebook` |

---

**Remember:** `data/history.csv` is your source of truth. Always edit that file, then regenerate features.

**Pro workflow:** Use Streamlit app for everything. It handles all the complexity automatically! 🍷✨
