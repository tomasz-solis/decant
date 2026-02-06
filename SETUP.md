# Decant Setup Guide

> **Production-ready wine recommendation app** • Score: 85/100 • Built with AI assistance

## Table of Contents

⏱️ **Estimated Setup Time: 10-15 minutes**

- [System Requirements](#system-requirements) - Prerequisites
- [Quick Start](#quick-start-5-minutes) - ⏱️ 5 minutes
- [Running the App](#running-the-app) - ⏱️ 2 minutes
- [Initial Data Setup](#initial-data-setup) - ⏱️ 5 minutes (first wine)
- [API Rate Limits & Cost Protection](#api-rate-limits--cost-protection) - 📖 Important info
- [Development Workflow](#development-workflow) - ⏱️ Ongoing
- [Project Structure](#project-structure) - 📖 Reference
- [Troubleshooting](#troubleshooting) - 📖 Reference guide
- [Deployment](#deployment) - ⏱️ 15 minutes (Streamlit Cloud)
- [Advanced Configuration](#advanced-configuration) - 📖 Optional
- [Development Patterns](#development-patterns-important) - ⚠️ Must read for contributors

---

## System Requirements

- **Python**: 3.9+ (3.10+ recommended)
- **OS**: macOS/Linux (primary), Windows (supported)
- **RAM**: 2GB minimum
- **Storage**: 500MB for dependencies + data
- **Internet**: Required for OpenAI API calls

## Quick Start (5 minutes)

### 1. Clone and Navigate

```bash
cd /path/to/decant
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Verify activation (should show venv path)
which python3
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install decant package in editable mode
pip install -e .
```

**Key Dependencies**:
- **streamlit** (2.0+): Web UI framework (main app)
- **openai** (1.0+): GPT-5.2 API client for feature extraction
- **pandas** (2.0+): Data manipulation and CSV operations
- **pydantic** (2.0+): Data validation and type safety
- **scipy** (1.11+): Statistical tests (notebook 03)
- **tenacity** (8.0+): Retry logic with exponential backoff
- **python-dotenv** (1.0+): Environment variable loading
- **plotly** (5.0+): Interactive visualizations
- **pillow** (10.0+): Image processing

### 4. Set Up OpenAI API Key

```bash
# Copy the template
cp .env.template .env

# Edit .env and add your actual API key
nano .env  # or use your preferred editor
```

**Your .env file should look like this**:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-your-actual-key-here

# Optional: Override default model (default: gpt-5.2-2025-12-11)
# OPENAI_MODEL=gpt-5.2-2025-12-11

# Optional: Custom temperature (default: 0.0 for determinism)
# OPENAI_TEMPERATURE=0.0

# Optional: Custom seed for reproducibility (default: 42)
# OPENAI_SEED=42
```

**Get your API key**: https://platform.openai.com/api-keys

**⚠️ CRITICAL SECURITY WARNINGS**:
1. **Never commit** `.env` to git (already in `.gitignore`)
2. **Never share** your API key publicly or in screenshots
3. **Rotate keys** immediately if accidentally exposed at platform.openai.com
4. **Streamlit Cloud free tier** = PUBLIC ACCESS
   - ⚠️ Don't use for sensitive wine data or private tasting notes
   - ⚠️ Anyone with the URL can access your app and see your wines
   - ⚠️ Consider private deployment or authentication for sensitive data
5. **Rate limiter** protects against accidental costs, NOT malicious users
   - No per-user quotas in current implementation
   - Multiple users share the same rate limit pool
   - Monitor usage at platform.openai.com to catch unexpected spikes

**💰 Cost Expectations**:
- **Model**: GPT-5.2 (latest, 2025-12-11)
- **Pricing**: ~$0.005/1K input tokens, ~$0.015/1K output tokens
- **Per wine extraction**: ~$0.01-0.05
- **Monthly (personal use)**: $2-10 for ~50-200 wine additions
- **Rate limits enforced**: 20 req/min, 500 req/hour, $5/hour (configurable in rate_limiter.py)

### 5. Verify Installation

```bash
# Test package imports
python3 -c "from decant.constants import WineColor, Verdict; print('✓ Decant installed successfully')"

# Test schema imports
python3 -c "from decant.schema import WineFeatures, WineExtraction; print('✓ Schemas loaded')"

# Test palate engine
python3 -c "from decant.palate_engine import PalateEngine; print('✓ Palate engine ready')"

# Run all tests (45 tests should pass)
python3 run_tests.py

# Expected output: 45 passed in <1s, 37% coverage
```

**If verification fails**, check:
1. Virtual environment is activated (`which python3` shows venv path)
2. Package is installed (`pip show decant`)
3. Dependencies are installed (`pip list | grep streamlit`)

---

## How Decant Works

**Data Flow Diagram**:
```
┌─────────────────┐
│  📸 Photo      │  User uploads wine bottle photo
│  Upload        │  OR enters details manually
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  🤖 GPT-5.2    │  OpenAI Vision API extracts:
│  Vision API    │  • Name, producer, vintage
│  (OpenAI)      │  • 5D flavor features (acidity,
└────────┬────────┘    minerality, fruitiness, tannin, body)
         │           • Wine metadata (color, region, etc.)
         ▼
┌─────────────────┐
│  ✅ Pydantic   │  Validates LLM response:
│  Validation    │  • Check all required fields present
│  (schema.py)   │  • Validate ranges (1-10 for features)
└────────┬────────┘  • Validate enums (wine colors, sweetness)
         │           • Prevent KeyError crashes
         ▼
┌─────────────────┐
│  🛡️ Rate       │  Check API limits BEFORE calling:
│  Limiter       │  • 20 requests/minute
│  (rate_limiter)│  • 500 requests/hour
└────────┬────────┘  • $5/hour cost limit
         │           • Track costs per request
         ▼
┌─────────────────┐
│  💾 Save CSV   │  Append to data/history.csv:
│  (data/)       │  • All wine details + ratings
│  history.csv   │  • Liked/disliked flag
└────────┬────────┘  • Flavor profile (5D vector)
         │
         ▼
┌─────────────────┐
│  📊 Palate     │  Calculate match scores:
│  Engine        │  • Cosine similarity in 5D space
│  (palate_      │  • Exponential confidence decay
│   engine.py)   │  • Color-specific matching
└────────┬────────┘  • Dual-metric scoring (palate + confidence)
         │
         ▼
┌─────────────────┐
│  🎯 Display    │  Show in Streamlit UI:
│  Match Score   │  • Palate match % (raw similarity)
│  (app.py)      │  • Likelihood % (confidence-adjusted)
└─────────────────┘  • Verdict (💙/🧡/🟡/⚪)
                     • Radar chart visualization
```

**Key Algorithm**: Palate Formula (SINGLE SOURCE OF TRUTH in `palate_formula.py`)
```
structure_score = acidity + minerality
acidity_body_ratio = acidity / (body + 0.1)
palate_score = structure_score + (acidity_body_ratio × 2)

confidence = 1 - e^(-0.4 × N)  where N = number of wines rated
likelihood_score = palate_match × confidence
```

---

## Running the App

### Launch Streamlit Web App

```bash
# Activate environment (if not already)
source venv/bin/activate

# Launch app
streamlit run app.py

# App opens automatically at http://localhost:8501
```

**⚠️ This is the PRIMARY way to use Decant** - not command-line scripts.

### First-Time Setup in App

On first launch, you'll see an empty wine gallery. To add wines:

1. **Navigate to "Add Wine" tab** in the sidebar
2. **Choose input method**:
   - **Photo Upload** (recommended): Upload wine bottle photo → GPT-5.2 extracts features
   - **Manual Entry**: Enter wine details manually
3. **Fill in tasting notes** and rate the wine (liked/disliked)
4. **Save** → Wine added to `data/history.csv`

### App Features

- **🍷 Wine Gallery**: Browse all wines with filtering (color, region, liked/disliked)
- **📊 Predictions**: Get palate match scores for new wines (after 3+ wines rated)
- **📈 Analytics**: View flavor profile radar charts
- **🔍 Search**: Find wines by name, producer, region
- **📸 Photo Extraction**: AI-powered feature extraction from bottle photos

## Initial Data Setup

### Data Structure

**Primary file**: `data/history.csv` (created automatically on first save)

**CSV Format** (12 required columns):
```csv
wine_name,producer,vintage,notes,score,liked,price,country,region,wine_color,is_sparkling,is_natural,sweetness,acidity,minerality,fruitiness,tannin,body
```

**Example row**:
```csv
"Fefiñanes Albariño 2022","Bodegas Fefiñanes",2022,"Crisp, mineral, citrus",8.5,True,18.99,"Spain","Rías Baixas","White",False,False,"Dry",8.5,8.0,7.5,2.0,5.5
```

### Data Initialization

**Option 1: Use the App** (recommended)
1. Launch app: `streamlit run app.py`
2. Add wines via UI (photo upload or manual entry)
3. Data automatically saved to `data/history.csv`

**Option 2: Import Existing Data**
If you have wine data in CSV format:
1. Ensure CSV has all 12 required columns
2. Copy to `data/history.csv`
3. Restart app to load data

**Option 3: Start from Scratch**
- No setup needed - file created automatically on first wine save
- App will show "No wines in your collection yet" initially

### Data Backup (IMPORTANT)

```bash
# Backup your wine data regularly
cp data/history.csv data/history_backup_$(date +%Y%m%d).csv

# For automated backups (macOS/Linux)
# Add to crontab: 0 0 * * 0 cd /path/to/decant && cp data/history.csv data/backups/history_$(date +\%Y\%m\%d).csv
```

## API Rate Limits & Cost Protection

Decant includes built-in rate limiting to prevent API cost overruns:

### Rate Limits (Configurable in `rate_limiter.py`)

- **20 requests/minute**
- **500 requests/hour**
- **$5/hour cost limit**

### What Happens When Limits Hit

**Error message**:
```
RateLimitError: API rate limit exceeded.
Current usage: 20/20 req/min, 142/500 req/hour, $0.84/$5.00/hour.
Please wait before retrying.
```

**Solutions**:
1. **Wait 60 seconds** - Limits reset in sliding windows
2. **Check current usage**:
   ```python
   from decant.predictor import VinoPredictor
   predictor = VinoPredictor()
   stats = predictor.rate_limiter.get_stats()
   print(stats)
   ```
3. **Adjust limits** (if needed) in `src/decant/rate_limiter.py`

### Cost Tracking

All API calls automatically track costs:
- Per-request cost calculation from token usage
- Hourly cost accumulation
- Warning at 80% of limits
- Hard stop at $5/hour (configurable)

## Development Workflow

### Making Code Changes

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Create feature branch (if using git)
git checkout -b feature/my-feature

# 3. Make code changes

# 4. Format code (do this FIRST)
black src/ tests/ scripts/

# 5. Run tests (verify nothing broke)
python3 run_tests.py

# 6. Check test coverage
pytest --cov=src/decant --cov-report=html
open htmlcov/index.html

# 7. Review CLAUDE_FIXES.md for patterns
cat CLAUDE_FIXES.md  # See what's been improved and why

# 8. Commit changes
git add .
git commit -m "Description of changes"
```

### Code Quality Standards

✅ **Required before committing**:
- All tests pass (45/45)
- No deprecation warnings
- Code formatted with black
- Use centralized constants (no magic strings)
- Use centralized palate formula (no duplication)
- Pydantic validation for all external inputs

### Running Tests

```bash
# Recommended: Use custom test runner
python3 run_tests.py

# OR: Use pytest directly
pytest

# With coverage report
pytest --cov=src/decant --cov-report=html

# Run specific test file
pytest tests/test_palate_engine.py

# Run specific test function
pytest tests/test_palate_engine.py::TestCosineSimilarity::test_identical_vectors_return_100_percent
```

**Test Suite**:
- 45 tests total (100% passing)
- Core algorithm tests (palate_engine, cosine similarity)
- Schema validation tests (Pydantic models)
- Edge case coverage
- 37% code coverage (focused on critical paths)

### Running Notebooks

```bash
# Start Jupyter Lab (recommended)
jupyter lab

# OR: Classic Jupyter Notebook
jupyter notebook

# Notebooks available:
# - 01_palate_exploration.ipynb: Flavor space analysis
# - 02_interactive_inference.ipynb: LLM prediction testing
# - 03_exponential_decay_analysis.ipynb: Statistical significance tests
```

**Notebook 03 requires scipy** for statistical tests (paired t-tests, confidence intervals).

## Project Structure

```
decant/
├── venv/                          # Virtual environment (not in git)
├── app.py                         # Main Streamlit application (1950+ lines)
│
├── src/decant/                    # Core Python package
│   ├── __init__.py
│   ├── constants.py               # Enums, constants, validation schemas (233 lines)
│   ├── palate_formula.py          # SINGLE SOURCE OF TRUTH for palate calculations (159 lines)
│   ├── palate_engine.py           # Matching algorithm with cosine similarity (455 lines)
│   ├── predictor.py               # LLM-based wine predictor with caching (160 lines)
│   ├── rate_limiter.py            # API rate limiting and cost tracking (324 lines)
│   ├── error_handling.py          # Standardized error handling (157 lines)
│   ├── schema.py                  # Pydantic data models (150 lines)
│   ├── utils.py                   # Utilities, prompt injection defense (143 lines)
│   └── config.py                  # Configuration constants (3 lines)
│
├── tests/                         # Test suite (45 tests, 100% passing)
│   ├── test_palate_engine.py     # Core algorithm tests (31 tests)
│   ├── test_schema.py             # Pydantic validation tests (14 tests)
│   └── run_tests.py               # Custom test runner
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_palate_exploration.ipynb          # Flavor space exploration
│   ├── 02_interactive_inference.ipynb       # LLM prediction testing
│   └── 03_exponential_decay_analysis.ipynb  # Statistical significance tests
│
├── data/                          # Data files (not in git)
│   ├── history.csv                # Wine tasting history (primary data)
│   ├── processed/                 # Processed features (auto-generated)
│   └── backups/                   # Manual backups (recommended)
│
├── scripts/                       # Utility scripts
│   └── migrate_to_sqlite.py       # CSV → SQLite migration (for 200+ wines)
│
├── models/                        # (Future) Trained models
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package configuration
├── pytest.ini                     # Test configuration
├── .env.template                  # Environment variable template
├── .gitignore                     # Git ignore rules
│
└── Documentation/
    ├── CLAUDE.md                  # Development guide (COMPREHENSIVE - read this)
    ├── CLAUDE_FIXES.md            # Implementation log (85/100 score improvements)
    ├── DATABASE_MIGRATION_GUIDE.md # CSV → SQLite migration guide
    ├── START_HERE.md              # Project overview
    ├── QUICK_REFERENCE.md         # Quick reference
    └── SECURITY.md                # Security considerations
```

**Total**: ~2000 lines of Python code across 13 modules + 45 tests + 3 notebooks

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'decant'`

**Solutions**:
1. Check virtual environment is activated:
   ```bash
   which python3  # Should show venv path
   source venv/bin/activate  # If not activated
   ```
2. Reinstall package in editable mode:
   ```bash
   pip install -e .
   ```
3. Verify installation:
   ```bash
   pip show decant
   ```

### OpenAI API Errors

**Problem**: `openai.AuthenticationError: Invalid API key`

**Solutions**:
1. Check API key is set:
   ```bash
   # macOS/Linux
   echo $OPENAI_API_KEY

   # If empty, check .env file
   cat .env
   ```
2. Verify API key is valid: https://platform.openai.com/api-keys
3. Check you have credits/access at platform.openai.com
4. Try a test API call:
   ```python
   from openai import OpenAI
   client = OpenAI()
   response = client.models.list()
   print("✓ API key valid")
   ```

**Problem**: `RateLimitError: API rate limit exceeded`

**Solutions**:
1. **Wait 60 seconds** - Rate limits reset on sliding windows
2. **Check current usage**:
   ```python
   from decant.predictor import VinoPredictor
   predictor = VinoPredictor()
   print(predictor.rate_limiter.get_stats())
   ```
3. **Adjust limits** in `src/decant/rate_limiter.py` if needed

### Streamlit Issues

**Problem**: `Address already in use` (port 8501)

**Solutions**:
1. Check what's using the port:
   ```bash
   lsof -i :8501
   ```
2. Kill the process or use alternative port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

**Problem**: Streamlit won't start

**Solutions**:
1. Check streamlit is installed: `pip show streamlit`
2. Try upgrading: `pip install --upgrade streamlit`
3. Check Python version: `python3 --version` (need 3.9+)

### Data Issues

**Problem**: Empty wine gallery / "No wines in your collection yet"

**Solutions**:
1. Add wines via "Add Wine" tab in app
2. Check `data/history.csv` exists and has data
3. Verify CSV format (12 columns, proper headers)

**Problem**: `KeyError: 'wine_color'` or missing column errors

**Solutions**:
1. Check CSV has all 12 required columns:
   ```bash
   head -1 data/history.csv
   ```
2. Compare with template in "Data Structure" section above
3. Backup and regenerate: `mv data/history.csv data/history_backup.csv`

### Pydantic Validation Errors

**Problem**: `ValidationError: [wine_color] Input should be 'White', 'Red', 'Rosé', or 'Orange'`

**Cause**: Invalid enum value in data or LLM response

**Solutions**:
1. Check wine_color values in CSV are exact matches (case-sensitive)
2. Valid values: `White`, `Red`, `Rosé`, `Orange` (note: é in Rosé)
3. Check LLM returned valid response (validation catches this automatically)

**Problem**: `ValidationError: [acidity] Input should be greater than or equal to 1.0`

**Cause**: Feature values out of range [1.0-10.0]

**Solutions**:
1. Check CSV values are in range 1.0-10.0
2. Validation automatically clamps values (warnings shown)
3. Re-extract features from photo if needed

### Test Failures

**Problem**: Tests failing after code changes

**Solutions**:
1. Check deprecation warnings: `python3 run_tests.py 2>&1 | grep -i deprecat`
2. Run specific failing test: `pytest tests/test_palate_engine.py::TestName::test_name -v`
3. Check CLAUDE_FIXES.md for patterns and common fixes
4. Ensure centralized modules used (constants.py, palate_formula.py)

### Performance Issues

**Problem**: App is slow with 200+ wines

**Cause**: CSV linear scan on every read (O(n) complexity)

**Solution**: Migrate to SQLite database
1. See `DATABASE_MIGRATION_GUIDE.md`
2. Run: `python scripts/migrate_to_sqlite.py --backup`
3. Performance improvement: 5-25x faster

## Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Visit: https://streamlit.io/cloud
   - Connect GitHub repository
   - Select `app.py` as main file
   - Add secrets: `OPENAI_API_KEY=sk-proj-...`

3. **Configure Secrets**:
   - Go to app settings → Secrets
   - Add:
     ```toml
     OPENAI_API_KEY = "sk-proj-your-key-here"
     ```

4. **Data Persistence**:
   - Upload `data/history.csv` manually (Streamlit Cloud doesn't persist local files)
   - OR: Use external storage (S3, PostgreSQL) for production

**⚠️ Important**:
- Free tier: 1GB RAM, 1 CPU, public access
- Rate limits enforced to prevent cost overruns
- Monitor usage at platform.openai.com

### Local Production

```bash
# Use production-ready WSGI server (not Streamlit's dev server)
# Note: Streamlit doesn't support WSGI, so use pm2 or systemd

# Install pm2 (Node.js required)
npm install -g pm2

# Start app
pm2 start "streamlit run app.py" --name decant

# Monitor
pm2 logs decant

# Stop
pm2 stop decant
```

## Advanced Configuration

### Adjusting Rate Limits

Edit `src/decant/rate_limiter.py`:

```python
# Default limits
limiter = RateLimiter(
    requests_per_minute=20,    # Adjust up/down
    requests_per_hour=500,     # Adjust up/down
    cost_limit_per_hour=5.0    # Adjust up/down (USD)
)
```

### Changing Exponential Decay Coefficient

Edit `src/decant/constants.py`:

```python
class AlgorithmConstants:
    EXPONENTIAL_ALPHA = 0.4  # Current value (validated via cross-validation)
    # Higher α = faster confidence growth
    # Lower α = more conservative predictions
```

**⚠️ Warning**: α=0.4 validated via statistical tests (see notebook 03). Only change if you re-run analysis with ≥100 wines.

### Custom OpenAI Model

Edit `src/decant/config.py`:

```python
OPENAI_MODEL = "gpt-5.2-2025-12-11"  # Current model
# Alternatives: "gpt-5.2-2025-12-11-mini" (cheaper), "gpt-4-turbo" (older)
```

## Additional Resources

- **CLAUDE.md**: Comprehensive development guide (read this for deep dive)
- **CLAUDE_FIXES.md**: Implementation log showing all improvements (77→85/100)
- **DATABASE_MIGRATION_GUIDE.md**: When/how to migrate to SQLite (200+ wines)
- **SECURITY.md**: Security considerations for production use
- **Streamlit Docs**: https://docs.streamlit.io
- **OpenAI API Docs**: https://platform.openai.com/docs

## Getting Help

1. **Check logs**:
   ```bash
   # Streamlit logs
   streamlit run app.py --logger.level=debug

   # Python logs
   python3 -c "import logging; logging.basicConfig(level=logging.DEBUG); from decant.predictor import VinoPredictor"
   ```

2. **Review documentation**:
   - Start with CLAUDE.md for architecture
   - Check CLAUDE_FIXES.md for known issues and fixes
   - See troubleshooting section above

3. **Verify installation**:
   ```bash
   python3 run_tests.py  # Should show 45/45 passing
   ```

## Development Patterns (IMPORTANT)

When making changes, follow these patterns from CLAUDE_FIXES.md:

✅ **DO**:
- Use enums from `constants.py` (no magic strings)
- Use `palate_formula.py` for palate calculations (SINGLE SOURCE OF TRUTH)
- Add Pydantic validation for all external inputs
- Use standardized error handling from `error_handling.py`
- Run tests before committing
- Reference CLAUDE_FIXES.md for architectural decisions

❌ **DON'T**:
- Duplicate palate formula calculations
- Use hardcoded strings for wine colors, sweetness, etc.
- Skip validation on LLM responses
- Break backward compatibility (use deprecation warnings)
- Commit `.env` files

## License & Attribution

**Built with Claude AI assistance** - This project leverages AI-powered development for:
- Initial code generation
- Architecture design
- Code review and improvements (77→85/100 score)
- Documentation generation

**Human oversight**: All AI-generated code reviewed and tested.

---

**Last Updated**: 2026-02-06 (After comprehensive code review and fixes)
**Maintained By**: Claude Code (automated improvements)
**Project Status**: Production-ready for personal use (Score: 85/100)
**Python Version**: 3.9+ required, 3.10+ recommended
