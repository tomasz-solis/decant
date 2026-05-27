# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**Decant** is a personal wine recommendation app using AI-powered flavor profile matching. Built with Streamlit and OpenAI (currently GPT-5.4-mini), it learns user preferences and predicts wine compatibility based on a 5-dimensional flavor space (acidity, minerality, fruitiness, tannin, body).

**Target Users**: Personal use (1-3 users, designed for a couple)
**Tech Stack**: Python 3.12+, Streamlit, OpenAI API, Supabase (PostgreSQL + Auth), Pandas, Pydantic
**Deployment**: Streamlit Cloud + Supabase
**Primary Use Case**: 📱 In-shop wine checking on mobile

## Quick Start

### Environment Setup
```bash
# Sync dependencies (uv reads pyproject.toml + uv.lock)
uv sync

# Set up environment variables — copy from .env.template if it exists,
# or write directly into .streamlit/secrets.toml for Streamlit
```

### Running the App
```bash
uv run streamlit run app.py

# The app will open at http://localhost:8501
```

### Testing
```bash
# Run all tests
uv run pytest --no-cov -q

# Or with coverage
uv run pytest --cov=src/decant --cov-report=html
open htmlcov/index.html
```

## Project Architecture

### Directory Structure
```
decant/
├── app.py                      # Main Streamlit application (1950+ lines)
├── data/
│   ├── history.csv            # Wine tasting history (not in git)
│   └── processed/             # Processed features (not in git)
├── src/decant/                # Core Python package
│   ├── constants.py           # Enums, constants, validation schemas
│   ├── palate_formula.py      # Centralized palate calculation (SINGLE SOURCE OF TRUTH)
│   ├── palate_engine.py       # Palate matching algorithm with cosine similarity
│   ├── predictor.py           # LLM-based wine preference predictor
│   ├── rate_limiter.py        # API rate limiting and cost tracking
│   ├── error_handling.py      # Standardized error handling
│   ├── schema.py              # Pydantic data models
│   ├── utils.py               # Utility functions, prompt injection defense
│   └── config.py              # Configuration constants
├── tests/                     # Pytest test suite (45 tests, 100% passing)
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 01_palate_exploration.ipynb
│   ├── 02_interactive_inference.ipynb
│   └── 03_exponential_decay_analysis.ipynb  # With statistical significance testing
├── scripts/                   # Utility scripts
└── models/                    # (Future) Trained models

Documentation:
├── CLAUDE_FIXES.md           # Implementation log of all code review fixes
├── DATABASE_MIGRATION_GUIDE.md  # CSV → SQLite migration guide (for 200+ wines)
├── START_HERE.md             # Project overview
├── QUICK_REFERENCE.md        # Quick reference guide
└── SECURITY.md               # Security considerations
```

### Core Modules

**app.py** - Main Streamlit UI
- Wine gallery with filtering
- Photo-based wine extraction (GPT-5.4 Vision)
- Palate matching predictions
- Feature visualization (radar charts)

**src/decant/constants.py** - Centralized Constants
- `WineColor`, `Sweetness`, `Verdict` enums
- `ColumnNames` for all CSV columns
- `AlgorithmConstants` (α=0.4 for exponential decay, etc.)
- Pydantic validation schemas for LLM responses

**src/decant/palate_formula.py** - Palate Calculation (SINGLE SOURCE OF TRUTH)
- `calculate_palate_features()`: Core formula
- `add_palate_features_to_dataframe()`: Vectorized for DataFrames
- `calculate_wine_similarity()`: Unified similarity metric

**src/decant/palate_engine.py** - Matching Algorithm
- Cosine similarity in 5D flavor space
- Exponential confidence decay: `1 - e^(-0.4 * N)` where N = sample size
- Color-specific matching with fallback
- Dual-metric scoring: palate_match (raw) + likelihood_score (confidence-adjusted)

**src/decant/predictor.py** - LLM-Based Prediction
- In-context learning with user's wine history
- Retry logic with exponential backoff
- LLM response caching (SHA256, 24h TTL)
- **NEW**: Integrated rate limiter and centralized palate formula

**src/decant/rate_limiter.py** - API Cost Protection
- 20 requests/minute, 500 requests/hour limits
- $5/hour cost limit
- Sliding window tracking
- Automatic cost calculation from token usage

**src/decant/error_handling.py** - Standardized Errors
- Exception hierarchy: `DecantError`, `LLMError`, etc.
- `handle_llm_error()`: Consistent LLM error handling
- Context managers and decorators

## Key Features

### Security
✅ **Pydantic validation on ALL LLM responses** (prevents KeyError crashes, jailbreak attacks)
✅ **Rate limiting** (prevents API cost overruns)
✅ **Prompt injection defense** (multi-line, punctuation-separated patterns)
✅ **Input sanitization** (all user inputs)

### Data Science
- **5D flavor space**: acidity, minerality, fruitiness, tannin, body
- **Palate formula**: `structure_score + (acidity_body_ratio * 2)`
- **Cosine similarity** for flavor profile matching
- **Exponential confidence decay**: `1 - e^(-0.4 * N)` (validated via cross-validation)
- **Statistical rigor**: Paired t-tests, 95% CIs, power analysis in notebook 03

### AI/LLM Integration
- **GPT-5.4** for wine feature extraction from photos
- **GPT-5.4** for text-based wine inference
- **In-context learning** using user's wine history
- **Response caching** (24h TTL, SHA256 keys)
- **Retry logic** (3 attempts, exponential backoff)

## Development Workflow

### Making Changes

1. **Read existing code first** - Use Read tool before editing
2. **Run tests after changes** - `python3 run_tests.py`
3. **Check CLAUDE_FIXES.md** - See what's been fixed and why
4. **Maintain SINGLE SOURCE OF TRUTH** - Use centralized modules (palate_formula.py, constants.py)

### Adding New Features

1. **Use existing patterns**:
   - Enums in `constants.py`
   - Pydantic validation for all external inputs
   - Error handling from `error_handling.py`
   - Palate calculations from `palate_formula.py`

2. **Write tests**: Add to `tests/` directory

3. **Update documentation**: Keep CLAUDE.md in sync

### Code Quality Standards

- ✅ All tests must pass (45/45)
- ✅ No deprecation warnings
- ✅ Pydantic validation for all LLM responses
- ✅ Use centralized constants (no magic strings)
- ✅ Use centralized palate formula (no duplication)
- ✅ Standardized error handling

## Important Constraints

### Data Storage
- **Current**: CSV files (`data/history.csv`)
- **Limit**: ~300 wines before performance degrades
- **Migration path**: See `DATABASE_MIGRATION_GUIDE.md` for SQLite migration at 200+ wines

### API Usage
- **Rate limits**: 20 req/min, 500 req/hour, $5/hour (configurable in rate_limiter.py)
- **Cost tracking**: Automatic via token usage
- **Caching**: 24h TTL on LLM responses

### Target Users
- **1-2 users max** (designed for a couple)
- **Personal deployment** on Streamlit Cloud
- **NOT for public/commercial use** without additional security hardening

## Known Limitations

1. **Small dataset size** (~30 wines currently)
   - Statistical tests are exploratory only
   - Need 100+ wines for robust conclusions
   - Re-run notebook 03 at 50, 100, 200 wines

2. **LLM non-determinism**
   - Despite `temperature=0` and `seed=42`, OpenAI responses not guaranteed deterministic
   - Caching mitigates this for repeated queries

3. **CSV performance**
   - Linear scan on every read
   - No indexing or query optimization
   - Migrate to SQLite at 200-300 wines

4. **App.py monolith**
   - 1950+ lines (not yet refactored into modules)
   - See MEDIUM priority fixes in CLAUDE_FIXES.md for refactoring plan

## Testing Philosophy

- **Unit tests**: 120 tests across palate engine, data access, schema, supabase session, styles regression, and wine matching
- **Integration tests**: not implemented — Streamlit UI is impractical to unit-test, so the meaningful end-to-end test is launching `streamlit run app.py` and clicking through
- **Manual testing**: required for UI changes

## Documentation

- **README.md**: project description, setup, repo layout
- **docs/ALGORITHM.md**: centred-cosine math, verdict logic, threshold reasoning
- **docs/SECURITY.md**: security considerations and limitations
- **docs/GET_SUPABASE_KEYS.md**: how to find the URL and anon key in the Supabase dashboard

## Troubleshooting

### Tests failing
```bash
uv run pytest --no-cov -q  # Check which tests are failing
# Common issues:
# - Missing dependencies (uv sync)
# - OPENAI_API_KEY not set (not needed for tests)
```

### Streamlit app not starting
```bash
# Check if port 8501 is in use
lsof -i :8501

# Try alternative port
streamlit run app.py --server.port 8502
```

### API rate limits hit
```python
# Check current usage
from decant.predictor import VinoPredictor
predictor = VinoPredictor()
stats = predictor.rate_limiter.get_stats()
print(stats)

# Reset if needed (dev only)
predictor.rate_limiter.reset()
```

## Contributing

This is a personal project, but if making changes:
1. Maintain backward compatibility (deprecated functions OK, breaking changes NOT OK)
2. Run all tests before committing
3. Update documentation
