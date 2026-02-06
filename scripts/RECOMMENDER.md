# Wine Recommender System

> Built with Claude AI assistance

A sophisticated wine recommendation engine using **In-Context Learning (ICL)** with OpenAI.

## Overview

The Decant recommender uses few-shot learning to predict wine preference based on your tasting history. It:

1. **Learns from your history** - Identifies your top liked and disliked wines
2. **Extracts features** - Uses LLM to convert tasting notes into numerical features
3. **Predicts compatibility** - Applies ICL to predict match score (0-100)
4. **Provides insights** - Explains why a wine will or won't work for your palate

---

## Architecture

### `src/decant/predictor.py` - Core Logic

**VinoPredictor Class:**
- Loads `data/processed/wine_features.csv` on initialization
- Selects top 3 liked wines (by structure score: acidity + minerality)
- Selects bottom 2 disliked wines as negative examples
- Implements In-Context Learning for preference prediction

**Key Methods:**
```python
# Extract features from tasting notes (Pydantic validated)
features = predictor.extract_features(tasting_notes)

# Predict palate match with ICL
features, match = predictor.predict_match(tasting_notes)
```

**Pydantic Schemas:**
- `WineFeatures`: Validates extracted features (all 1-10 scale)
- `PalateMatch`: Validates recommendation output (match_score 0-100)

### `scripts/recommend.py` - CLI Interface

Beautiful command-line interface using the `rich` library:
- Feature table with visual bars
- Color-coded verdict panel
- Match score and qualitative analysis

---

## Usage

### Basic Usage

```bash
# Activate environment
source venv/bin/activate

# Get recommendation
python scripts/recommend.py "Crisp acidity, mineral notes, lemon zest, light body"
```

### Example Output

```
┌──────────────────────────────────────────┐
│  🍷 Decant Wine Recommender          │
│  Powered by In-Context Learning & OpenAI │
└──────────────────────────────────────────┘

🔍 Extracting wine features...
✓ Features extracted: Acid=9, Mineral=8, Fruit=6

🧠 Analyzing palate compatibility with In-Context Learning...
✓ Match score: 92/100

┌─ 🍷 Wine Feature Analysis ─────────────┐
│ Feature     Score    Bar              │
│ Acidity     9/10     🟦🟦🟦🟦🟦🟦🟦🟦🟦⚪  │
│ Minerality  8/10     ⬜⬜⬜⬜⬜⬜⬜⬜⚪⚪  │
│ Fruitiness  6/10     🟩🟩🟩🟩🟩🟩⚪⚪⚪⚪  │
│ Structure   17/20    🔷🔷🔷🔷🔷🔷🔷🔷   │
└─────────────────────────────────────────┘

┌─ 🎯 Palate Match Verdict ──────────────┐
│ ✅ Strong Match                         │
│                                          │
│ Match Score: 92/100                     │
│                                          │
│ This wine strongly aligns with your     │
│ preference for high-acid, mineral-driven│
│ wines like Chablis and Riesling.        │
└─────────────────────────────────────────┘
```

---

## In-Context Learning Explained

### What is ICL?

Instead of training a traditional ML model, we provide the LLM with **examples** of wines you liked/disliked, then ask it to reason about a new wine.

**Advantages:**
- No training data collection needed
- Adapts immediately to new preferences
- Provides explainable reasoning
- Works with small datasets (5+ wines)

### How It Works

**Step 1: Context Selection**
```python
# Top 3 liked wines (highest structure score)
Fefiñanes Albariño - Acid: 9, Mineral: 8
William Fèvre Chablis - Acid: 10, Mineral: 9
Trimbach Riesling - Acid: 10, Mineral: 7

# Bottom 2 disliked wines
Rombauer Chardonnay - Acid: 4, Mineral: 3, Body: 9
```

**Step 2: Feature Extraction**
- Convert tasting notes → numerical features (1-10)
- Validated with Pydantic `WineFeatures` schema

**Step 3: ICL Prompt**
```
You are analyzing palate compatibility.

WINES I LOVED:
1. Fefiñanes - Acid: 9, Mineral: 8...
2. William Fèvre - Acid: 10, Mineral: 9...

WINES I DISLIKED:
1. Rombauer - Acid: 4, Mineral: 3...

NEW WINE TO EVALUATE:
Acid: 8, Mineral: 7, Fruit: 6...

Predict match score and explain reasoning.
```

**Step 4: Structured Output**
- LLM returns `PalateMatch` (Pydantic validated)
- Match score (0-100), analysis, alignment, concerns

---

## Advanced Features

This implementation demonstrates modern data science and machine learning engineering:

### 1. **Few-Shot Learning**
- Uses ICL instead of traditional supervised learning
- Adapts to small datasets (cold-start problem)
- No model training/retraining required

### 2. **Pydantic Validation**
- Type-safe LLM outputs
- Guaranteed schema compliance
- Error handling with fallbacks

### 3. **Feature Engineering**
- Derives "structure score" (acidity + minerality)
- Ranks wines by preference strength
- Selects most informative examples

### 4. **Production-Ready Code**
- Environment variable management
- Clean error handling
- Comprehensive logging
- Beautiful CLI with `rich`

### 5. **Explainability**
- Not just predictions - explanations
- Shows alignment and concerns
- Traces reasoning back to examples

---

## Testing Examples

### High Match (Should Score 80+)
```bash
python scripts/recommend.py "Steely, razor-sharp acidity, flinty minerality, bone dry, citrus peel"
```

### Low Match (Should Score <40)
```bash
python scripts/recommend.py "Rich and buttery, toasted oak, vanilla cream, full-bodied, low acidity"
```

### Moderate Match (Should Score 50-70)
```bash
python scripts/recommend.py "Ripe stone fruit, balanced acidity, medium body, soft finish"
```

---

## Extending the System

### Add More Wines
1. Update `data/history.json` with new wines
2. Run `python scripts/extract_features.py`
3. Recommender automatically uses updated data

### Tune Context Selection
Modify `_select_liked_examples()` in `predictor.py`:
```python
# Use different criteria
liked_df['my_score'] = liked_df['acidity'] * liked_df['minerality']
top_liked = liked_df.nlargest(5, 'my_score')
```

### Customize Match Logic
Edit the ICL prompt in `_build_context_prompt()` to emphasize different aspects.

---

## Technical Details

**Dependencies:**
- `openai>=1.12.0` - LLM API with structured outputs
- `pydantic>=2.5.0` - Schema validation
- `pandas>=2.0.0` - Data handling
- `rich>=13.7.0` - CLI formatting
- `python-dotenv>=1.0.0` - Environment management

**Performance:**
- Feature extraction: ~2-3 seconds
- Match prediction: ~3-4 seconds
- Total latency: ~5-7 seconds (acceptable for interactive use)

**Cost:**
- ~$0.01-0.02 per recommendation (GPT-4o pricing)
- Batching possible for efficiency
