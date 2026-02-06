# Scripts

> Built with Claude AI assistance

Utility scripts for Decant data processing, feature extraction, and data migration.

## extract_features.py

Extracts numerical features from wine tasting notes using OpenAI API.

**What it does:**
1. Reads `data/history.json` with your wine comparisons
2. Uses OpenAI with structured outputs to analyze tasting notes
3. Extracts 5 numerical features (1-10 scale):
   - **Acidity**: Perceived acidity/crispness
   - **Minerality**: Mineral/saline character
   - **Fruitiness**: Fruit intensity
   - **Tannin**: Tannin structure
   - **Body**: Weight and texture
4. Validates LLM output using Pydantic schemas
5. Outputs `data/processed/wine_features.csv` with features + liked labels

**Usage:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run the script
python scripts/extract_features.py
```

**Output:**
- CSV file with wine metadata + extracted features + liked labels
- Summary statistics printed to console
- Validated using `decant.schema.WineFeatures` Pydantic model

**Key Features:**
This script demonstrates:
- LLM integration with structured outputs (Pydantic validation)
- Robust error handling and validation
- Feature engineering from unstructured text
- Production-ready logging and output

---

## migrate_to_csv.py

Converts nested JSON wine history to flat CSV format for easier analysis.

**What it does:**
1. Reads `data/history.json` with nested structure
2. Flattens wine data into tabular format
3. Combines tasting notes into single text field
4. Outputs `data/history.csv` with all wine details

**Usage:**
```bash
python scripts/migrate_to_csv.py
```

**Output Columns:**
- `wine_name`: Producer + vintage
- `producer`: Winery name
- `vintage`: Year
- `notes`: Combined tasting notes (appearance, nose, palate, overall)
- `score`: Overall score from tasting
- `liked`: Boolean preference
- `price`: Price in USD
- `acidity`: Acidity score (1-10)
- `minerality`: Minerality score (placeholder, filled by extract_features.py)
- `fruitiness`: Fruitiness score (placeholder, filled by extract_features.py)
- `tannin`: Tannin score (placeholder, filled by extract_features.py)
- `body`: Body score (1-10)
- `complexity`: Complexity score
- `finish`: Finish score
- `wine_type`: Wine category (Albariño, White Wine, etc.)
- `region`: Geographic region
- `comparison_date`: Date of tasting

**Note:**
The `minerality`, `fruitiness`, and `tannin` columns are initialized to 0. Run `extract_features.py` to populate these with AI-extracted values from tasting notes.

**Example Output:**
```
wine_name                    liked  price  acidity  body  score
Fefiñanes 2022              True   18.99  9        7     8.5
Rombauer 2021               False  42.99  4        9     5.5
```

---

## manage_wines.py

Interactive CLI tool for managing your wine database (add, remove, list wines).

**What it does:**
1. Lists all wines with row numbers and key details
2. Removes wines by ID or name pattern
3. Creates automatic backups before changes
4. Auto-regenerates features after removal
5. Shows CSV template for manual additions

**Usage:**

```bash
# List all wines with IDs
python scripts/manage_wines.py --list

# Remove wine by row number
python scripts/manage_wines.py --remove 3

# Remove multiple wines
python scripts/manage_wines.py --remove 3,5,7

# Remove by wine name (partial match)
python scripts/manage_wines.py --remove-name "Rombauer"

# Regenerate features only
python scripts/manage_wines.py --regenerate

# Show CSV template for adding wines
python scripts/manage_wines.py --template
```

**Features:**
- 🎨 Beautiful rich console output with tables
- 💾 Automatic backups before any changes
- ✅ Confirmation prompts before deletion
- 🔄 Auto-regenerates features after removal
- 🔍 Search by name with partial matching
- 📝 CSV template for manual additions

**Example Output:**
```
🍷 Wine Database Manager

🍷 Wine Database
Total wines: 5

┏━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━┓
┃# ┃ Wine Name             ┃ Producer     ┃ Vintage┃ Liked┃  Price ┃
┡━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━┩
│0 │ Fefiñanes 2022        │ Fefiñanes    │ 2022   │ ✅   │ €18.99 │
│1 │ Martín Códax 2022     │ Martín Códax │ 2022   │ ✅   │ €14.99 │
└──┴───────────────────────┴──────────────┴────────┴──────┴────────┘
```

**Workflow for Removing Wines:**
1. List wines to find row numbers: `--list`
2. Remove wine(s): `--remove 3` or `--remove-name "Rombauer"`
3. Script automatically:
   - Creates backup
   - Removes from history.csv
   - Regenerates wine_features.csv
   - Shows confirmation

**Workflow for Adding Wines:**
1. Use Streamlit app (recommended), OR
2. Get CSV template: `--template`
3. Add row to `data/history.csv`
4. Run: `--regenerate`
