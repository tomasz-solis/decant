# Decant

*Taste, with confidence.*

A household wine app. Take a photo of a bottle, or type the name, and Decant predicts how much you'll like it based on the wines you've already rated. Built for two users sharing a single cellar — not multi-tenant SaaS.

## What it does

- **Add wines** — upload a label photo (OpenAI Vision extracts the producer, region, vintage, and tasting features) or type the name (the model infers a feature profile from training-data context).
- **Predict palate match** — a centred-cosine score against the average flavour profile of the wines you've liked. See [docs/ALGORITHM.md](docs/ALGORITHM.md) for the math.
- **Browse the collection** — search and filter, with a "you've had this" badge if a wine you're adding matches one already in the cellar.
- **See your palate at a glance** — a stats tab with top regions, top wines, and your ideal flavour profile per wine colour.

## Architecture

- **Streamlit** for the UI
- **Supabase** (PostgreSQL + Auth) for the wine table and household login
- **OpenAI** for photo extraction and text inference (Vision + chat completions)
- **No background workers, no scheduled tasks, no caching layer beyond Streamlit's built-ins.** It's a single Python process per user session.

The household uses one Supabase account. Anonymous visitors can browse the gallery and palate maps but can't add wines or call the OpenAI API.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # install dependencies
uv run streamlit run app.py   # launch locally at http://localhost:8501
```

You'll need a `.streamlit/secrets.toml` with:

```toml
OPENAI_API_KEY = "sk-..."
SUPABASE_URL = "https://YOURPROJECT.supabase.co"
SUPABASE_KEY = "eyJ..."         # the anon key, not service_role
CELLAR_ID = "..."               # UUID of the row in the cellars table
CONTACT_EMAIL = "you@example.com"  # optional; used for the 'need help?' mailto
```

See [docs/GET_SUPABASE_KEYS.md](docs/GET_SUPABASE_KEYS.md) for how to find the URL and anon key. The `CELLAR_ID` is the UUID of the cellar row your household owns.

## Tests

```bash
uv run pytest --no-cov -q
```

The suite covers the palate engine, data access layer, schema validation, Supabase session helpers, the styling regression (no inline CSS in `app.py`), and wine matching for prior-tasting detection.

## Repository layout

```
app.py                          Streamlit entry point. Routes to tab modules.
src/decant/
    palate_engine.py            Centred-cosine scoring + verdict assignment.
    schema.py                   Pydantic models for the wine schema.
    supabase_session.py         Auth and client-getter helpers.
    wines_repo.py               CRUD against the wines table.
    services/
        data_access.py          Normalise wine DataFrames; single source of schema truth.
        image_storage.py        Local bottle-photo storage.
        vision_extract.py       OpenAI Vision call for label extraction.
        wine_match.py           Token-based "have I had this before" matching.
    ui/
        auth_form.py            Sign-in popover and help-by-email fallback.
        components.py           Plotly chart functions (radars, decision boundary).
        helpers.py              Small UI helpers (vintage display, empty-state diagnostics).
        styles.py               Inline CSS, both global and gallery-scoped.
        tab_add_wine.py         Tab 1 body.
        tab_palate_maps.py      Tab 2 body.
        tab_stats.py            Tab 3 body.
        tab_gallery.py          Tab 4 body.
tests/                          pytest suite.
docs/                           Algorithm, security, Supabase setup.
sql/                            Migrations and RLS policies.
```

## Status

Personal project, household use only. Not actively soliciting external contributions or feature requests.

## License

MIT — see [LICENSE](LICENSE).
