"""Detect whether a wine has been tasted before.

Two-stage token-based matching:

1. **Producer match** (gate). Producer is the strongest identity
   signal - same producer name almost certainly means the same
   winemaker. Jaccard >= 0.7 on producer tokens required to proceed.

2. **Name + vintage** (rank). Among producer-matching history rows,
   compute Jaccard on the name tokens; pick the best one. The
   vintage match decides the `match_kind`:
   - same vintage → "exact"
   - different vintage → "different_vintage"

Tokenization strips punctuation, lowercases, drops stopwords common
to wine names (Domaine, Chateau, Cru, etc.), and drops tokens shorter
than two characters.

The module is pure: no Streamlit, no Supabase, no IO. Inputs are
plain types and a pandas DataFrame; outputs are dataclasses or None.
That keeps it cheap to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


# Common wine-name words that carry no identifying signal - dropping
# them prevents false matches like "Chateau X" matching "Chateau Y"
# purely on the "chateau" token.
_STOPWORDS = frozenset({
    "the", "domaine", "domaines", "chateau", "château", "weingut",
    "vino", "vinos", "vina", "viña", "vinha", "cantina", "estate",
    "cellars", "winery", "vineyard", "vineyards", "bodega", "bodegas",
    "azienda", "agricola", "tenuta",
    # Quality tier markers that appear on many wines from the same
    # producer - useful to drop so different cuvées don't accidentally
    # match each other on the tier word alone.
    "1er", "premier", "grand", "cru", "reserve", "réserve", "reserva",
    "gran", "riserva",
})

_PUNCT_RE = re.compile(r"[.,;:!?\"'&()\[\]{}\- - - _/\\]")

_PRODUCER_MATCH_THRESHOLD = 0.7
_NAME_MATCH_THRESHOLD = 0.5


@dataclass(frozen=True)
class PriorTasting:
    """A prior tasting of the same (or related) wine.

    `match_kind`:
        "exact" - same producer, same wine name, same vintage
        "different_vintage" - same producer, same wine name, different vintage
    """

    wine_name: str
    producer: str
    vintage: Optional[int]
    score: Optional[float]
    liked: Optional[bool]
    match_kind: Literal["exact", "different_vintage"]


def tokenize_wine(text: Optional[str]) -> frozenset[str]:
    """Normalise a wine name or producer string into a token set.

    Empty / None input returns an empty set. Stopwords and tokens
    shorter than 2 chars are dropped. Diacritics are preserved (the
    stopword list includes both "chateau" and "château").
    """
    if not text:
        return frozenset()
    cleaned = _PUNCT_RE.sub(" ", str(text).lower())
    tokens = (t for t in cleaned.split() if len(t) >= 2 and t not in _STOPWORDS)
    return frozenset(tokens)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Set similarity: |intersection| / |union|. Returns 0.0 on empty union."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_prior_tasting(
    candidate_name: str,
    candidate_producer: Optional[str],
    candidate_vintage: Optional[int],
    history_df: pd.DataFrame,
) -> Optional[PriorTasting]:
    """Find the best prior tasting that matches the candidate wine.

    Returns the strongest match, preferring same-vintage over different-
    vintage when both qualify. None if no prior tasting passes the
    producer-and-name thresholds.

    Args:
        candidate_name: The wine name being added (e.g. from extraction).
        candidate_producer: Producer name. If None or empty, the function
            falls back to name-only matching with a higher threshold.
        candidate_vintage: Year, or None if unknown.
        history_df: The wine history (already normalised - assumes the
            standard columns: wine_name, producer, vintage, score, liked).

    Behaviour notes:
        - With no producer, this degrades to name-Jaccard only at the
          higher threshold (0.7). Less reliable; an LLM that doesn't
          extract the producer is the main risk here.
        - "Exact" requires vintage equality. Vintage=0 (the schema
          sentinel for unknown) on either side disqualifies exact.
    """
    if history_df is None or history_df.empty:
        return None

    candidate_name_tokens = tokenize_wine(candidate_name)
    candidate_producer_tokens = tokenize_wine(candidate_producer)

    if not candidate_name_tokens:
        # Nothing to match on. Don't pretend we have signal.
        return None

    best_score = -1.0
    best_row: Optional[pd.Series] = None
    best_match_kind: Optional[Literal["exact", "different_vintage"]] = None

    for _, row in history_df.iterrows():
        row_name_tokens = tokenize_wine(row.get("wine_name"))
        row_producer_tokens = tokenize_wine(row.get("producer"))

        if candidate_producer_tokens and row_producer_tokens:
            # Producer present on both sides - gate on producer match.
            producer_score = _jaccard(candidate_producer_tokens, row_producer_tokens)
            if producer_score < _PRODUCER_MATCH_THRESHOLD:
                continue
            name_threshold = _NAME_MATCH_THRESHOLD
        else:
            # No producer to gate on; demand a stronger name match
            # instead. This is the degraded path - false positives
            # are more likely here.
            name_threshold = 0.7

        name_score = _jaccard(candidate_name_tokens, row_name_tokens)
        if name_score < name_threshold:
            continue

        # Vintage decides match kind. A vintage of 0 (the schema
        # sentinel for "unknown") is not eligible for exact match.
        row_vintage = row.get("vintage")
        try:
            row_vintage_int = int(row_vintage) if pd.notna(row_vintage) else 0
        except (ValueError, TypeError):
            row_vintage_int = 0

        candidate_vintage_int = (
            int(candidate_vintage)
            if candidate_vintage and candidate_vintage > 0
            else 0
        )

        if (
            candidate_vintage_int > 0
            and row_vintage_int > 0
            and candidate_vintage_int == row_vintage_int
        ):
            match_kind: Literal["exact", "different_vintage"] = "exact"
            # Exact matches get a score bonus so they win over a
            # higher-name-similarity different-vintage candidate.
            ranked = name_score + 1.0
        else:
            match_kind = "different_vintage"
            ranked = name_score

        if ranked > best_score:
            best_score = ranked
            best_row = row
            best_match_kind = match_kind

    if best_row is None or best_match_kind is None:
        return None

    # Coerce types out of the row safely.
    row_vintage_val = best_row.get("vintage")
    try:
        vintage_out: Optional[int] = (
            int(row_vintage_val) if pd.notna(row_vintage_val) and row_vintage_val else None
        )
    except (ValueError, TypeError):
        vintage_out = None

    row_score_val = best_row.get("score")
    try:
        score_out: Optional[float] = (
            float(row_score_val) if pd.notna(row_score_val) else None
        )
    except (ValueError, TypeError):
        score_out = None

    row_liked_val = best_row.get("liked")
    liked_out: Optional[bool] = bool(row_liked_val) if pd.notna(row_liked_val) else None

    return PriorTasting(
        wine_name=str(best_row.get("wine_name") or ""),
        producer=str(best_row.get("producer") or ""),
        vintage=vintage_out,
        score=score_out,
        liked=liked_out,
        match_kind=best_match_kind,
    )
