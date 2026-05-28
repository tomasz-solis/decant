# Algorithm — centred cosine + alignment-led verdict

Two related changes to how the palate match score is computed and presented. Both shipped after the original "plain cosine" v1.

## What changed

The palate match score used to be plain cosine similarity between a wine's feature vector and the mean of liked wines, mapped from `[-1, 1]` to `[0, 100]` via `(x + 1) / 2 * 100`.

That mapping is mathematically correct in general, but wrong for this app's data. All five features (acidity, fruitiness, body, tannin, minerality) are positive, bounded in `[1, 10]`. Cosine between two positive 5D vectors is bounded approximately in `[0.5, 1]`. The `(x + 1) / 2` mapping therefore produces scores in `[75, 100]` — a "neutral" wine reads as ~85% match, a strong match as ~95%.

Under v2, both vectors are centred against the population mean (the average across all rated wines) before cosine is computed. Centred cosine genuinely uses the full `[-1, 1]` range, so the `(x + 1) / 2` mapping recovers a meaningful `[0, 100]` spread.

## What "match" means now

- **100**: this wine deviates from typical wines in the same direction as the wines you've liked
- **50**: no correlation between this wine's deviation from typical and your liked wines' deviation from typical
- **0**: this wine deviates in the opposite direction

The semantic shift is real. Under v1, a high score meant "this wine's raw feature vector points in a similar direction to your liked wines' raw feature vector" — which, given all features are positive, was almost always true. Under v2, a high score means "this wine is unusual in the same way your liked wines are unusual."

## What old scores would now look like

Run on the test fixture (5 wines: 3 liked Albariños, 1 liked Rioja, 1 disliked Red):

| wine                         | v1 match | v2 match |
|------------------------------|----------|----------|
| similar Albariño-style white | ~95      | ~100     |
| wine at population mean      | ~85      | 50       |
| opposite style (tannic red)  | ~25      | 0        |

The headline differences:
- A wine that's *average* now reads as 50, not 85
- An opposite-style wine reads as 0, not 25
- Strong matches stay high

If you look at a wine you previously rated 92% and it now scores 65%, that isn't a regression. The old 92% was inflated.

## Verdict logic — alignment and confidence as separate dimensions

The verdict shown to the user is **not** derived from a single threshold on `palate_match × confidence_factor`. That used to be the design and it was wrong, for the reason called out in [the display fix commit](#): high alignment with low confidence was rendering as "65% Strong Match," which read like a regression to anyone whose mental model was "Strong Match means a high number."

Now the engine considers alignment and confidence as separate facts:

| condition | verdict |
|---|---|
| `palate_match ≥ 70` AND `confidence_factor ≥ 0.6` | 💙 Strong Match |
| `palate_match ≥ 70` AND `confidence_factor < 0.6` | 🌱 Promising — rate more to confirm |
| `palate_match ≥ 55` | 🟡 Worth Exploring |
| `palate_match < 55` | ⚪ Different Style |

The UI shows `palate_match` as the headline number, with confidence presented qualitatively ("moderate, based on 3 wines") in a breakdown panel. The multiplication formula is no longer surfaced anywhere — it conflated two different things that the user benefits from reading independently.

`palate_match ≥ 70` corresponds to centred cosine ≥ 0.4 — solidly positive alignment with the liked-deviation pattern. `confidence_factor ≥ 0.6` corresponds to roughly 3+ liked wines under the exponential confidence formula.

`likelihood_score` (the product) is still computed and exposed on the `PalateScore` dataclass for backward compatibility, but nothing in the UI reads it. The `Verdict` enum in `constants.py` is marked legacy for the same reason — no live code consumes it.

These thresholds are based on theoretical reasoning, not empirical calibration. With a real labelled dataset ("the app said Strong Match — was it actually a strong match?") they should be retuned. Until then, treat them as defensible-but-uncertified.

## Cold-start behaviour

Centring against a population mean requires the population mean to be meaningful. With fewer than 3 rated wines, the mean is too noisy to centre against — centring would *add* variance rather than removing baseline bias. In that regime, the engine falls back to plain cosine. This is documented in `_centred_cosine` and tested in `test_falls_back_to_plain_cosine_with_small_history`.

The fallback applies regardless of how many wines are liked vs disliked — it's about total rated wines (`n_total >= 3`), not liked wines (`n_liked`).

## What this doesn't fix

The centring point is the *user's own history* mean, which is small-sample noisy until you have a few dozen wines rated. A better long-term option is centring against a reference population (a corpus of typical wines). That's not in scope for v2.

The exponential confidence factor (`1 - e^(-0.4N)`) was not changed. It still dampens scores when sample size is low. The α=0.4 is itself uncalibrated — empirically it gives moderate confidence around N=5 and high confidence around N=10, which feels right for a household tool, but isn't derived from anything.

## File-level summary

- `palate_engine.py:_compute_ideal_profile`: stores `population_mean` and `n_total` alongside the per-colour ideal vectors
- `palate_engine.py:_centred_cosine`: the v2 similarity function. Falls back to plain cosine when `n_total < 3`.
- `palate_engine.py:cosine_similarity`: still around for the fallback path; docstring warns about the positive-vector caveat
- `palate_engine.py:calculate_match`: uses `_centred_cosine`; verdict is built from `palate_match` and `confidence_factor` separately, not from `likelihood_score`
- `ui/tab_add_wine.py`: headline display number is `palate_match`. Breakdown panel presents alignment and confidence as parallel facts.
- `constants.py:Verdict`: marked legacy; not read anywhere
- `tests/test_palate_engine.py`: `TestCentredCosine` and `TestPopulationMean` pin the v2 math; the new verdict tests pin "high alignment + low confidence → Promising, not Strong"

## Score stability under repeated extraction

Two layers of the score path interact with non-determinism in different ways, and it's worth being honest about both.

**The engine math is deterministic.** Given the same candidate feature vector and the same history, `calculate_match` returns the bit-identical score every time. This is pinned by `TestScoreDeterminism` (5 successive calls on the same inputs must produce one unique value; row order in the history must not affect the result).

**The candidate feature vector is only as stable as the LLM that produced it.** For text-entered wines, the five flavour features (acidity, fruitiness, body, tannin, minerality) come from `predictor.extract_wine_data` — a single OpenAI call. The call uses `temperature=0` and `seed=OPENAI_SEED`, but OpenAI's own docs state seed is best-effort; the `system_fingerprint` can change between requests and the output can vary. Centred cosine measures the *angle* between (candidate − population_mean) and (ideal − population_mean), so when a wine sits near the population mean, a small feature wobble swings the angle non-linearly and the score moves a few percent.

To bound this, `extract_wine_data` is cached by wine name (file-based SHA256 keyed on the prompt + model). The first extraction is frozen; subsequent calls with the same wine name return the cached features bit-identical. With the cache populated, the candidate vector cannot drift, and the score is stable for the lifetime of the cache entry.

**Where the residual instability comes from.** When the cache is cold — first extraction of a wine, after the 24h TTL expires, or in any scenario that hits the LLM fresh — the score reflects whatever features that particular call returned. Two cold extractions of the same wine, hours apart, can land on slightly different feature vectors (e.g. acidity 7.2 vs 7.6) and so produce scores a few percent apart (observed spread: ~3–4 points on a Burgundy white near the user's population mean). This is upstream of the algorithm and not fixable on our side without either (a) replacing the LLM-inferred features with a deterministic source, or (b) accepting OpenAI's eventual deterministic-output guarantees if they ship them.

**What the user actually experiences.** Within a single Streamlit session, after the first extraction, the score is rock-stable — every rerun reads cached features. Across full app reloads or after the cache expires, the score can drift by a few percent. The headline number is rounded to one decimal, the verdict bands are wide ("Strong Match" covers ~75–100), and a 3-point swing rarely crosses a band — so the qualitative judgement is stable even when the digit-level number isn't.

**Why not just disable the LLM path.** The text-entry flow exists because users often enter wines they've seen on a list or at a restaurant without having a photo. Removing LLM inference here would force everyone to either upload a photo or enter five sliders manually, which defeats the point. The trade-off — accept a few percent of LLM-induced jitter in exchange for "type a name, get a score" — is the right one for a household tool. A production system with stricter stability requirements would need a deterministic feature source (a vetted producer/region lookup table, or a fine-tuned model exposed via batch inference).

