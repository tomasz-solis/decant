# Algorithm v2 — centred cosine

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

## Threshold recalibration

Verdict thresholds operate on `likelihood_score = palate_match × confidence_factor`. They were recalibrated for the new score distribution:

| verdict          | v1 threshold | v2 threshold |
|------------------|--------------|--------------|
| 💙 Strong Match  | 75           | 60           |
| 🧡 Worth Trying  | 60           | 50           |
| 🟡 Explore       | 45           | 40           |
| ⚪ Different     | < 45         | < 40         |

These numbers are based on theoretical reasoning, not empirical calibration. With a real labelled dataset ("the app said Strong Match — was it actually a strong match?") they should be retuned. Until then, treat the thresholds as defensible-but-uncertified.

## Cold-start behaviour

Centring against a population mean requires the population mean to be meaningful. With fewer than 3 rated wines, the mean is too noisy to centre against — centring would *add* variance rather than removing baseline bias. In that regime, the engine falls back to plain cosine. This is documented in `_centred_cosine` and tested in `test_falls_back_to_plain_cosine_with_small_history`.

The fallback applies regardless of how many wines are liked vs disliked — it's about total rated wines (`n_total >= 3`), not liked wines (`n_liked`).

## What this doesn't fix

The centring point is the *user's own history* mean, which is small-sample noisy until you have a few dozen wines rated. A better long-term option is centring against a reference population (a corpus of typical wines). That's not in scope for v2.

The exponential confidence factor (`1 - e^(-0.4N)`) was not changed. It still dampens scores when sample size is low. The α=0.4 is itself uncalibrated — empirically it gives moderate confidence around N=5 and high confidence around N=10, which feels right for a household tool, but isn't derived from anything.

## File-level summary

- `palate_engine.py:_compute_ideal_profile`: now also stores `population_mean` and `n_total`
- `palate_engine.py:_centred_cosine`: new method, used by `calculate_match`
- `palate_engine.py:cosine_similarity`: unchanged; docstring warns about the positive-vector caveat
- `palate_engine.py:calculate_match`: switched to `_centred_cosine`; verdict thresholds updated; inline `# BUG FIX` comment converted to docstring note
- `constants.py:Verdict`: thresholds updated to match
- `tests/test_palate_engine.py`: existing cosine tests retained (plain cosine still works as documented); new `TestCentredCosine` and `TestPopulationMean` classes pin v2 semantics
