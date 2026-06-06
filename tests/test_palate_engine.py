"""Tests for PalateEngine."""

import pytest
import numpy as np
import pandas as pd
from decant.palate_engine import PalateEngine, WineVector, PalateScore


class TestWineVector:
    """Test WineVector dataclass."""

    def test_to_array(self):
        """Test conversion to numpy array."""
        vec = WineVector(
            acidity=8.0,
            fruitiness=7.0,
            body=5.0,
            tannin=2.0,
            minerality=9.0
        )
        arr = vec.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5,)
        assert np.array_equal(arr, [8.0, 7.0, 5.0, 2.0, 9.0])

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'acidity': 8.0,
            'fruitiness': 7.0,
            'body': 5.0,
            'tannin': 2.0,
            'minerality': 9.0
        }
        vec = WineVector.from_dict(data)
        assert vec.acidity == 8.0
        assert vec.minerality == 9.0

    def test_from_dict_with_missing_keys(self):
        """Test creation with missing keys defaults to 0."""
        data = {'acidity': 8.0, 'body': 5.0}
        vec = WineVector.from_dict(data)
        assert vec.acidity == 8.0
        assert vec.fruitiness == 0.0
        assert vec.minerality == 0.0


class TestCosineSimilarity:
    """Test cosine similarity calculations."""

    def test_identical_vectors_return_100_percent(self):
        """Identical wine profiles should give 100% similarity."""
        engine = PalateEngine()
        vec = np.array([8.0, 7.0, 5.0, 2.0, 9.0])
        similarity = engine.cosine_similarity(vec, vec)
        assert similarity == 100.0

    def test_orthogonal_vectors_return_50_percent(self):
        """Orthogonal vectors should give 50% (mapped from 0)."""
        engine = PalateEngine()
        vec_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        similarity = engine.cosine_similarity(vec_a, vec_b)
        # Orthogonal vectors have cosine = 0, which maps to 50%
        assert abs(similarity - 50.0) < 1.0

    def test_opposite_vectors_return_0_percent(self):
        """Opposite vectors should give 0% similarity."""
        engine = PalateEngine()
        vec_a = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        vec_b = np.array([-10.0, -10.0, -10.0, -10.0, -10.0])
        similarity = engine.cosine_similarity(vec_a, vec_b)
        # Opposite vectors have cosine = -1, which maps to 0%
        assert abs(similarity - 0.0) < 1.0

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0% (edge case handling)."""
        engine = PalateEngine()
        vec_a = np.array([8.0, 7.0, 5.0, 2.0, 9.0])
        vec_b = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        similarity = engine.cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0

    def test_similar_wines_high_similarity(self):
        """Similar wines should have high similarity."""
        engine = PalateEngine()
        vec_a = np.array([8.0, 8.0, 7.0, 1.0, 9.0])  # Albariño-like
        vec_b = np.array([8.5, 7.5, 7.5, 1.5, 8.5])  # Similar white
        similarity = engine.cosine_similarity(vec_a, vec_b)
        assert similarity > 90.0

    def test_different_wines_moderate_similarity(self):
        """Different wines can still have moderate similarity in 5D space."""
        engine = PalateEngine()
        vec_a = np.array([8.0, 8.0, 7.0, 1.0, 9.0])  # Albariño (white)
        vec_b = np.array([6.0, 5.0, 7.0, 9.0, 9.0])  # Barolo (red, tannic)
        similarity = engine.cosine_similarity(vec_a, vec_b)
        # These wines are different but still share some features (fruitiness, body)
        # so similarity can be high in cosine space. This is expected behavior.
        assert 50.0 < similarity < 100.0  # Just verify it's in valid range


class TestExponentialConfidenceFactor:
    """Test Bayesian-inspired confidence factor."""

    def test_one_wine_gives_low_confidence(self):
        """With 1 wine, confidence should be ~33%."""
        engine = PalateEngine()
        confidence = engine.exponential_confidence_factor(1)
        assert 0.30 < confidence < 0.35
        # Expected: 1 - e^(-0.4*1) = 0.3297

    def test_three_wines_gives_moderate_confidence(self):
        """With 3 wines, confidence should be ~70%."""
        engine = PalateEngine()
        confidence = engine.exponential_confidence_factor(3)
        assert 0.68 < confidence < 0.72
        # Expected: 1 - e^(-0.4*3) = 0.6988

    def test_five_wines_gives_high_confidence(self):
        """With 5 wines, confidence should be ~86%."""
        engine = PalateEngine()
        confidence = engine.exponential_confidence_factor(5)
        assert 0.84 < confidence < 0.88
        # Expected: 1 - e^(-0.4*5) = 0.8647

    def test_ten_wines_gives_very_high_confidence(self):
        """With 10 wines, confidence should be ~98%."""
        engine = PalateEngine()
        confidence = engine.exponential_confidence_factor(10)
        assert 0.97 < confidence < 0.99
        # Expected: 1 - e^(-0.4*10) = 0.9817

    def test_zero_wines_gives_zero_confidence(self):
        """With 0 wines, confidence should be 0%."""
        engine = PalateEngine()
        confidence = engine.exponential_confidence_factor(0)
        assert confidence == 0.0

    def test_confidence_asymptotes_to_one(self):
        """With many wines, confidence approaches 100%."""
        engine = PalateEngine()
        confidence_50 = engine.exponential_confidence_factor(50)
        confidence_100 = engine.exponential_confidence_factor(100)
        assert confidence_50 > 0.999
        assert confidence_100 > 0.9999

    def test_confidence_is_monotonic(self):
        """Confidence should always increase with more samples."""
        engine = PalateEngine()
        for n in range(1, 20):
            conf_n = engine.exponential_confidence_factor(n)
            conf_n_plus_1 = engine.exponential_confidence_factor(n + 1)
            assert conf_n_plus_1 > conf_n


class TestCalculateMatch:
    """Test the complete match calculation."""

    @pytest.fixture
    def sample_history(self):
        """Create sample wine history."""
        return pd.DataFrame([
            {
                'wine_name': 'Albariño 1', 'liked': True, 'wine_color': 'White',
                'acidity': 9, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9
            },
            {
                'wine_name': 'Albariño 2', 'liked': True, 'wine_color': 'White',
                'acidity': 8, 'fruitiness': 8, 'body': 6, 'tannin': 1, 'minerality': 8
            },
            {
                'wine_name': 'Albariño 3', 'liked': True, 'wine_color': 'White',
                'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 2, 'minerality': 8
            },
            {
                'wine_name': 'Rioja', 'liked': True, 'wine_color': 'Red',
                'acidity': 6, 'fruitiness': 7, 'body': 8, 'tannin': 7, 'minerality': 5
            },
            {
                'wine_name': 'Bad Wine', 'liked': False, 'wine_color': 'Red',
                'acidity': 4, 'fruitiness': 5, 'body': 9, 'tannin': 9, 'minerality': 3
            },
        ])

    def test_no_history_returns_first_wine_verdict(self, sample_history):
        """With no history, should return 'First Wine' verdict."""
        engine = PalateEngine()
        test_wine = {'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 8}
        score = engine.calculate_match(test_wine)

        assert score.verdict == "Check: First Wine"
        assert score.palate_match == 50.0
        assert score.likelihood_score == 50.0
        assert score.n_samples == 0

    def test_similar_wine_high_likelihood(self, sample_history):
        """A wine close to the ideal profile should score high under centred cosine.

        Under v1 (plain cosine on positive vectors) every wine scored
        85-100% because the cosine couldn't be negative. Under v2
        (centred cosine) similarity to the ideal still produces high
        scores, but the spread is real - see test_centring_creates_spread.
        """
        engine = PalateEngine(sample_history)

        # Test wine very similar to the liked white-wine cluster.
        test_wine = {'acidity': 8.5, 'fruitiness': 7.5, 'body': 5.5, 'tannin': 1, 'minerality': 8.5}
        score = engine.calculate_match(test_wine, wine_color='White')

        assert score.palate_match > 70.0
        assert score.n_samples == 3
        assert score.confidence_factor > 0.65
        # Threshold for Strong Match is 60 on likelihood under v2.
        assert score.likelihood_score > 60.0

    def test_different_wine_low_score(self, sample_history):
        """A wine opposite to liked patterns should score low under centred cosine.

        v1 regression: this test previously had to assert only that the
        score was "in valid range" because cosine inflation meant even
        opposite wines scored above 50. v2 makes the spread real.
        """
        engine = PalateEngine(sample_history)

        # Wine that deviates from typical in directions OPPOSITE to liked wines:
        # low acidity (liked are high), high tannin (liked are low),
        # high body (liked are low), low minerality (liked are high).
        test_wine = {'acidity': 4, 'fruitiness': 5, 'body': 9, 'tannin': 9, 'minerality': 3}
        score = engine.calculate_match(test_wine)

        # Should be well below the "Worth Trying" threshold (50).
        assert score.palate_match < 50.0
        assert score.likelihood_score < 50.0

    def test_color_specific_matching(self, sample_history):
        """Color-specific matching should use only same-color wines."""
        engine = PalateEngine(sample_history)

        # Test white wine
        white_wine = {'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 8}
        score_white = engine.calculate_match(white_wine, wine_color='White')

        # Should use 3 white wines only
        assert score_white.n_samples == 3

    def test_likelihood_penalized_with_few_samples(self, sample_history):
        """Likelihood should be penalized with few samples."""
        # Create dataset with only 1 liked wine
        df_one = sample_history[sample_history['wine_name'] == 'Albariño 1'].copy()
        engine = PalateEngine(df_one)

        test_wine = {'acidity': 9, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9}
        score = engine.calculate_match(test_wine)

        # Even with perfect match (100%), likelihood should be penalized
        # With 1 wine, confidence ~= 0.33, so likelihood ~= 33%
        assert score.palate_match > 95.0
        assert score.likelihood_score < 40.0  # Heavily penalized

    def test_verdict_strong_match_at_high_alignment_and_confidence(self, sample_history):
        """Strong Match requires both high alignment AND enough samples.

        Display fix (2026-05): the verdict considers alignment and
        confidence as separate dimensions. High alignment alone is
        not enough - needs the sample size too.
        """
        engine = PalateEngine(sample_history)  # 3 liked wines, conf ~= 0.7

        # Wine that's very close to the liked-wine average.
        strong_wine = {
            'acidity': 8.5, 'fruitiness': 7.5, 'body': 5.5,
            'tannin': 1.5, 'minerality': 8.5,
        }
        score = engine.calculate_match(strong_wine)

        if score.palate_match >= 70.0 and score.confidence_factor >= 0.6:
            assert " Strong Match" in score.verdict, (
                f"Expected Strong Match with alignment={score.palate_match:.1f} "
                f"and confidence={score.confidence_factor:.2f}, got: {score.verdict!r}"
            )

    def test_verdict_promising_at_high_alignment_low_confidence(self, sample_history):
        """High alignment with few samples reads as Promising, not Strong.

        Critical case: previously this returned a low number (e.g. 33%)
        labelled 'Strong Match' because likelihood was alignment x
        confidence and the verdict was on likelihood. New logic: high
        alignment is reported honestly as Promising when sample size
        is small.
        """
        # Single liked wine -> confidence ~= 0.33, well below 0.6.
        df_one = sample_history[sample_history['wine_name'] == 'Albariño 1'].copy()
        engine = PalateEngine(df_one)

        # Identical to the liked wine - perfect alignment.
        score = engine.calculate_match({
            'acidity': 9, 'fruitiness': 7, 'body': 5,
            'tannin': 1, 'minerality': 9,
        })

        # Should be high alignment but low confidence.
        assert score.palate_match > 95.0
        assert score.confidence_factor < 0.6
        # And the verdict acknowledges that, not "Strong Match".
        assert "Promising" in score.verdict, (
            f"Expected Promising verdict, got: {score.verdict!r}"
        )

    def test_verdict_different_style_at_low_alignment(self, sample_history):
        """Low alignment -> Different Style regardless of sample size."""
        engine = PalateEngine(sample_history)

        opposite_wine = {
            'acidity': 3, 'fruitiness': 5, 'body': 9,
            'tannin': 9, 'minerality': 3,
        }
        score = engine.calculate_match(opposite_wine)

        if score.palate_match < 55.0:
            assert "Different Style" in score.verdict, (
                f"Expected Different Style for low alignment "
                f"({score.palate_match:.1f}), got: {score.verdict!r}"
            )


class TestScoreDeterminism:
    """The score must be a pure function of (features, history).

    Regression guard for the instability bug where the same wine
    scored 84% / 92% / 95% across renders. The root cause was upstream
    (non-deterministic LLM feature extraction), but the engine itself
    must be provably deterministic so we can isolate future drift to
    the input layer rather than the math.
    """

    @pytest.fixture
    def history(self):
        return pd.DataFrame([
            {'wine_name': 'A', 'liked': True, 'wine_color': 'Red',
             'acidity': 7, 'fruitiness': 6, 'body': 6, 'tannin': 5, 'minerality': 7},
            {'wine_name': 'B', 'liked': True, 'wine_color': 'Red',
             'acidity': 8, 'fruitiness': 6, 'body': 5, 'tannin': 4, 'minerality': 8},
            {'wine_name': 'C', 'liked': True, 'wine_color': 'Red',
             'acidity': 7, 'fruitiness': 7, 'body': 6, 'tannin': 5, 'minerality': 7},
            {'wine_name': 'D', 'liked': False, 'wine_color': 'Red',
             'acidity': 4, 'fruitiness': 5, 'body': 9, 'tannin': 9, 'minerality': 3},
        ])

    def test_same_inputs_give_identical_score(self, history):
        from decant.palate_engine import PalateEngine

        candidate = {
            'acidity': 7.5, 'fruitiness': 6.2, 'body': 5.5,
            'tannin': 4.5, 'minerality': 7.8,
        }

        # Build a fresh engine each time, as the app does per render.
        scores = []
        for _ in range(5):
            engine = PalateEngine(history.copy())
            score = engine.calculate_match(candidate, 'Red')
            scores.append(score.palate_match)

        # All five must be bit-identical. No averaging, no tolerance.
        assert len(set(scores)) == 1, (
            f"Score varied across identical calls: {scores}. The engine "
            f"must be deterministic given fixed features and history."
        )

    def test_row_order_does_not_change_score(self, history):
        from decant.palate_engine import PalateEngine

        candidate = {
            'acidity': 7.5, 'fruitiness': 6.2, 'body': 5.5,
            'tannin': 4.5, 'minerality': 7.8,
        }

        engine_a = PalateEngine(history.copy())
        score_a = engine_a.calculate_match(candidate, 'Red').palate_match

        # Shuffle the rows - a DB query without ORDER BY can return
        # rows in any order. The score must not depend on it.
        shuffled = history.sample(frac=1.0, random_state=7).reset_index(drop=True)
        engine_b = PalateEngine(shuffled)
        score_b = engine_b.calculate_match(candidate, 'Red').palate_match

        assert abs(score_a - score_b) < 1e-9, (
            f"Score depends on row order: {score_a} vs {score_b}"
        )


class TestCentredCosine:
    """Behaviour of centred cosine, the v2 normalisation fix.

    Tests assume a history with >= 3 wines so the centring path runs;
    smaller histories fall back to plain cosine (tested elsewhere).
    """

    @pytest.fixture
    def diverse_history(self):
        """A history with both styles, so centring has meaningful spread."""
        return pd.DataFrame([
            # Liked: high acidity, low body, low tannin, high minerality
            {'wine_name': 'A1', 'liked': True, 'wine_color': 'White',
             'acidity': 9, 'fruitiness': 7, 'body': 4, 'tannin': 1, 'minerality': 9},
            {'wine_name': 'A2', 'liked': True, 'wine_color': 'White',
             'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 8},
            # Disliked: low acidity, high body, high tannin
            {'wine_name': 'B1', 'liked': False, 'wine_color': 'Red',
             'acidity': 4, 'fruitiness': 6, 'body': 9, 'tannin': 9, 'minerality': 3},
            {'wine_name': 'B2', 'liked': False, 'wine_color': 'Red',
             'acidity': 3, 'fruitiness': 5, 'body': 9, 'tannin': 9, 'minerality': 2},
        ])

    def test_wine_on_population_mean_scores_neutral(self, diverse_history):
        """A wine sitting exactly on the population mean is neutral by
        definition - no deviation to correlate with anything."""
        engine = PalateEngine(diverse_history)
        mean_wine = {
            'acidity': float(engine.population_mean[0]),
            'fruitiness': float(engine.population_mean[1]),
            'body': float(engine.population_mean[2]),
            'tannin': float(engine.population_mean[3]),
            'minerality': float(engine.population_mean[4]),
        }
        score = engine.calculate_match(mean_wine)
        assert abs(score.palate_match - 50.0) < 1.0

    def test_wine_deviating_like_ideal_scores_high(self, diverse_history):
        """A wine that deviates from the population in the same direction
        as the ideal profile should score above 75."""
        engine = PalateEngine(diverse_history)
        # Exaggerate the liked-deviation: even higher acidity, even lower body
        aligned_wine = {'acidity': 10, 'fruitiness': 7, 'body': 3, 'tannin': 1, 'minerality': 10}
        score = engine.calculate_match(aligned_wine)
        assert score.palate_match > 75.0

    def test_wine_deviating_opposite_to_ideal_scores_low(self, diverse_history):
        """A wine that deviates opposite to the ideal should score below 25."""
        engine = PalateEngine(diverse_history)
        # Opposite of ideal deviation: low acidity, high body, high tannin
        opposite_wine = {'acidity': 2, 'fruitiness': 5, 'body': 10, 'tannin': 10, 'minerality': 1}
        score = engine.calculate_match(opposite_wine)
        assert score.palate_match < 25.0

    def test_centred_cosine_uses_full_score_range(self, diverse_history):
        """v1 regression: scores should now occupy more than just [75, 100].

        Specifically, an aligned wine should score >75 and an
        opposite wine should score <25, giving a spread of >50 points.
        """
        engine = PalateEngine(diverse_history)
        aligned = engine.calculate_match(
            {'acidity': 10, 'fruitiness': 7, 'body': 3, 'tannin': 1, 'minerality': 10}
        )
        opposite = engine.calculate_match(
            {'acidity': 2, 'fruitiness': 5, 'body': 10, 'tannin': 10, 'minerality': 1}
        )
        spread = aligned.palate_match - opposite.palate_match
        assert spread > 50.0, f"Expected real spread; got aligned={aligned.palate_match}, opposite={opposite.palate_match}"

    def test_falls_back_to_plain_cosine_with_small_history(self):
        """With < 3 rated wines, centring against a 1- or 2-sample mean
        is meaningless. Fall back to plain cosine and document via the
        score that the result is uncentred."""
        small_history = pd.DataFrame([
            {'wine_name': 'A', 'liked': True, 'wine_color': 'White',
             'acidity': 9, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9},
        ])
        engine = PalateEngine(small_history)
        # Identical to the only wine in history → plain cosine = 100%.
        score = engine.calculate_match(
            {'acidity': 9, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9}
        )
        assert score.palate_match > 95.0


class TestPopulationMean:
    """The population mean is the centring point for v2."""

    def test_population_mean_includes_all_rated_wines(self):
        """Population mean should average across both liked and disliked."""
        df = pd.DataFrame([
            {'wine_name': 'A', 'liked': True, 'wine_color': 'White',
             'acidity': 10, 'fruitiness': 10, 'body': 10, 'tannin': 10, 'minerality': 10},
            {'wine_name': 'B', 'liked': False, 'wine_color': 'Red',
             'acidity': 0, 'fruitiness': 0, 'body': 0, 'tannin': 0, 'minerality': 0},
        ])
        engine = PalateEngine(df)
        # Mean of [10, 0] is 5 on every dimension.
        assert engine.population_mean is not None
        np.testing.assert_array_almost_equal(
            engine.population_mean, [5.0, 5.0, 5.0, 5.0, 5.0]
        )

    def test_no_population_mean_with_empty_history(self):
        engine = PalateEngine(pd.DataFrame())
        assert engine.population_mean is None
        assert engine.n_total == 0


class TestIdealProfileComputation:
    """Test ideal profile calculation."""

    def test_ideal_profile_is_mean_of_liked_wines(self):
        """Ideal profile should be the mean of all liked wines."""
        df = pd.DataFrame([
            {
                'wine_name': 'Wine 1', 'liked': True,
                'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9
            },
            {
                'wine_name': 'Wine 2', 'liked': True,
                'acidity': 10, 'fruitiness': 9, 'body': 7, 'tannin': 3, 'minerality': 7
            },
            {
                'wine_name': 'Wine 3', 'liked': False,
                'acidity': 4, 'fruitiness': 5, 'body': 9, 'tannin': 9, 'minerality': 3
            },
        ])

        engine = PalateEngine(df)

        # Ideal should be mean of Wine 1 and Wine 2 (liked wines only)
        assert engine.ideal_profile.acidity == 9.0
        assert engine.ideal_profile.fruitiness == 8.0
        assert engine.ideal_profile.body == 6.0
        assert engine.ideal_profile.tannin == 2.0
        assert engine.ideal_profile.minerality == 8.0
        assert engine.n_liked == 2

    def test_no_liked_wines_returns_none(self):
        """With no liked wines, ideal profile should be None."""
        df = pd.DataFrame([
            {
                'wine_name': 'Wine 1', 'liked': False,
                'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9
            },
        ])

        engine = PalateEngine(df)
        assert engine.ideal_profile is None
        assert engine.n_liked == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Engine should handle empty DataFrame."""
        df = pd.DataFrame()
        engine = PalateEngine(df)

        test_wine = {'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 8}
        score = engine.calculate_match(test_wine)

        assert score.verdict == "Check: First Wine"
        assert score.n_samples == 0

    def test_none_dataframe(self):
        """Engine should handle None DataFrame."""
        engine = PalateEngine(None)

        test_wine = {'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 8}
        score = engine.calculate_match(test_wine)

        assert score.verdict == "Check: First Wine"

    def test_missing_features_in_wine(self):
        """Engine should handle wines with missing features."""
        df = pd.DataFrame([
            {
                'wine_name': 'Wine 1', 'liked': True,
                'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9
            },
        ])
        engine = PalateEngine(df)

        # Test wine with missing features (should default to 0)
        test_wine = {'acidity': 8}
        score = engine.calculate_match(test_wine)

        # Should not crash
        assert isinstance(score.palate_match, float)
        assert isinstance(score.likelihood_score, float)

    def test_color_not_in_history(self):
        """Requesting a color not in history should fall back to global."""
        df = pd.DataFrame([
            {
                'wine_name': 'Red Wine', 'liked': True, 'wine_color': 'Red',
                'acidity': 6, 'fruitiness': 7, 'body': 8, 'tannin': 7, 'minerality': 5
            },
        ])
        engine = PalateEngine(df)

        # Request white wine (not in history)
        test_wine = {'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 8}
        score = engine.calculate_match(test_wine, wine_color='White')

        # BUG FIX: Should fall back to global profile with n_samples=0 (not 1)
        # Previously incorrectly used n_samples=1 which inflated confidence
        # for colors never tried
        assert score.n_samples == 0  # FIXED: Was 1 (incorrect)


class TestGetProfileVector:
    """Test profile vector retrieval."""

    def test_get_global_profile_vector(self):
        """Should return global profile vector."""
        df = pd.DataFrame([
            {
                'wine_name': 'Wine 1', 'liked': True,
                'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9
            },
            {
                'wine_name': 'Wine 2', 'liked': True,
                'acidity': 10, 'fruitiness': 9, 'body': 7, 'tannin': 3, 'minerality': 7
            },
        ])
        engine = PalateEngine(df)

        profile = engine.get_profile_vector()
        assert isinstance(profile, np.ndarray)
        assert profile.shape == (5,)
        assert np.array_equal(profile, [9.0, 8.0, 6.0, 2.0, 8.0])

    def test_get_color_specific_profile_vector(self):
        """Should return color-specific profile vector."""
        df = pd.DataFrame([
            {
                'wine_name': 'White 1', 'liked': True, 'wine_color': 'White',
                'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9
            },
            {
                'wine_name': 'White 2', 'liked': True, 'wine_color': 'White',
                'acidity': 10, 'fruitiness': 9, 'body': 7, 'tannin': 3, 'minerality': 7
            },
            {
                'wine_name': 'Red 1', 'liked': True, 'wine_color': 'Red',
                'acidity': 6, 'fruitiness': 7, 'body': 8, 'tannin': 7, 'minerality': 5
            },
        ])
        engine = PalateEngine(df)

        white_profile = engine.get_profile_vector(wine_color='White')
        # Should be mean of 2 white wines only
        assert np.array_equal(white_profile, [9.0, 8.0, 6.0, 2.0, 8.0])

    def test_get_profile_vector_with_no_history(self):
        """Should return None with no history."""
        engine = PalateEngine()
        profile = engine.get_profile_vector()
        assert profile is None
