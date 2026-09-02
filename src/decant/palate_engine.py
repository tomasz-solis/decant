"""Wine preference matching engine using cosine similarity and confidence decay."""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
import math
import json

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


_APP_FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"
_CHART_TEXT = "#211A16"
_CHART_MUTED = "#7F7568"
_CHART_GRID = "rgba(33, 26, 22, 0.12)"
_CHART_CARD = "#FFFCF6"
_CHART_TRANSPARENT = "rgba(0, 0, 0, 0)"
_CHART_WINE = "#7A1730"
_CHART_WINE_FILL = "rgba(122, 23, 48, 0.18)"


@dataclass
class WineVector:
    """5-dimensional wine feature vector (1-10 scale)"""
    acidity: float
    fruitiness: float
    body: float
    tannin: float
    minerality: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for calculations"""
        return np.array([
            self.acidity,
            self.fruitiness,
            self.body,
            self.tannin,
            self.minerality
        ])

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'WineVector':
        """Create from dictionary"""
        return cls(
            acidity=data.get('acidity', 0),
            fruitiness=data.get('fruitiness', 0),
            body=data.get('body', 0),
            tannin=data.get('tannin', 0),
            minerality=data.get('minerality', 0)
        )


@dataclass
class PalateScore:
    """Dual-metric scoring system for wine matching"""
    palate_match: float  # Cosine similarity (0-100%)
    likelihood_score: float  # Bayesian-adjusted (0-100%)
    n_samples: int  # Number of liked wines used
    confidence_factor: float  # Bayesian penalty factor (0-1)
    verdict: str  # Human-readable verdict
    explanation: str  # Why this score


class PalateEngine:
    """Calculates wine match scores using cosine similarity and confidence decay."""

    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        """Initialise the PalateEngine.

        Args:
            history_df: DataFrame with wine history. Required columns:
                wine_name, liked, wine_color,
                acidity, fruitiness, body, tannin, minerality.
        """
        self.history_df = history_df
        self.feature_cols = ['acidity', 'fruitiness', 'body', 'tannin', 'minerality']

        # Ideal profile: mean of liked wines (the "what you tend to enjoy" vector).
        self.ideal_profile: Optional[WineVector] = None
        self.n_liked: int = 0

        # Population mean: mean of *all* rated wines, used as the centre point
        # for centred cosine. With centred cosine, similarity measures whether
        # a wine deviates from typical in the same direction as your liked
        # wines, rather than whether the raw vectors point in similar directions.
        # See docs/algorithm_v2.md for the rationale.
        self.population_mean: Optional[np.ndarray] = None
        self.n_total: int = 0

        if history_df is not None:
            self._compute_ideal_profile()

    def _compute_ideal_profile(self) -> None:
        """Compute the ideal-profile vector (mean of liked wines) and the
        population mean (mean of all rated wines).

        The population mean is what makes centred cosine work. Without it,
        cosine between positive-only vectors is bounded above 0 and the
        score range degenerates.
        """
        if self.history_df is None or len(self.history_df) == 0:
            return

        # Population mean across everything rated (both liked and disliked).
        self.n_total = len(self.history_df)
        if self.n_total > 0:
            pop_vals = self.history_df[self.feature_cols].mean()
            self.population_mean = np.array([
                pop_vals['acidity'],
                pop_vals['fruitiness'],
                pop_vals['body'],
                pop_vals['tannin'],
                pop_vals['minerality'],
            ])

        liked_wines = self.history_df[self.history_df['liked'].eq(True)]
        self.n_liked = len(liked_wines)

        if self.n_liked > 0:
            mean_vals = liked_wines[self.feature_cols].mean()
            self.ideal_profile = WineVector(
                acidity=mean_vals['acidity'],
                fruitiness=mean_vals['fruitiness'],
                body=mean_vals['body'],
                tannin=mean_vals['tannin'],
                minerality=mean_vals['minerality'],
            )

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Plain cosine similarity, mapped from [-1, 1] to [0, 100].

        WARNING: For all-positive vectors (e.g. wine features in [1, 10]),
        cosine is bounded approximately in [0.5, 1], which makes this
        mapping produce scores compressed in [75, 100]. Use
        `_centred_cosine` instead for matching against user profiles.
        This method is kept because it's useful when comparing arbitrary
        vectors (e.g., color similarities derived from histograms) where
        the [-1, 1] range is genuinely available.
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        normalized = ((similarity + 1) / 2) * 100
        return max(0, min(100, normalized))

    def _centred_cosine(
        self,
        vec_current: np.ndarray,
        vec_ideal: np.ndarray,
    ) -> float:
        """Centred cosine similarity, mapped to [0, 100].

        Both vectors are centred against `self.population_mean` before
        the cosine is computed. The resulting cosine genuinely uses the
        full [-1, 1] range:
        - 1   = current wine deviates from population mean in the same
                direction as the ideal profile (strong match)
        - 0   = no correlation between the deviations (neutral)
        - -1  = current wine deviates in the opposite direction (anti-match)

        Falls back to plain `cosine_similarity` if the population mean
        isn't available (fewer than 3 rated wines), since centring
        against a noisy 1- or 2-sample mean adds noise rather than
        removing it.
        """
        if self.population_mean is None or self.n_total < 3:
            return self.cosine_similarity(vec_current, vec_ideal)

        centred_current = vec_current - self.population_mean
        centred_ideal = vec_ideal - self.population_mean

        norm_c = np.linalg.norm(centred_current)
        norm_i = np.linalg.norm(centred_ideal)

        # If either centred vector is the zero vector, the wine sits
        # exactly on the population mean - neutral by construction.
        if norm_c == 0 or norm_i == 0:
            return 50.0

        cos = np.dot(centred_current, centred_ideal) / (norm_c * norm_i)
        normalized = ((cos + 1) / 2) * 100
        return max(0.0, min(100.0, normalized))

    def exponential_confidence_factor(self, n_samples: int) -> float:
        """Confidence factor: 1 - e^(-α * N) where α = 0.4."""
        from decant.constants import AlgorithmConstants
        return 1 - math.exp(-AlgorithmConstants.EXPONENTIAL_ALPHA * n_samples)

    def bayesian_confidence_factor(self, n_samples: int) -> float:
        """Deprecated: use exponential_confidence_factor() instead."""
        import warnings
        warnings.warn(
            "Use exponential_confidence_factor() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.exponential_confidence_factor(n_samples)

    def calculate_match(
        self,
        wine_features: Dict[str, float],
        wine_color: Optional[str] = None,
    ) -> PalateScore:
        """Calculate a match score for a wine against the user's ideal profile.

        Uses centred cosine similarity (see `_centred_cosine`) to avoid the
        baseline inflation that plain cosine produces on positive-only
        feature vectors. Scores are then multiplied by the exponential
        confidence factor based on history size.

        For color-specific matching, the color-filtered liked-wine mean is
        used as the ideal vector. When no liked wine exists for the
        requested color, the global ideal is used with `n_samples=0` so
        the confidence factor reflects that we have no evidence for this
        colour, not the (irrelevant) total count of liked wines.
        """
        current_wine = WineVector.from_dict(wine_features)
        current_vec = current_wine.to_array()

        if self.ideal_profile is None or self.n_liked == 0:
            return PalateScore(
                palate_match=50.0,
                likelihood_score=50.0,
                n_samples=0,
                confidence_factor=0.0,
                verdict="First Wine",
                explanation="No history yet - this will establish your baseline",
            )

        if wine_color and self.history_df is not None:
            color_liked = self.history_df[
                (self.history_df['liked'].eq(True))
                & (self.history_df['wine_color'] == wine_color)
            ]

            if len(color_liked) > 0:
                color_mean = color_liked[self.feature_cols].mean()
                ideal_vec = np.array([
                    color_mean['acidity'],
                    color_mean['fruitiness'],
                    color_mean['body'],
                    color_mean['tannin'],
                    color_mean['minerality'],
                ])
                n_samples = len(color_liked)
            else:
                # No liked wine of this colour yet. We can still compute a
                # similarity against the global profile, but confidence
                # must be zero - we have no evidence for this colour.
                ideal_vec = self.ideal_profile.to_array()
                n_samples = 0
        else:
            ideal_vec = self.ideal_profile.to_array()
            n_samples = self.n_liked

        palate_match = self._centred_cosine(current_vec, ideal_vec)
        confidence_factor = self.exponential_confidence_factor(n_samples)
        likelihood_score = palate_match * confidence_factor

        # Verdict considers alignment and confidence as two separate
        # facts, not as a multiplication. This matches what the UI
        # shows the user and avoids the "65% Strong Match" confusion
        # where a high-alignment wine reads as a weak match purely
        # because the user hasn't rated enough wines yet.
        #
        # `palate_match` is the centred-cosine output mapped to
        # [0, 100]. 50 corresponds to neutral (zero alignment with
        # the liked-deviation pattern); 70 corresponds to centred
        # cosine ~0.4 - solid positive alignment.
        #
        # `confidence_factor` is exponential, hitting ~0.6 at n=3
        # liked wines (the "we have enough data to commit" threshold).
        STRONG_ALIGNMENT = 70.0
        EXPLORE_ALIGNMENT = 55.0
        CONFIDENT_ENOUGH = 0.6

        if palate_match >= STRONG_ALIGNMENT and confidence_factor >= CONFIDENT_ENOUGH:
            verdict = "Strong Match"
            explanation = (
                f"High flavor alignment ({palate_match:.0f}%) with "
                f"strong confidence ({n_samples} wines)"
            )
        elif palate_match >= STRONG_ALIGNMENT:
            # High alignment but not enough data: call it Promising
            # rather than Strong, so the verdict copy doesn't oversell
            # what's still an early signal.
            verdict = "Promising - rate more to confirm"
            explanation = (
                f"High flavor alignment ({palate_match:.0f}%), but "
                f"only {n_samples} wine(s) rated so far"
            )
        elif palate_match >= EXPLORE_ALIGNMENT:
            verdict = "Worth Exploring"
            explanation = (
                f"Moderate alignment ({palate_match:.0f}%) - could "
                f"go either way"
            )
        else:
            verdict = "Different Style"
            explanation = (
                f"Low alignment ({palate_match:.0f}%) - departure "
                f"from your usual profile"
            )

        return PalateScore(
            palate_match=round(palate_match, 1),
            likelihood_score=round(likelihood_score, 1),
            n_samples=n_samples,
            confidence_factor=round(confidence_factor, 2),
            verdict=verdict,
            explanation=explanation,
        )

    def get_profile_vector(self, wine_color: Optional[str] = None) -> Optional[np.ndarray]:
        """Get the ideal profile vector, optionally filtered by wine color."""
        if self.ideal_profile is None:
            return None

        if wine_color and self.history_df is not None:
            color_liked = self.history_df[
                (self.history_df['liked'].eq(True)) &
                (self.history_df['wine_color'] == wine_color)
            ]

            if len(color_liked) > 0:
                color_mean = color_liked[self.feature_cols].mean()
                return np.array([
                    color_mean['acidity'],
                    color_mean['fruitiness'],
                    color_mean['body'],
                    color_mean['tannin'],
                    color_mean['minerality']
                ])

        return self.ideal_profile.to_array()

    def explain_metrics(self) -> Dict[str, str]:
        """Return metric name → description mapping."""
        return {
            "Flavor Profile Alignment": "Cosine similarity between this wine and your ideal profile (0-100%).",
            "Match Likelihood": "Flavor alignment adjusted by confidence from sample size (0-100%)."
        }

    def get_confidence_breakdown(self, n_samples: int) -> Dict[str, float]:
        """Return confidence factors for various sample sizes."""
        sample_sizes = [1, 2, 3, 5, 10, 15, 20, 30, 50]
        return {
            f"{n}_wines": round(self.bayesian_confidence_factor(n), 3)
            for n in sample_sizes
        }

    def generate_radar_chart(
        self,
        current_wine: Dict[str, float],
        wine_color: Optional[str] = None
    ):
        """Generate Plotly radar chart comparing current wine to ideal profile."""
        if not PLOTLY_AVAILABLE:
            return None

        if self.ideal_profile is None:
            return None

        # Get ideal profile vector
        ideal_vec = self.get_profile_vector(wine_color)
        if ideal_vec is None:
            return None

        # Create figure
        fig = go.Figure()

        categories = ['Acidity', 'Fruitiness', 'Body', 'Tannin', 'Minerality']

        # Current wine vector
        current_vec = [
            current_wine.get('acidity', 0),
            current_wine.get('fruitiness', 0),
            current_wine.get('body', 0),
            current_wine.get('tannin', 0),
            current_wine.get('minerality', 0)
        ]

        # Close the polygons
        current_vec_closed = current_vec + [current_vec[0]]
        ideal_vec_closed = ideal_vec.tolist() + [ideal_vec[0]]

        # USER'S IDEAL TARGET - soft wine filled area
        fig.add_trace(go.Scatterpolar(
            r=ideal_vec_closed,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=_CHART_WINE_FILL,
            line=dict(color=_CHART_WINE, width=2.4),
            name="Your Ideal Target",
            marker=dict(size=6, symbol='circle', color=_CHART_WINE)
        ))

        # CURRENT WINE - ink solid line
        fig.add_trace(go.Scatterpolar(
            r=current_vec_closed,
            theta=categories + [categories[0]],
            fill='none',
            line=dict(color=_CHART_TEXT, width=3),
            name="Current Wine",
            marker=dict(size=7, symbol='circle', color=_CHART_TEXT)
        ))

        # Layout with 1-10 scale
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2,
                    showticklabels=True,
                    tickfont=dict(size=11, family=_APP_FONT_FAMILY, color=_CHART_MUTED),
                    gridcolor=_CHART_GRID,
                    linecolor=_CHART_GRID,
                ),
                angularaxis=dict(
                    tickfont=dict(size=13, family=_APP_FONT_FAMILY, color=_CHART_TEXT),
                    gridcolor=_CHART_GRID,
                    linecolor=_CHART_GRID,
                ),
                bgcolor=_CHART_TRANSPARENT
            ),
            showlegend=True,
            title=dict(
                text='Flavor Profile Comparison',
                font=dict(size=16, family=_APP_FONT_FAMILY, color=_CHART_TEXT),
                x=0,
                xanchor='left'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family=_APP_FONT_FAMILY, color=_CHART_TEXT)
            ),
            height=500,
            font=dict(family=_APP_FONT_FAMILY, color=_CHART_TEXT),
            paper_bgcolor=_CHART_TRANSPARENT,
            margin=dict(t=64, b=92, l=64, r=64),
            hoverlabel=dict(
                bgcolor=_CHART_TEXT,
                font=dict(family=_APP_FONT_FAMILY, color=_CHART_CARD),
                bordercolor=_CHART_TEXT,
            ),
        )

        return fig

    def get_ui_data(
        self,
        wine_features: Dict[str, float],
        wine_color: Optional[str] = None,
        include_chart: bool = True
    ) -> Dict:
        """Return all match data needed for UI rendering."""
        # Calculate match score
        score = self.calculate_match(wine_features, wine_color)

        ui_data = {
            "match_likelihood": score.likelihood_score,
            "flavor_alignment": score.palate_match,
            "confidence_score": score.confidence_factor * 100,
            "n_samples": score.n_samples,
            "verdict": score.verdict,
            "explanation": score.explanation,
            "wine_color": wine_color,
            "metric_definitions": self.explain_metrics()
        }

        # Add radar chart if requested
        if include_chart and PLOTLY_AVAILABLE:
            chart = self.generate_radar_chart(wine_features, wine_color)
            if chart:
                ui_data["plotly_chart"] = chart
                ui_data["plotly_json"] = chart.to_json()

        return ui_data

    def to_json(
        self,
        wine_features: Dict[str, float],
        wine_color: Optional[str] = None
    ) -> str:
        """
        Get UI data as JSON string

        Returns:
            JSON string with all UI data (excluding Plotly objects)
        """
        ui_data = self.get_ui_data(wine_features, wine_color, include_chart=False)
        return json.dumps(ui_data, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Example: Create sample history
    sample_data = pd.DataFrame([
        {'wine_name': 'Albariño 1', 'liked': True, 'wine_color': 'White',
         'acidity': 9, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9},
        {'wine_name': 'Albariño 2', 'liked': True, 'wine_color': 'White',
         'acidity': 8, 'fruitiness': 8, 'body': 6, 'tannin': 1, 'minerality': 8},
        {'wine_name': 'Rioja', 'liked': True, 'wine_color': 'Red',
         'acidity': 6, 'fruitiness': 7, 'body': 8, 'tannin': 7, 'minerality': 5},
    ])

    # Initialize engine
    engine = PalateEngine(sample_data)

    # Test wine (similar to liked Albariños)
    test_wine = {
        'acidity': 8.5,
        'fruitiness': 7.5,
        'body': 5.5,
        'tannin': 1,
        'minerality': 8.5
    }

    # Calculate match
    score = engine.calculate_match(test_wine, wine_color='White')

    print("PalateEngine Test")
    print(f"Flavor Profile Alignment: {score.palate_match}%")
    print(f"Match Likelihood: {score.likelihood_score}%")
    print(f"Confidence Factor: {score.confidence_factor} (based on {score.n_samples} wines)")
    print(f"Verdict: {score.verdict}")
    print(f"Explanation: {score.explanation}")

    print("\n=== Confidence Breakdown ===")
    for size, conf in engine.get_confidence_breakdown(20).items():
        print(f"{size}: {conf*100:.1f}%")
