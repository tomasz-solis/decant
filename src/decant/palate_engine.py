"""Wine preference matching engine using cosine similarity and confidence decay."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import math
import json

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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
        """
        Initialize the PalateEngine

        Args:
            history_df: DataFrame with wine history (columns: wine_name, liked,
                       acidity, fruitiness, body, tannin, minerality)
        """
        self.history_df = history_df
        self.feature_cols = ['acidity', 'fruitiness', 'body', 'tannin', 'minerality']

        # Dynamic Ideal Profile (Mean of all liked wines)
        self.ideal_profile: Optional[WineVector] = None
        self.n_liked: int = 0

        if history_df is not None:
            self._compute_ideal_profile()

    def _compute_ideal_profile(self):
        """Compute the Dynamic Ideal Profile (mean of all liked wines)"""
        if self.history_df is None or len(self.history_df) == 0:
            return

        liked_wines = self.history_df[self.history_df['liked'] == True]
        self.n_liked = len(liked_wines)

        if self.n_liked > 0:
            # Calculate mean vector
            mean_vals = liked_wines[self.feature_cols].mean()
            self.ideal_profile = WineVector(
                acidity=mean_vals['acidity'],
                fruitiness=mean_vals['fruitiness'],
                body=mean_vals['body'],
                tannin=mean_vals['tannin'],
                minerality=mean_vals['minerality']
            )

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity normalized to 0-100 scale."""
        # Handle zero vectors
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)

        # Normalize to 0-100 scale (similarity ranges from -1 to 1)
        # We use (similarity + 1) / 2 to map [-1, 1] → [0, 1]
        normalized = ((similarity + 1) / 2) * 100

        return max(0, min(100, normalized))

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
        wine_color: Optional[str] = None
    ) -> PalateScore:
        """Calculate match score for a wine against user's ideal profile."""
        # Convert to vector
        current_wine = WineVector.from_dict(wine_features)
        current_vec = current_wine.to_array()

        # Check if we have history
        if self.ideal_profile is None or self.n_liked == 0:
            return PalateScore(
                palate_match=50.0,
                likelihood_score=50.0,
                n_samples=0,
                confidence_factor=0.0,
                verdict="🔍 First Wine",
                explanation="No history yet - this will establish your baseline"
            )

        # Color-specific matching (if requested)
        if wine_color and self.history_df is not None:
            color_liked = self.history_df[
                (self.history_df['liked'] == True) &
                (self.history_df['wine_color'] == wine_color)
            ]

            if len(color_liked) > 0:
                # Use color-specific profile
                color_mean = color_liked[self.feature_cols].mean()
                ideal_vec = np.array([
                    color_mean['acidity'],
                    color_mean['fruitiness'],
                    color_mean['body'],
                    color_mean['tannin'],
                    color_mean['minerality']
                ])
                n_samples = len(color_liked)
            else:
                # Fall back to global profile BUT with n_samples=0
                # BUG FIX: Previously used self.n_liked which inflated confidence
                # for colors never tried. Now correctly uses 0 samples.
                ideal_vec = self.ideal_profile.to_array()
                n_samples = 0  # FIXED: Was self.n_liked (incorrect)
        else:
            # Use global profile
            ideal_vec = self.ideal_profile.to_array()
            n_samples = self.n_liked

        palate_match = self.cosine_similarity(current_vec, ideal_vec)
        confidence_factor = self.exponential_confidence_factor(n_samples)
        likelihood_score = palate_match * confidence_factor

        # Generate verdict based on likelihood (not raw similarity)
        if likelihood_score >= 75:
            verdict = "💙 Strong Match"
            explanation = f"High flavor alignment ({palate_match:.0f}%) with strong confidence ({n_samples} wines)"
        elif likelihood_score >= 60:
            verdict = "🧡 Worth Trying"
            explanation = f"Good alignment ({palate_match:.0f}%), moderate confidence ({n_samples} wines)"
        elif likelihood_score >= 45:
            verdict = "🟡 Explore"
            explanation = f"Moderate alignment ({palate_match:.0f}%), building confidence ({n_samples} wines)"
        else:
            verdict = "⚪ Different Style"
            explanation = f"Low alignment ({palate_match:.0f}%) - departure from your usual profile"

        return PalateScore(
            palate_match=round(palate_match, 1),
            likelihood_score=round(likelihood_score, 1),
            n_samples=n_samples,
            confidence_factor=round(confidence_factor, 2),
            verdict=verdict,
            explanation=explanation
        )

    def get_profile_vector(self, wine_color: Optional[str] = None) -> Optional[np.ndarray]:
        """Get the ideal profile vector, optionally filtered by wine color."""
        if self.ideal_profile is None:
            return None

        if wine_color and self.history_df is not None:
            color_liked = self.history_df[
                (self.history_df['liked'] == True) &
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

        # USER'S IDEAL TARGET - Red/Pink filled area
        fig.add_trace(go.Scatterpolar(
            r=ideal_vec_closed,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(255, 100, 150, 0.3)',  # Red/Pink with transparency
            line=dict(color='rgb(220, 50, 100)', width=2),
            name="Your Ideal Target",
            marker=dict(size=6, color='rgb(220, 50, 100)')
        ))

        # CURRENT WINE - Black solid line
        fig.add_trace(go.Scatterpolar(
            r=current_vec_closed,
            theta=categories + [categories[0]],
            fill='none',
            line=dict(color='black', width=4),
            name="Current Wine",
            marker=dict(size=8, symbol='star', color='black')
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
                    tickfont=dict(size=12),
                    gridcolor='rgba(128, 128, 128, 0.2)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=14, family='Arial', color='#1A202C')
                ),
                bgcolor='rgba(250, 250, 250, 0.5)'
            ),
            showlegend=True,
            title=dict(
                text='<b>Flavor Profile Comparison</b>',
                font=dict(size=16, color='#1A202C'),
                x=0.5,
                xanchor='center'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            height=500,
            paper_bgcolor='white'
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

    print("=== PalateEngine Test ===")
    print(f"Flavor Profile Alignment: {score.palate_match}%")
    print(f"Match Likelihood: {score.likelihood_score}%")
    print(f"Confidence Factor: {score.confidence_factor} (based on {score.n_samples} wines)")
    print(f"Verdict: {score.verdict}")
    print(f"Explanation: {score.explanation}")

    print("\n=== Confidence Breakdown ===")
    for size, conf in engine.get_confidence_breakdown(20).items():
        print(f"{size}: {conf*100:.1f}%")
