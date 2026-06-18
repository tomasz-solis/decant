"""Decant - Taste, with confidence. A wine analytics and recommendation platform."""

__version__ = "0.1.0"

__all__ = ['VinoPredictor', 'PalateMatch', 'WineFeatures', '__version__']


def __getattr__(name: str):
    """Lazy-load OpenAI-backed predictor classes on demand."""
    if name in {'VinoPredictor', 'PalateMatch'}:
        from decant.predictor import PalateMatch, VinoPredictor

        globals()['VinoPredictor'] = VinoPredictor
        globals()['PalateMatch'] = PalateMatch
        return globals()[name]

    if name == 'WineFeatures':
        from decant.schema import WineFeatures

        globals()['WineFeatures'] = WineFeatures
        return WineFeatures

    raise AttributeError(f"module 'decant' has no attribute {name!r}")
