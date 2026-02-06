"""
Tests for Pydantic schemas.

Validates that data models enforce correct constraints.
"""

import pytest
from pydantic import ValidationError
from decant.schema import WineFeatures, WineExtraction


class TestWineFeatures:
    """Test WineFeatures schema validation."""

    def test_valid_features(self):
        """Valid features should pass validation."""
        features = WineFeatures(
            acidity=8.0,
            minerality=7.5,
            fruitiness=7.0,
            tannin=1.5,
            body=5.5,
            reasoning="Test wine with good acidity"
        )
        assert features.acidity == 8.0
        assert features.minerality == 7.5

    def test_features_within_range(self):
        """All features must be between 1.0 and 10.0."""
        # Test minimum
        features_min = WineFeatures(
            acidity=1.0,
            minerality=1.0,
            fruitiness=1.0,
            tannin=1.0,
            body=1.0
        )
        assert features_min.acidity == 1.0

        # Test maximum
        features_max = WineFeatures(
            acidity=10.0,
            minerality=10.0,
            fruitiness=10.0,
            tannin=10.0,
            body=10.0
        )
        assert features_max.acidity == 10.0

    def test_acidity_below_minimum_raises_error(self):
        """Acidity below 1.0 should raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            WineFeatures(
                acidity=0.5,
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )
        assert "acidity" in str(exc_info.value)

    def test_acidity_above_maximum_raises_error(self):
        """Acidity above 10.0 should raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            WineFeatures(
                acidity=10.5,
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )
        assert "acidity" in str(exc_info.value)

    def test_minerality_validation(self):
        """Minerality must be in valid range."""
        with pytest.raises(ValidationError):
            WineFeatures(
                acidity=8.0,
                minerality=0.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )

        with pytest.raises(ValidationError):
            WineFeatures(
                acidity=8.0,
                minerality=11.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )

    def test_missing_required_field_raises_error(self):
        """Missing required fields should raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            WineFeatures(
                acidity=8.0,
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0
                # Missing 'body'
            )
        assert "body" in str(exc_info.value)

    def test_decimal_values_accepted(self):
        """Decimal values should be accepted."""
        features = WineFeatures(
            acidity=8.5,
            minerality=7.25,
            fruitiness=6.75,
            tannin=1.1,
            body=5.9
        )
        assert features.acidity == 8.5
        assert features.minerality == 7.25

    def test_reasoning_is_optional(self):
        """Reasoning field should be optional."""
        features = WineFeatures(
            acidity=8.0,
            minerality=7.0,
            fruitiness=7.0,
            tannin=1.0,
            body=5.0
        )
        assert features.reasoning is None


class TestWineExtraction:
    """Test WineExtraction schema validation."""

    def test_valid_wine_extraction(self):
        """Valid extraction should pass validation."""
        extraction = WineExtraction(
            wine_name="Fefiñanes Albariño 2022",
            producer="Fefiñanes",
            vintage=2022,
            notes="Fresh, mineral, citrus notes",
            score=8.5,
            country="Spain",
            region="Rías Baixas",
            wine_color="White",
            is_sparkling=False,
            is_natural=False,
            sweetness="Dry",
            acidity=8.5,
            minerality=8.0,
            fruitiness=7.0,
            tannin=1.0,
            body=5.5
        )
        assert extraction.wine_name == "Fefiñanes Albariño 2022"
        assert extraction.country == "Spain"

    def test_vintage_range_validation(self):
        """Vintage must be between 1900 and 2100."""
        # Valid vintage
        extraction = WineExtraction(
            wine_name="Test Wine 2020",
            producer="Test Producer",
            vintage=2020,
            notes="Test notes",
            score=7.0,
            country="Spain",
            region="Test Region",
            wine_color="White",
            is_sparkling=False,
            is_natural=False,
            sweetness="Dry",
            acidity=8.0,
            minerality=7.0,
            fruitiness=7.0,
            tannin=1.0,
            body=5.0
        )
        assert extraction.vintage == 2020

        # Invalid vintage (too old)
        with pytest.raises(ValidationError):
            WineExtraction(
                wine_name="Ancient Wine",
                producer="Test",
                vintage=1800,
                notes="Too old",
                score=7.0,
                country="Spain",
                region="Test",
                wine_color="White",
                is_sparkling=False,
                is_natural=False,
                sweetness="Dry",
                acidity=8.0,
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )

        # Invalid vintage (too new)
        with pytest.raises(ValidationError):
            WineExtraction(
                wine_name="Future Wine",
                producer="Test",
                vintage=2150,
                notes="Too new",
                score=7.0,
                country="Spain",
                region="Test",
                wine_color="White",
                is_sparkling=False,
                is_natural=False,
                sweetness="Dry",
                acidity=8.0,
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )

    def test_score_range_validation(self):
        """Score must be between 1.0 and 10.0."""
        with pytest.raises(ValidationError):
            WineExtraction(
                wine_name="Test Wine",
                producer="Test",
                vintage=2020,
                notes="Test",
                score=0.5,  # Too low
                country="Spain",
                region="Test",
                wine_color="White",
                is_sparkling=False,
                is_natural=False,
                sweetness="Dry",
                acidity=8.0,
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )

        with pytest.raises(ValidationError):
            WineExtraction(
                wine_name="Test Wine",
                producer="Test",
                vintage=2020,
                notes="Test",
                score=10.5,  # Too high
                country="Spain",
                region="Test",
                wine_color="White",
                is_sparkling=False,
                is_natural=False,
                sweetness="Dry",
                acidity=8.0,
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )

    def test_boolean_fields(self):
        """Boolean fields should accept True/False."""
        extraction = WineExtraction(
            wine_name="Natural Sparkling Wine",
            producer="Test",
            vintage=2020,
            notes="Test",
            score=8.0,
            country="France",
            region="Champagne",
            wine_color="White",
            is_sparkling=True,
            is_natural=True,
            sweetness="Dry",
            acidity=9.0,
            minerality=7.0,
            fruitiness=6.0,
            tannin=1.0,
            body=5.0
        )
        assert extraction.is_sparkling is True
        assert extraction.is_natural is True

    def test_flavor_features_validation(self):
        """All 5 flavor features must be in valid range."""
        # Test that all features are validated
        with pytest.raises(ValidationError):
            WineExtraction(
                wine_name="Test",
                producer="Test",
                vintage=2020,
                notes="Test",
                score=8.0,
                country="Spain",
                region="Test",
                wine_color="White",
                is_sparkling=False,
                is_natural=False,
                sweetness="Dry",
                acidity=11.0,  # Invalid
                minerality=7.0,
                fruitiness=7.0,
                tannin=1.0,
                body=5.0
            )

    def test_all_required_fields_present(self):
        """All required fields must be provided."""
        with pytest.raises(ValidationError) as exc_info:
            WineExtraction(
                wine_name="Test Wine",
                producer="Test Producer",
                vintage=2020
                # Missing many required fields
            )
        error_str = str(exc_info.value)
        assert "notes" in error_str or "field required" in error_str.lower()
