"""Tests for decant.services.text_infer.

The LLM call is mocked. We're testing the parsing, rounding, and
failure-handling logic — not the model. The point of the module is
to run inference once and return clean, rounded features; these tests
pin that contract.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from decant.services.text_infer import infer_features_from_text


def _mock_client(content: str) -> MagicMock:
    """Build a mock OpenAI client whose completion returns `content`."""
    client = MagicMock()
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def _valid_payload(**overrides) -> str:
    profile = {
        "acidity": 8.5,
        "fruitiness": 6.0,
        "body": 5.5,
        "tannin": 2.0,
        "minerality": 8.0,
    }
    profile.update(overrides)
    return json.dumps({
        "wine_metadata": {"name": "Test", "region": "Test", "style": "x"},
        "technical_profile": profile,
        "sommelier_verdict": "Crisp and mineral.",
    })


class TestInferFeaturesFromText:

    def test_returns_all_five_features(self):
        client = _mock_client(_valid_payload())
        result = infer_features_from_text("Albariño", "Galicia", client)
        assert result is not None
        assert set(result.keys()) == {
            "acidity", "fruitiness", "body", "tannin", "minerality"
        }

    def test_rounds_to_one_decimal(self):
        # Sub-decimal jitter should be rounded away.
        client = _mock_client(_valid_payload(acidity=8.4732, minerality=7.951))
        result = infer_features_from_text("Albariño", "Galicia", client)
        assert result["acidity"] == 8.5
        assert result["minerality"] == 8.0

    def test_returns_none_on_invalid_json(self):
        client = _mock_client("not json at all")
        result = infer_features_from_text("X", "Y", client)
        assert result is None

    def test_returns_none_on_missing_profile(self):
        client = _mock_client(json.dumps({"wine_metadata": {}}))
        result = infer_features_from_text("X", "Y", client)
        assert result is None

    def test_returns_none_on_missing_feature(self):
        # technical_profile present but missing 'tannin'
        payload = json.dumps({
            "technical_profile": {
                "acidity": 8.0, "fruitiness": 6.0, "body": 5.0, "minerality": 7.0
            }
        })
        client = _mock_client(payload)
        result = infer_features_from_text("X", "Y", client)
        assert result is None

    def test_returns_none_on_non_numeric_feature(self):
        client = _mock_client(_valid_payload(acidity="high"))
        result = infer_features_from_text("X", "Y", client)
        assert result is None

    def test_returns_none_when_client_raises(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network down")
        result = infer_features_from_text("X", "Y", client)
        assert result is None

    def test_passes_deterministic_params(self):
        # The call should request temperature/seed for best-effort
        # determinism. We assert the kwargs are forwarded.
        client = _mock_client(_valid_payload())
        infer_features_from_text("Albariño", "Galicia", client)
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == 42
        assert kwargs["response_format"] == {"type": "json_object"}
