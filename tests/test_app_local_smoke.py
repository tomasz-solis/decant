"""Local Streamlit smoke test."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).parent.parent


def test_app_renders_locally_with_core_tabs():
    """The app should boot locally and expose the stable four-tab layout."""
    at = AppTest.from_file(str(REPO_ROOT / "app.py"))
    at.run(timeout=30)

    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert [err.value for err in at.error] == []
    assert [tab.label for tab in at.tabs] == [
        "Add Wine",
        "My Palate Maps",
        "Stats",
        "Wine Gallery",
    ]

