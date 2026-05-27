"""Tests for decant.ui.styles.

These pin two things:
1. The styles module exposes the two functions app.py needs.
2. Inline `<style>` blocks no longer live in app.py — CSS is
   sourced from decant.ui.styles. A regression test catches the
   "someone hand-pasted CSS back into app.py" failure mode.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


class TestStylesModule:
    """The module exports what app.py imports."""

    def test_apply_global_styles_is_callable(self):
        from decant.ui.styles import apply_global_styles
        assert callable(apply_global_styles)

    def test_apply_gallery_styles_is_callable(self):
        from decant.ui.styles import apply_gallery_styles
        assert callable(apply_gallery_styles)

    def test_global_styles_contains_core_theme(self):
        from decant.ui.styles import _GLOBAL_STYLES
        # Theme variables and key selectors that the rest of the app
        # references via class names in unsafe_allow_html markup.
        for landmark in [
            "--wine-red",
            ".glass-card",
            ".main-title",
            "@media (max-width: 768px)",
        ]:
            assert landmark in _GLOBAL_STYLES, (
                f"core theme landmark missing from _GLOBAL_STYLES: {landmark!r}"
            )

    def test_gallery_styles_contains_grid_selectors(self):
        from decant.ui.styles import _GALLERY_STYLES
        for landmark in [
            ".wine-gallery-grid",
            ".wine-card-notes",
            ".icon-row",
            ".wine-card-footer",
        ]:
            assert landmark in _GALLERY_STYLES, (
                f"gallery landmark missing from _GALLERY_STYLES: {landmark!r}"
            )


class TestNoInlineCssInApp:
    """app.py must not embed `<style>` blocks directly.

    The styles module is the one source of truth for CSS. If a
    future hand-edit puts an inline `<style>` block back into
    app.py, this test fails. Adjust by moving the CSS into
    decant.ui.styles instead.
    """

    def test_app_has_no_inline_style_blocks(self):
        app_source = (REPO_ROOT / "app.py").read_text()
        # The `<style>` opener can only appear in app.py if someone
        # inlined CSS. There's no other reason for that string in a
        # Streamlit script.
        assert "<style>" not in app_source, (
            "<style> block found in app.py. Move the CSS into "
            "decant.ui.styles (apply_global_styles or "
            "apply_gallery_styles) instead."
        )

    def test_app_calls_apply_global_styles(self):
        """Defensive: confirm the global theme is actually applied."""
        app_source = (REPO_ROOT / "app.py").read_text()
        assert "apply_global_styles()" in app_source, (
            "app.py imports but never calls apply_global_styles — "
            "the theme wouldn't render."
        )
