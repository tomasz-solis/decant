"""Tests for decant.ui.styles.

These pin three things:
1. The styles module exposes the two functions app.py needs.
2. Inline `<style>` blocks no longer live in app.py — CSS is
   sourced from decant.ui.styles. A regression test catches the
   "someone hand-pasted CSS back into app.py" failure mode.
3. The Streamlit config theme palette matches the CSS palette.
   This catches the exact bug where .streamlit/config.toml was
   still set to dark-mode colours while the CSS had moved to the
   light Mediterranean theme — which made every portaled widget
   (dropdowns, file uploader, tooltips) render dark.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


class TestConfigThemeConsistency:
    """The Streamlit config theme must match the CSS palette.

    Streamlit's own widgets (dropdown menus, file uploader, tooltips)
    render from config.toml, not from our CSS. If the two drift, the
    portaled widgets paint with the wrong palette and CSS can't reach
    them. These assertions pin the key colours to the same values the
    CSS uses.
    """

    def _read_config(self) -> str:
        config_path = REPO_ROOT / ".streamlit" / "config.toml"
        assert config_path.exists(), "Missing .streamlit/config.toml"
        return config_path.read_text()

    def test_config_is_light_base(self):
        config = self._read_config()
        assert 'base = "light"' in config, (
            "Config theme must be light base for the Mediterranean theme; "
            "a dark base makes dropdowns and widgets render dark"
        )

    def test_config_background_is_cream(self):
        config = self._read_config()
        assert "#FAF6F0" in config, "Config backgroundColor must be cream #FAF6F0"

    def test_config_primary_is_terracotta(self):
        config = self._read_config()
        assert "#C2410C" in config, "Config primaryColor must be terracotta #C2410C"

    def test_config_has_no_dark_leftovers(self):
        config = self._read_config()
        # The old dark-theme values that caused the black-dropdown bug.
        for stale in ["#0F0F12", "#1A1A1E", "#E8E8EB", "#8B0000"]:
            assert stale not in config, (
                f"Stale dark-theme colour {stale} still in config.toml — "
                f"this is what made the dropdowns render black"
            )


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
        # Phase 4: Mediterranean palette (terracotta + olive + cream)
        # with Playfair Display headings + DM Sans body.
        for landmark in [
            "--terracotta",
            "--olive",
            "--bg-primary: #FAF6F0",
            "Playfair Display",
            "DM Sans",
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
