"""Tests for decant.ui.styles.

These pin three things:
1. The styles module exposes the two functions app.py needs.
2. Inline `<style>` blocks no longer live in app.py — CSS is
   sourced from decant.ui.styles. A regression test catches the
   "someone hand-pasted CSS back into app.py" failure mode.
3. The Streamlit config theme palette and fonts match the CSS theme.
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
    render from config.toml before our CSS can polish them. If the two
    drift, portaled widgets paint with the wrong palette or fall back to
    Streamlit's default font. These assertions pin the key values to
    the same tokens the CSS uses.
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

    def test_config_uses_decant_fonts(self):
        config = self._read_config()
        assert 'font = "DM Sans:' in config, (
            "Config theme.font must load DM Sans so Streamlit widgets "
            "do not fall back to the default sans-serif"
        )
        assert 'headingFont = "Playfair Display:' in config, (
            "Config theme.headingFont must load Playfair Display for "
            "Streamlit headings"
        )
        assert "baseFontWeight = 400" in config

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
        """The core theme exposes the typography and colour tokens."""
        from decant.ui.styles import _GLOBAL_STYLES
        # Theme variables and key selectors that the rest of the app
        # references via class names in unsafe_allow_html markup.
        # Phase 4: Mediterranean palette (terracotta + olive + cream)
        # with Playfair Display display titles + DM Sans body.
        for landmark in [
            "--terracotta",
            "--terracotta-dark",
            "--olive",
            "--bg-primary: #FAF6F0",
            "--text-on-accent",
            "--wine-fill: rgba(124, 45, 18, 0.4)",
            "Playfair Display",
            "DM Sans",
            ".main-title",
            "STREAMLIT WIDGET TYPOGRAPHY",
            "DISPLAY HEADINGS",
            "h3",
            "h4",
            '[data-baseweb="select"]',
            '[data-testid="stExpander"]',
            '[data-testid="InputInstructions"]',
            "@media (max-width: 768px)",
        ]:
            assert landmark in _GLOBAL_STYLES, (
                f"core theme landmark missing from _GLOBAL_STYLES: {landmark!r}"
            )

    def test_font_links_are_loaded_outside_stylesheet(self):
        """Google fonts should be linked before the inline stylesheet."""
        from decant.ui.styles import _FONT_LINKS, _GLOBAL_STYLES

        assert "fonts.googleapis.com" in _FONT_LINKS
        assert "fonts.gstatic.com" in _FONT_LINKS
        assert 'rel="stylesheet"' in _FONT_LINKS
        assert "@import" not in _GLOBAL_STYLES

    def test_plotly_theme_matches_core_tokens(self):
        """Plotly mirrors the CSS tokens it cannot read directly."""
        from decant.ui.components import _THEME

        assert _THEME["bg"] == "#FAF6F0"
        assert _THEME["bg_card"] == "#FFFDF8"
        assert _THEME["text"] == "#3D2817"
        assert _THEME["text_muted"] == "#8B7E6D"
        assert _THEME["accent"] == "#C2410C"
        assert _THEME["olive"] == "#65733E"
        assert _THEME["wine"] == "#7C2D12"
        assert _THEME["wine_fill"] == "rgba(124, 45, 18, 0.4)"
        assert (
            _THEME["font_family"]
            == "DM Sans, system-ui, -apple-system, sans-serif"
        )
        assert "font_family_display" not in _THEME

    def test_plotly_chart_palette_uses_constants(self):
        """Wine-category chart colours have one source of truth."""
        from decant.constants import UIConstants
        from decant.ui.components import _WINE_COLOR_CHART

        expected = {
            color.value: spec
            for color, spec in UIConstants.WINE_COLORS_CHART.items()
        }
        assert _WINE_COLOR_CHART == expected

    def test_gallery_styles_contains_grid_selectors(self):
        """Gallery-specific CSS keeps the card grid selectors."""
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
