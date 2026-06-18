"""Tests for decant.ui.styles.

These pin three things:
1. The styles module exposes the two functions app.py needs.
2. Inline `<style>` blocks no longer live in app.py - CSS is
   sourced from decant.ui.styles. A regression test catches the
   "someone hand-pasted CSS back into app.py" failure mode.
3. The Streamlit config theme palette and fonts match the CSS theme.
   This catches the exact bug where .streamlit/config.toml was
   still set to dark-mode colours while the CSS had moved to the
   light editorial theme - which made every portaled widget
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
            "Config theme must be light base for the editorial theme; "
            "a dark base makes dropdowns and widgets render dark"
        )

    def test_config_background_is_ivory(self):
        config = self._read_config()
        assert "#F6F3EC" in config, "Config backgroundColor must be ivory #F6F3EC"

    def test_config_primary_is_bordeaux(self):
        config = self._read_config()
        assert "#8A1F3D" in config, "Config primaryColor must be bordeaux #8A1F3D"

    def test_config_uses_decant_fonts(self):
        config = self._read_config()
        assert 'font = "Inter:' in config, (
            "Config theme.font must load Inter so Streamlit widgets "
            "do not fall back to the default sans-serif"
        )
        assert 'headingFont = "Inter:' in config, (
            "Config theme.headingFont must use Inter so Streamlit headings "
            "do not add a third-looking typographic layer"
        )
        assert "baseFontWeight = 400" in config

    def test_config_has_no_dark_leftovers(self):
        config = self._read_config()
        # The old dark-theme values that caused the black-dropdown bug.
        for stale in ["#0F0F12", "#1A1A1E", "#E8E8EB", "#8B0000"]:
            assert stale not in config, (
                f"Stale dark-theme colour {stale} still in config.toml - "
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
        # Editorial cellar palette with a single Inter UI layer and
        # a Newsreader masthead wordmark.
        for landmark in [
            "--terracotta",
            "--terracotta-dark",
            "--olive",
            "--bg-primary: #F6F3EC",
            "--text-on-accent",
            "--wine-fill: rgba(122, 23, 48, 0.18)",
            "Newsreader",
            "Inter",
            ".main-title",
            ".app-masthead",
            "--hero-image",
            ".cellar-snapshot",
            ".feature-profile",
            ".ranked-list",
            ".gallery-result-line",
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

        assert _THEME["bg"] == "#F6F3EC"
        assert _THEME["bg_card"] == "#FFFCF6"
        assert _THEME["text"] == "#211A16"
        assert _THEME["text_muted"] == "#7F7568"
        assert _THEME["accent"] == "#8A1F3D"
        assert _THEME["olive"] == "#55614B"
        assert _THEME["wine"] == "#7A1730"
        assert _THEME["wine_fill"] == "rgba(122, 23, 48, 0.18)"
        assert (
            _THEME["font_family"]
            == "Inter, system-ui, -apple-system, sans-serif"
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
            ".wine-card-title",
            ".wine-card-facts",
            ".placeholder-name",
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
            "app.py imports but never calls apply_global_styles - "
            "the theme wouldn't render."
        )
