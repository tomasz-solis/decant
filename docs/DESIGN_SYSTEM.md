# Decant Design System

A short reference for the design tokens that drive Decant's UI. The goal isn't to be exhaustive - it's to have one place where the choices and trade-offs are explained, so the CSS in `src/decant/ui/styles.py` and the Plotly theming in `src/decant/ui/components.py` don't drift apart over time.

If you're touching colour, type, spacing or shadow anywhere in this codebase, this is the source of truth. The tokens live in three coordinated definitions:

- CSS variables at the top of `_GLOBAL_STYLES` in `src/decant/ui/styles.py` - for HTML/Streamlit
- Streamlit theme config in `.streamlit/config.toml` - for widget defaults, portal layers, and initial font loading
- `_THEME` dict in `src/decant/ui/components.py` - for Plotly charts (which can't read CSS variables)
- `UIConstants.WINE_COLORS_CHART` in `src/decant/constants.py` - for per-wine-category chart traces

The CSS variables and Plotly `_THEME` have to stay in sync by hand. The per-wine-category chart palette is defined once in `UIConstants.WINE_COLORS_CHART` and imported by `components.py`, so don't duplicate those category colours inside chart functions. Streamlit's own widget renderer also reads `.streamlit/config.toml`; that file must use the same body and heading font families as the CSS.

## The theme in one sentence

An editorial cellar palette: ivory page, parchment panels, bordeaux primary, mineral green secondary, near-ink text. Newsreader is reserved for the Decant masthead only. Inter handles headings, controls, cards, metadata, helper copy, dense values, and charts. The reference points are modern wine magazines, restrained cellar labels, and the quieter side of gallery publishing - crisp rules, generous whitespace, no glow effects. Saturated colours are limited to Plotly's wine-category traces, feature bars, and tiny card accents, where colour recognition matters more than page chrome restraint.

## Colours

### Surfaces

| Token | Hex | Use |
|---|---|---|
| `--bg-primary` | `#F6F3EC` | The page background. Ivory, warm but cleaner than cream. |
| `--bg-secondary` | `#ECE6DA` | A half-step deeper. Tab nav, sidebar, gallery image placeholders. The "behind cards" surface. |
| `--card-bg` | `#FFFCF6` | Parchment white for raised surfaces - wine cards, info panels, the inside of expanders. |
| `--card-border` | `#D8D0C2` | A printed hairline that separates a card from the page without competing with content. |

### Accents

| Token | Hex | Use |
|---|---|---|
| `--terracotta` | `#8A1F3D` | Primary accent. The token name is historical; the colour is bordeaux. Use for buttons, active tab underline, focus rings. |
| `--terracotta-dark` | `#66162E` | Hover/active state of bordeaux surfaces. |
| `--terracotta-soft` | `#F1DCE1` | A quiet bordeaux tint. Hover background on neutral controls. |
| `--olive` | `#55614B` | Secondary accent. Sparingly: a second chart series and the signed-in pill. |
| `--olive-soft` | `#E2E7D8` | Mineral green tint. Background of soft notification chips. |
| `--wine` | `#7A1730` | Deep red used for verdicts, the "liked" badge, and red-wine chart traces. |
| `--wine-fill` | `rgba(122, 23, 48, 0.18)` | The same wine red as a translucent fill - for chart areas under the trace. |
| `--gold` | `#B89A4D` | White-wine feature bars and placeholder accents. |
| `--rose` | `#C77683` | Rose-wine feature bars and placeholder accents. |
| `--orange` | `#B86A3C` | Orange-wine feature bars and placeholder accents. |

### Text

| Token | Hex | Use |
|---|---|---|
| `--text-primary` | `#211A16` | Near-ink. Body text, headings, headline numbers. Softer than pure black. |
| `--text-secondary` | `#5B544B` | Slightly lighter brown for captions, subtitles, secondary metadata. Still readable as body. |
| `--text-muted` | `#7F7568` | For tertiary info: timestamps, placeholder hints, things the eye should slide over. |
| `--text-on-accent` | `#FFFCF6` | Parchment text on bordeaux/mineral/wine backgrounds. |

### When to use which accent

- Bordeaux (`--terracotta`) is the primary call-to-action colour. Anything the user is supposed to *do* lives in bordeaux.
- Mineral green (`--olive`) is for "this is happening" rather than "do this." Status pills, secondary chart series.
- Wine is for "this is wine the user has decided they like." Verdict labels, the heart icon, primary radar trace.

If two of these would compete for the user's eye in the same view, escalate to a design review - they're meant to land on different conceptual layers.

## Typography

| Token | Stack | Use |
|---|---|---|
| `--font-display` | Newsreader, Georgia, Times New Roman, serif | Decant masthead only. |
| `--font-body` | Inter, system-ui, -apple-system, sans-serif | Everything else: tabs, h3/h4 labels, body text, inputs, chart titles, and chart labels. |

The two-font pairing earns its complexity from the editorial direction - a wine magazine can have a distinctive masthead without making every label decorative. If you find yourself wanting a third font, push back hard.

Fonts are loaded with a Google Fonts `<link>` emitted by `_FONT_LINKS`, before `_GLOBAL_STYLES`. Do not use `@import` inside the style block.

### Type roles

| Role | Font | Weight | Size | Use |
|---|---|---:|---|---|
| Page title | Newsreader | 700 | `5.8rem` desktop, `3.1rem` mobile | `Decant` masthead |
| Section title | Inter | 700 | Streamlit `h2` default | Main tab titles |
| Subsection title | Inter | 700 | `1.5rem` | Section blocks inside tabs |
| Sub-subsection title | Inter | 700 | `1.25rem` | Nested technical sections |
| Big metric value | Inter | 700 | `2.05rem` | Scores, prices, palate percentages |
| Body text | Inter | 400 | `1rem` | Paragraphs, lists, normal Markdown |
| Body bold | Inter | 700 | `1rem` | Markdown `bold` inside body text |
| Card title | Inter | 700 | `1.02rem` | Wine card names in the gallery |
| Metric label | Inter | 600 | `0.8rem` | Uppercase metric labels |
| Caption/subtitle | Inter | 400 | `0.85rem` | Captions and secondary metadata |
| Button text | Inter | 600 | `1rem` | Buttons and download buttons |
| Input text | Inter | 400 | `1rem` | Form inputs |

Everything is upright. Do not use italic Markdown, `<em>`, or `font-style: italic`.

Do not put emoji in Markdown headings, tab labels, metric labels, or chart labels. If an icon is needed, render it separately as an inline icon or badge so it doesn't become part of the typographic hierarchy.

Streamlit widgets need both config and explicit typography selectors.
`.streamlit/config.toml` sets `theme.font` to Inter and
`theme.headingFont` to Inter so Streamlit/BaseWeb starts
from the right family. `_GLOBAL_STYLES` still targets tabs, filters,
selectboxes, radio labels, helper text, file uploaders, expanders,
alerts, and buttons because those surfaces use generated nested
markup. Do not rely on `body { font-family: ... }` inheritance for
these surfaces; keep the widget typography block aligned with any new
widget type.

Never use blanket selectors like `body span`, `body div`,
`[role="button"] *`, or widget-wide descendant `*` rules for
typography. Streamlit uses nested Material icon spans whose text
content is glyph names such as `expand_more`; broad font rules turn
those names into visible overlapping text.

## Shadow and elevation

| Token | Value | Use |
|---|---|---|
| `--shadow-card` | `0 1px 0 rgba(33, 26, 22, 0.04), 0 18px 50px rgba(69, 42, 30, 0.07)` | The resting shadow under wine cards and chart panels. Still restrained, but with more print-editorial depth. |
| `--shadow-card-hover` | `0 1px 0 rgba(33, 26, 22, 0.05), 0 24px 60px rgba(69, 42, 30, 0.10)` | The same direction, more spread. For hover states on repeated cards. |

No other elevation levels exist deliberately. Two levels are enough for a single-screen app; adding more creates a hierarchy users will try to read meaning into.

## Editorial Components

The app now uses a small set of custom HTML components in `src/decant/ui/editorial.py` for read-only surfaces where native Streamlit widgets look too plain:

- `.cellar-snapshot` and `.editorial-stat-grid` render high-level stat tiles with a printed top rule.
- `.feature-profile` and `.feature-profile-grid` render palate values as horizontal tasting bars instead of five separate `st.metric` blocks.
- `.ranked-list` renders Top Regions and Top Wines as numbered editorial rows.
- `.gallery-result-line`, `.wine-card-title`, `.wine-card-facts`, and `.wine-card-img-placeholder` give the gallery a bottle-note feel even when no photo has been uploaded.

Use native Streamlit inputs for interaction, but prefer these components for read-only summaries, rankings, and feature profiles.

## Radii

| Token | Value | Use |
|---|---|---|
| `--radius-card` | `8px` | Wine cards, hero card, info panels, gallery image placeholders. |
| `--radius-button` | `8px` | Buttons, inputs, selectboxes, the auth popover. |

## The Plotly mirror

`src/decant/ui/components.py`'s `_THEME` dict mirrors the CSS palette for chart rendering. The keys are simpler (no `--` prefix, snake_case) but the values match exactly:

| `_THEME` key | CSS variable |
|---|---|
| `bg` | `--bg-primary` |
| `bg_card` | `--card-bg` |
| `text` | `--text-primary` |
| `text_muted` | `--text-muted` |
| `accent` | `--terracotta` |
| `olive` | `--olive` |
| `wine` | `--wine` |
| `wine_fill` | `--wine-fill` |
| `font_family` | `--font-body` |

Plotly can't read CSS variables, so we duplicate. If you change a value on one side, change the other. A test in `tests/test_styles.py` pins the most important values to catch drift; consider extending it if you add new tokens that need cross-checking.

### Wine-category chart colours

Wine-category chart colours live in `UIConstants.WINE_COLORS_CHART` and are imported by `components.py`. They are not general UI accents.

These deliberately diverge from the muted CSS feature-bar tokens (`--gold`/`--wine`/`--rose`/`--orange`). On the consolidated palate radar every wine colour overlays in one plot; the brand tokens are all warm and close in value, so their translucent fills blended into one indistinguishable mass. The chart palette pulls the hues apart and gives each colour a distinct marker `symbol` (used by `create_consolidated_palate_radar`) so overlapping traces stay separable, including for colour-blind readers. Fills are kept light so stacked overlaps don't muddy. The feature-bar tokens in the cards are unchanged — distinctness only matters where the colours overlap.

| Wine colour | Primary | Fill | Marker |
|---|---|---|---|
| White | `#C9A227` | `rgba(201, 162, 39, 0.12)` | circle |
| Red | `#B0142F` | `rgba(176, 20, 47, 0.12)` | diamond |
| Rosé | `#E879A6` | `rgba(232, 121, 166, 0.12)` | square |
| Orange | `#D2691E` | `rgba(210, 105, 30, 0.12)` | triangle-up |

## What this system deliberately doesn't have

- No dark mode. Phase 4 dropped it. If we want dark mode again it's a full theme switcher, not a few overrides.
- No success/warning/error colour set. Streamlit's `st.success`/`st.error`/`st.warning` are used as-is for those states. Custom theming of those is a TODO if they ever feel jarring.
- No scale of muted colours (`--muted-100`, `--muted-200`, etc). The three text colours plus `card-border` are deliberately the only neutral tones. Adding more creates a system where future me has to pick "is this -300 or -400 muted" and the answer doesn't matter.
- No spacing scale tokens (`--space-1`, `--space-2`, etc). Streamlit's own spacing primitives (`st.columns`, `st.divider`, the gap between widgets) are deliberately the only sources. Trying to enforce a spacing scale on top of Streamlit's layout system fights the framework. If a specific spot needs a specific number of pixels, use the number inline and write a comment about why.

## Drift watch

The most likely way this system breaks is value drift between the CSS variables and `_THEME`. Two things help:

1. The four `test_styles.py` assertions on the config palette catch drift between `.streamlit/config.toml` and the CSS - see `TestConfigThemeConsistency`.
2. Anytime you grep for a literal hex code (e.g. `#7A1730`) and find it outside the source-of-truth definitions, that's a leak. Fix it back to a token or to `UIConstants.WINE_COLORS_CHART`.

The leakage audit that produced this doc found six such sites; they're now all using tokens. If a future audit finds more, the same fix pattern applies: replace the literal with a token, or if no token fits, add a token first.
