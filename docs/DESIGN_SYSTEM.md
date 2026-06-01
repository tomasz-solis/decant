# Decant Design System

A short reference for the design tokens that drive Decant's UI. The goal isn't to be exhaustive — it's to have one place where the choices and trade-offs are explained, so the CSS in `src/decant/ui/styles.py` and the Plotly theming in `src/decant/ui/components.py` don't drift apart over time.

If you're touching colour, type, spacing or shadow anywhere in this codebase, this is the source of truth. The tokens live in three coordinated definitions:

- **CSS variables** at the top of `_GLOBAL_STYLES` in `src/decant/ui/styles.py` — for HTML/Streamlit
- **Streamlit theme config** in `.streamlit/config.toml` — for widget defaults, portal layers, and initial font loading
- **`_THEME` dict** in `src/decant/ui/components.py` — for Plotly charts (which can't read CSS variables)
- **`UIConstants.WINE_COLORS_CHART`** in `src/decant/constants.py` — for per-wine-category chart traces

The CSS variables and Plotly `_THEME` have to stay in sync by hand. The per-wine-category chart palette is defined once in `UIConstants.WINE_COLORS_CHART` and imported by `components.py`, so don't duplicate those category colours inside chart functions. Streamlit's own widget renderer also reads `.streamlit/config.toml`; that file must use the same body and heading font families as the CSS.

## The theme in one sentence

A Mediterranean light palette: cream paper, terracotta accents, olive secondary, deep wine red for verdicts, warm brown text. Playfair Display is reserved for the masthead, `h2`-`h4` section headings, and big metric numbers. DM Sans handles everything else. The reference points are wine magazines and the cleaner end of Stripe's documentation — restrained, editorial, no glow effects. Saturated colours are limited to Plotly's wine-category traces, where colour recognition matters more than page chrome restraint.

## Colours

### Surfaces

| Token | Hex | Use |
|---|---|---|
| `--bg-primary` | `#FAF6F0` | The page background. Cream, not white — closer to baked paper than fresh copier paper. |
| `--bg-secondary` | `#F3EDE3` | A half-step darker. Tab nav, sidebar, gallery image placeholders. The "behind cards" surface. |
| `--card-bg` | `#FFFDF8` | Warm white for raised surfaces — wine cards, info panels, the inside of expanders. |
| `--card-border` | `#E8DFCF` | A hairline that separates a card from the page without competing with content. |

### Accents

| Token | Hex | Use |
|---|---|---|
| `--terracotta` | `#C2410C` | Primary accent. Buttons, the active tab underline, the masthead bar, focus rings. |
| `--terracotta-dark` | `#9A330A` | Hover/active state of terracotta surfaces. |
| `--terracotta-soft` | `#FED7AA` | A 10–15% tint of terracotta. Hover background on neutral controls. |
| `--olive` | `#65733E` | Secondary accent. Sparingly: a second chart series, the "logged in" pill, decorative hairlines. |
| `--olive-soft` | `#DCE3C4` | Olive tint. Background of soft notification chips. |
| `--wine` | `#7C2D12` | Deep red used for verdicts, the "liked" badge, and red-wine chart traces. |
| `--wine-fill` | `rgba(124, 45, 18, 0.4)` | The same wine red as a translucent fill — for chart areas under the trace. |

### Text

| Token | Hex | Use |
|---|---|---|
| `--text-primary` | `#3D2817` | Warm dark brown. Body text, headings, headline numbers. Not pure black — pure black against cream reads as harsh. |
| `--text-secondary` | `#5C4D3F` | Slightly lighter brown for captions, subtitles, secondary metadata. Still readable as body. |
| `--text-muted` | `#8B7E6D` | For tertiary info: timestamps, placeholder hints, things the eye should slide over. |
| `--text-on-accent` | `#FFFFFF` | White text on terracotta/olive/wine backgrounds. The only place pure white appears. |

### When to use which accent

- **Terracotta** is the primary call-to-action colour. Anything the user is supposed to *do* lives in terracotta.
- **Olive** is for "this is happening" rather than "do this." Status pills, secondary chart series.
- **Wine** is for "this is wine the user has decided they like." Verdict labels, the heart icon, primary radar trace.

If two of these would compete for the user's eye in the same view, escalate to a design review — they're meant to land on different conceptual layers.

## Typography

| Token | Stack | Use |
|---|---|---|
| `--font-display` | Playfair Display, Georgia, Times New Roman, serif | Masthead, `h2`-`h4` section headings, and big metric numbers. |
| `--font-body` | DM Sans, system-ui, -apple-system, sans-serif | Everything else: tabs, h3/h4 labels, body text, inputs, chart titles, and chart labels. |

The two-font pairing earns its complexity from the editorial direction — a wine magazine can have a distinctive masthead without making every label decorative. If you find yourself wanting a third font, push back hard.

Fonts are loaded with a Google Fonts `<link>` emitted by `_FONT_LINKS`, before `_GLOBAL_STYLES`. Do not use `@import` inside the style block.

### Type roles

| Role | Font | Weight | Size | Use |
|---|---|---:|---|---|
| Page title | Playfair Display | 700 | `3.4rem` desktop, `2.4rem` mobile | `Decant` masthead |
| Section title | Playfair Display | 700 | Streamlit `h2` default | Main tab titles |
| Subsection title | Playfair Display | 700 | `1.5rem` | Section blocks inside tabs |
| Sub-subsection title | Playfair Display | 700 | `1.25rem` | Nested technical sections |
| Big metric value | Playfair Display | 700 | Streamlit metric default | Scores, prices, palate percentages |
| Body text | DM Sans | 400 | `1rem` | Paragraphs, lists, normal Markdown |
| Body bold | DM Sans | 700 | `1rem` | Markdown `**bold**` inside body text |
| Card title | DM Sans | 700 | `16px` | Wine card names in the gallery |
| Metric label | DM Sans | 600 | `0.8rem` | Uppercase metric labels |
| Caption/subtitle | DM Sans | 400 | `0.85rem` | Captions and secondary metadata |
| Button text | DM Sans | 600 | `1rem` | Buttons and download buttons |
| Input text | DM Sans | 400 | `1rem` | Form inputs |

Everything is upright. Do not use italic Markdown, `<em>`, or `font-style: italic`.

Do not put emoji in Markdown headings, tab labels, metric labels, or chart labels. If an icon is needed, render it separately as an inline icon or badge so it doesn't become part of the typographic hierarchy.

Streamlit widgets need both config and explicit typography selectors.
`.streamlit/config.toml` sets `theme.font` to DM Sans and
`theme.headingFont` to Playfair Display so Streamlit/BaseWeb starts
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
| `--shadow-card` | `0 2px 8px rgba(120, 60, 30, 0.06)` | The resting shadow under wine cards, the hero card, info panels. Warm rather than neutral — the shadow itself is a faint brown tint. |
| `--shadow-card-hover` | `0 4px 14px rgba(120, 60, 30, 0.10)` | The same direction, more spread. For hover states and the primary button at rest. |

No other elevation levels exist deliberately. Two levels are enough for a single-screen app; adding more creates a hierarchy users will try to read meaning into.

## Radii

| Token | Value | Use |
|---|---|---|
| `--radius-card` | `12px` | Wine cards, hero card, info panels, gallery image placeholders. |
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

Plotly can't read CSS variables, so we duplicate. **If you change a value on one side, change the other.** A test in `tests/test_styles.py` pins the most important values to catch drift; consider extending it if you add new tokens that need cross-checking.

### Wine-category chart colours

Wine-category chart colours live in `UIConstants.WINE_COLORS_CHART` and are imported by `components.py`. They are not general UI accents.

| Wine colour | Primary | Fill |
|---|---|---|
| White | `#FFD700` | `rgba(255, 215, 0, 0.4)` |
| Red | `#7C2D12` | `rgba(124, 45, 18, 0.4)` |
| Rosé | `#FF69B4` | `rgba(255, 105, 180, 0.4)` |
| Orange | `#FF8C00` | `rgba(255, 140, 0, 0.4)` |

## What this system deliberately doesn't have

- **No dark mode.** Phase 4 dropped it. If we want dark mode again it's a full theme switcher, not a few overrides.
- **No success/warning/error colour set.** Streamlit's `st.success`/`st.error`/`st.warning` are used as-is for those states. Custom theming of those is a TODO if they ever feel jarring.
- **No scale of muted colours** (`--muted-100`, `--muted-200`, etc). The three text colours plus `card-border` are deliberately the only neutral tones. Adding more creates a system where future me has to pick "is this -300 or -400 muted" and the answer doesn't matter.
- **No spacing scale tokens** (`--space-1`, `--space-2`, etc). Streamlit's own spacing primitives (`st.columns`, `st.divider`, the gap between widgets) are deliberately the only sources. Trying to enforce a spacing scale on top of Streamlit's layout system fights the framework. If a specific spot needs a specific number of pixels, use the number inline and write a comment about why.

## Drift watch

The most likely way this system breaks is value drift between the CSS variables and `_THEME`. Two things help:

1. The four `test_styles.py` assertions on the config palette catch drift between `.streamlit/config.toml` and the CSS — see `TestConfigThemeConsistency`.
2. Anytime you grep for a literal hex code (e.g. `#7C2D12`) and find it outside the source-of-truth definitions, that's a leak. Fix it back to a token or to `UIConstants.WINE_COLORS_CHART`.

The leakage audit that produced this doc found six such sites; they're now all using tokens. If a future audit finds more, the same fix pattern applies: replace the literal with a token, or if no token fits, add a token first.
