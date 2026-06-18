"""Inline CSS for the Streamlit UI.

Two functions, two scopes:

- `apply_global_styles()` ships the main theme - typography, colour
  variables, layout polish for tabs, buttons, cards, mobile
  responsiveness. Call it once at app boot, right after
  `st.set_page_config`.

- `apply_gallery_styles()` ships gallery-specific CSS (grid layout,
  card clamps). Call it inside the Wine Gallery tab render path,
  just before the cards are drawn. Scoping it locally avoids
  polluting the global namespace with grid selectors that only
  matter on that one view.

Both functions emit a single `st.markdown(..., unsafe_allow_html=True)`
call. That's the only way Streamlit accepts custom CSS. The
`unsafe_allow_html=True` here is benign - the markup is a static
string we control, not user-supplied content.

Phase 5 - editorial cellar theme:
- Ivory background (#F6F3EC), parchment cards (#FFFCF6)
- Bordeaux primary (#8A1F3D), mineral green accent (#55614B),
  deep wine red for verdicts (#7A1730)
- Newsreader for the Decant masthead only; Inter for headings,
  body text, widgets, cards, labels, values, and controls
- Flat premium surfaces, thin rules, and restrained depth
"""

from __future__ import annotations

import streamlit as st


_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700&'
    'family=Newsreader:opsz,wght@6..72,500;6..72,650;6..72,700&display=swap" '
    'rel="stylesheet">'
)


_GLOBAL_STYLES = """
<style>
    /* ---- FONT LOAD ------------------------------------------------- */
    /* Loaded separately via st.markdown(_FONT_LINKS) before this
       stylesheet. Keep external font loading out of this style block. */

    /* ---- Editorial cellar palette ---------------------------------- */
    :root {
        --bg-primary: #F6F3EC;         /* ivory - warm but cleaner than cream */
        --bg-secondary: #ECE6DA;       /* deeper parchment for controls */
        --card-bg: #FFFCF6;            /* parchment white for raised surfaces */
        --card-border: #D8D0C2;        /* editorial hairline */
        --ink-panel: #211A16;          /* dark editorial contrast panel */
        --line-strong: rgba(33, 26, 22, 0.78);

        --terracotta: #8A1F3D;         /* primary accent - bordeaux */
        --terracotta-dark: #66162E;    /* hover state for primary controls */
        --terracotta-soft: #F1DCE1;    /* bordeaux tint for hover/highlight */
        --olive: #55614B;              /* secondary accent - mineral green */
        --olive-soft: #E2E7D8;         /* mineral tint */
        --wine: #7A1730;               /* deep red for verdicts, liked badge */
        --wine-fill: rgba(122, 23, 48, 0.18);
        --gold: #B89A4D;
        --rose: #C77683;
        --orange: #B86A3C;

        --text-primary: #211A16;       /* near-ink, softer than black */
        --text-secondary: #5B544B;     /* readable secondary metadata */
        --text-muted: #7F7568;         /* tertiary info only */
        --text-on-accent: #FFFCF6;     /* text on accent surfaces */

        --shadow-card: 0 1px 0 rgba(33, 26, 22, 0.05), 0 18px 44px rgba(69, 42, 30, 0.09);
        --shadow-card-hover: 0 1px 0 rgba(33, 26, 22, 0.06), 0 24px 58px rgba(69, 42, 30, 0.13);

        --radius-card: 8px;
        --radius-button: 8px;

        --font-display: 'Newsreader', Georgia, 'Times New Roman', serif;
        --font-body: 'Inter', system-ui, -apple-system, sans-serif;
        --hero-image: url("https://images.unsplash.com/photo-1643087448435-72f70bd4ce88?auto=format&fit=crop&fm=jpg&q=72&w=2400");
    }

    /* ---- Global base ---------------------------------------------- */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 400;
    }

    .main .block-container {
        max-width: 1220px;
        padding-top: 1.35rem;
        padding-bottom: 4rem;
    }

    /* A faint editorial column grid adds structure without turning
       into decoration. */
    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(90deg, rgba(33, 26, 22, 0.035) 1px, transparent 1px) !important;
        background-size: 96px 100%;
    }

    /* Masthead bar: a crisp printed-rule signal at the top edge. */
    .stApp::before {
        content: "";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(to right,
            var(--wine) 0%,
            var(--wine) 34%,
            var(--text-primary) 34%,
            var(--text-primary) 100%);
        z-index: 1000;
        pointer-events: none;
    }

    /* Streamlit injects a top header bar; on light theme we want it
       to blend with the page rather than show as a darker strip. */
    [data-testid="stHeader"] {
        background: var(--bg-primary) !important;
    }

    /* ---- DISPLAY HEADINGS (Newsreader) ---------------------------- */
    /* Targets the heading element AND its descendants (Streamlit
       renders heading text inside a <span> child). The body +
       stMarkdownContainer chains lift specificity above generated
       .st-emotion-cache-XXXXX classes.

       Per the Decant type scale, h1-h4 are display headings.
       Non-heading metadata must not use Markdown heading syntax. */
    body h1, body h2, body h3, body h4,
    body h1 *, body h2 *, body h3 *, body h4 *,
    body .main-title,
    body .main-title *,
    body [data-testid="stMarkdownContainer"] h1,
    body [data-testid="stMarkdownContainer"] h2,
    body [data-testid="stMarkdownContainer"] h3,
    body [data-testid="stMarkdownContainer"] h4,
    body [data-testid="stMarkdownContainer"] h1 *,
    body [data-testid="stMarkdownContainer"] h2 * {
        font-family: var(--font-display) !important;
        font-style: normal !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: 0;
    }

    body [data-testid="stMarkdownContainer"] h3 *,
    body [data-testid="stMarkdownContainer"] h4 * {
        font-family: var(--font-display) !important;
        font-style: normal !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: 0;
    }

    .main-title {
        font-size: 5.8rem !important;
        line-height: 0.86 !important;
        margin: 0 !important;
    }

    .app-masthead {
        border: none;
        padding: 0;
        margin: 0;
        position: relative;
    }

    .app-masthead::after {
        display: none;
    }

    /* Masthead band: the hero photo now sits *behind* the wordmark and
       the auth control instead of in a separate strip below. The dark
       gradient overlay is the contrast guard - it keeps the light title
       and the Sign-in button legible no matter which slice of the
       cellar photo lands behind them. */
    [data-testid="stHorizontalBlock"]:has(.app-masthead) {
        position: relative;
        padding: 30px 30px 32px 30px;
        margin-bottom: 22px;
        border: 1px solid var(--card-border);
        border-radius: var(--radius-card);
        overflow: hidden;
        background-image:
            linear-gradient(
                90deg,
                rgba(18, 12, 9, 0.86) 0%,
                rgba(22, 15, 11, 0.58) 52%,
                rgba(18, 12, 9, 0.80) 100%
            ),
            var(--hero-image);
        background-size: cover;
        background-position: center 50%;
        box-shadow: var(--shadow-card);
    }

    .masthead-kicker {
        margin: 0 0 8px 0 !important;
        font-family: var(--font-body) !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        color: #E7C27C !important;          /* warm gold reads on the photo */
    }

    /* The global heading rules force near-ink colour with !important;
       on the photo that would be unreadable, so re-light the masthead
       text here (higher specificity wins) and back it with a soft
       shadow for extra contrast insurance. */
    body [data-testid="stHorizontalBlock"]:has(.app-masthead) .main-title,
    body [data-testid="stHorizontalBlock"]:has(.app-masthead) .main-title * {
        color: var(--text-on-accent) !important;
        text-shadow: 0 2px 20px rgba(0, 0, 0, 0.5);
    }
    body [data-testid="stHorizontalBlock"]:has(.app-masthead) .subtitle,
    body [data-testid="stHorizontalBlock"]:has(.app-masthead) .subtitle * {
        color: rgba(255, 252, 246, 0.9) !important;
        text-shadow: 0 1px 12px rgba(0, 0, 0, 0.45);
    }

    /* Lift the Sign-in / account control off the busy photo so it stays
       a crisp, tappable target. */
    [data-testid="stHorizontalBlock"]:has(.app-masthead) [data-testid="stPopover"] button {
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.38) !important;
    }

    /* Section titles get a short editorial rule. */
    [data-testid="stMarkdownContainer"] h2 {
        position: relative;
        padding-bottom: 0.4rem;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stMarkdownContainer"] h2::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 54px;
        height: 1px;
        background: var(--text-primary);
        border-radius: 1px;
    }

    body [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.5rem !important;
        line-height: 1.25 !important;
    }

    body [data-testid="stMarkdownContainer"] h4 {
        font-size: 1.25rem !important;
        line-height: 1.3 !important;
    }

    body h1, body h2, body h3, body h4,
    body h1 *, body h2 *, body h3 *, body h4 *,
    body [data-testid="stMarkdownContainer"] h1,
    body [data-testid="stMarkdownContainer"] h2,
    body [data-testid="stMarkdownContainer"] h3,
    body [data-testid="stMarkdownContainer"] h4,
    body [data-testid="stMarkdownContainer"] h1 *,
    body [data-testid="stMarkdownContainer"] h2 *,
    body [data-testid="stMarkdownContainer"] h3 *,
    body [data-testid="stMarkdownContainer"] h4 * {
        font-family: var(--font-body) !important;
    }

    body .main-title,
    body .main-title * {
        font-family: var(--font-display) !important;
    }

    /* ---- LOWER-LEVEL TITLES (Inter) ------------------------------- */
    body h5, body h6,
    body h5 *, body h6 *,
    body [data-testid="stMarkdownContainer"] h5,
    body [data-testid="stMarkdownContainer"] h6,
    body [data-testid="stMarkdownContainer"] h5 *,
    body [data-testid="stMarkdownContainer"] h6 * {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: 0;
    }

    body [data-testid="stMarkdownContainer"] a[href^="#"] {
        display: none !important;
    }

    /* Compact headings inside popovers/forms are control labels, not
       page sections, so they stay in the body family. */
    body .form-title,
    body .form-title * {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        line-height: 1.3 !important;
        color: var(--text-primary) !important;
        margin: 0 0 0.75rem 0 !important;
    }

    /* ---- STREAMLIT WIDGET TYPOGRAPHY (Inter) --------------------- */
    /* Keep this narrow. Streamlit renders chevrons and controls as
       Material icon spans whose text content is names like
       "expand_more". Styling every descendant with `*`, or styling
       all `span`/`div` globally, breaks the icon font and exposes
       those names as overlapping text. */
    body [data-testid="stTabs"],
    body [data-testid="stWidgetLabel"],
    body [data-testid="stTextInput"],
    body [data-testid="stTextArea"],
    body [data-testid="stNumberInput"],
    body [data-testid="stSelectbox"],
    body [data-testid="stRadio"],
    body [data-testid="stCheckbox"],
    body [data-testid="stToggle"],
    body [data-testid="stSlider"],
    body [data-testid="stFileUploader"],
    body [data-testid="stDownloadButton"],
    body [data-testid="stButton"],
    body [data-testid="stPopover"],
    body [data-testid="stAlert"],
    body [data-testid="stMetric"],
    body [data-baseweb="tab"],
    body [data-baseweb="select"],
    body [data-baseweb="menu"],
    body [data-baseweb="popover"],
    body [data-baseweb="tooltip"],
    body [data-baseweb="input"],
    body [data-baseweb="textarea"],
    body [data-baseweb="radio"],
    body [data-baseweb="checkbox"],
    body [role="tab"],
    body [role="combobox"],
    body [role="listbox"],
    body [role="option"],
    body [role="tooltip"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        letter-spacing: 0 !important;
    }

    /* Streamlit widgets often put the visible text one or two nodes
       below the wrapper above. Reach those text leaves explicitly,
       while excluding Material icon spans so chevrons stay icons. */
    body :where(
        [data-testid="stTabs"],
        [data-testid="stWidgetLabel"],
        [data-testid="stTextInput"],
        [data-testid="stTextArea"],
        [data-testid="stNumberInput"],
        [data-testid="stSelectbox"],
        [data-testid="stRadio"],
        [data-testid="stCheckbox"],
        [data-testid="stToggle"],
        [data-testid="stSlider"],
        [data-testid="stFileUploader"],
        [data-testid="stDownloadButton"],
        [data-testid="stButton"],
        [data-testid="stPopover"],
        [data-testid="stAlert"],
        [data-testid="stExpander"],
        [data-baseweb="tab"],
        [data-baseweb="select"],
        [data-baseweb="menu"],
        [data-baseweb="popover"],
        [data-baseweb="tooltip"],
        [data-baseweb="input"],
        [data-baseweb="textarea"],
        [data-baseweb="radio"],
        [data-baseweb="checkbox"]
    ) :where(
        p,
        label,
        li,
        small,
        strong,
        em,
        input,
        textarea,
        select,
        button,
        a
    ) {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        letter-spacing: 0 !important;
    }

    body :where(
        [data-testid="stSelectbox"],
        [data-testid="stRadio"],
        [data-testid="stCheckbox"],
        [data-testid="stToggle"],
        [data-testid="stFileUploader"],
        [data-baseweb="select"],
        [data-baseweb="menu"],
        [data-baseweb="popover"],
        [data-baseweb="tooltip"]
    ) :where(
        span:not([aria-hidden="true"]):not([class*="material"]):not([class*="Material"]),
        div:not([aria-hidden="true"]):not([role="presentation"]):not([class*="material"]):not([class*="Material"])
    ) {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        letter-spacing: 0 !important;
    }

    /* ---- METRIC VALUE (Newsreader big numbers) -------------------- */
    body [data-testid="stMetricValue"],
    body [data-testid="stMetricValue"] > div,
    body [data-testid="stMetricValue"] * {
        font-family: var(--font-display) !important;
        font-style: normal !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
    }

    /* ---- METRIC LABEL (Inter uppercase) --------------------------- */
    body [data-testid="stMetricLabel"],
    body [data-testid="stMetricLabel"] * {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0;
    }

    /* ---- BODY BASELINE (Inter, upright, everything else) ---------- */
    /* The body+descendant chain beats Streamlit's emotion-cache classes.
       font-style: normal on em/strong is the kill-switch for any
       accidental emphasized markdown (_text_) or emphasis tags. */
    body p, body label, body li, body td,
    body em, body strong, body small,
    body input, body textarea, body select,
    body .stMarkdown,
    body [data-testid="stMarkdownContainer"],
    body [data-testid="stMarkdownContainer"] p,
    body [data-testid="stMarkdownContainer"] li,
    body [data-testid="stMarkdownContainer"] strong,
    body [data-testid="stMarkdownContainer"] em {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        letter-spacing: 0 !important;
        color: var(--text-primary);
    }

    body strong,
    body [data-testid="stMarkdownContainer"] strong {
        font-weight: 700 !important;
    }

    /* ---- CAPTION (Inter, muted, smaller) -------------------------- */
    body .stCaption,
    body [data-testid="stCaptionContainer"],
    body [data-testid="stCaptionContainer"] p,
    body [data-testid="stCaptionContainer"] small,
    body [data-testid="stCaptionContainer"] span:not([aria-hidden="true"]):not([class*="material"]):not([class*="Material"]),
    body small {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 400 !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
    }

    /* ---- SUBTITLE (masthead "Taste, with confidence.") ----------- */
    body .subtitle,
    body .subtitle * {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 500 !important;
        font-size: 1.06rem !important;
        color: var(--text-secondary) !important;
        margin: 10px 0 0 0 !important;
    }

    /* ---- Editorial page furniture -------------------------------- */
    .tab-heading {
        margin: 20px 0 18px 0;
        padding: 0 0 14px 0;
        border-bottom: 1px solid var(--card-border);
    }

    .tab-heading span {
        display: block;
        margin: 0 0 6px 0;
        font-family: var(--font-body);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--terracotta);
    }

    .tab-heading h2 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 2.15rem !important;
        line-height: 1 !important;
    }

    .tab-heading h2::after {
        display: none !important;
    }

    .tab-heading p {
        max-width: 620px;
        margin: 8px 0 0 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.96rem !important;
        line-height: 1.5 !important;
    }

    .cellar-snapshot,
    .editorial-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 18px 0 24px 0;
    }

    /* Column count follows the number of tiles, set by render_stat_grid
       via a stat-grid-cols-N modifier. Without this a 3-tile grid keeps
       the 4-column track and leaves an empty slot on the right. These
       come after the base rule so the matching count wins. */
    .stat-grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .stat-grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .stat-grid-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }

    .cellar-snapshot {
        padding: 14px 0;
        border-top: 1px solid var(--text-primary);
        border-bottom: 1px solid var(--card-border);
    }

    .stat-tile {
        min-height: 104px;
        padding: 16px 16px 14px 16px;
        border: 1px solid var(--card-border);
        border-top: 3px solid var(--line-strong);
        border-radius: var(--radius-card);
        background:
            linear-gradient(180deg, rgba(255, 252, 246, 0.96), rgba(250, 247, 239, 0.92));
        box-shadow: var(--shadow-card);
    }

    .stat-label,
    .stat-note {
        font-family: var(--font-body);
        font-style: normal;
        color: var(--text-secondary);
    }

    .stat-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .stat-value {
        margin-top: 8px;
        font-family: var(--font-body);
        font-size: 2.55rem;
        font-weight: 700;
        line-height: 0.95;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }

    .stat-note {
        margin-top: 8px;
        font-size: 0.86rem;
        line-height: 1.35;
    }

    .feature-profile-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
        gap: 14px;
        margin: 18px 0 24px 0;
    }

    .feature-profile-grid-3 {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .feature-profile-grid-4 {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }

    .feature-profile {
        border: 1px solid var(--card-border);
        border-radius: var(--radius-card);
        background: var(--card-bg);
        box-shadow: var(--shadow-card);
        padding: 16px;
        position: relative;
        overflow: hidden;
    }

    .feature-profile::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--terracotta);
    }

    .feature-profile.tone-white::before { background: var(--gold); }
    .feature-profile.tone-red::before { background: var(--wine); }
    .feature-profile.tone-rose::before { background: var(--rose); }
    .feature-profile.tone-orange::before { background: var(--orange); }
    .feature-profile.tone-neutral::before { background: var(--olive); }

    .feature-profile-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }

    .feature-profile-head h3 {
        margin: 0 !important;
        font-size: 1.32rem !important;
        line-height: 1.08 !important;
    }

    .feature-profile-head p {
        margin: 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        white-space: nowrap;
    }

    .feature-rows {
        display: grid;
        gap: 9px;
    }

    .feature-row {
        display: grid;
        grid-template-columns: 86px minmax(96px, 1fr) 36px;
        align-items: center;
        gap: 10px;
        min-height: 26px;
    }

    .feature-name,
    .feature-number {
        font-family: var(--font-body);
        font-size: 0.84rem;
        color: var(--text-secondary);
    }

    .feature-number {
        color: var(--text-primary);
        font-weight: 700;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .feature-track {
        display: block;
        height: 8px;
        border-radius: 999px;
        background: rgba(33, 26, 22, 0.09);
        overflow: hidden;
    }

    .feature-fill {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: var(--terracotta);
    }

    .tone-white .feature-fill { background: var(--gold); }
    .tone-red .feature-fill { background: var(--wine); }
    .tone-rose .feature-fill { background: var(--rose); }
    .tone-orange .feature-fill { background: var(--orange); }
    .tone-neutral .feature-fill { background: var(--olive); }

    .ranked-list {
        display: grid;
        gap: 10px;
        padding: 0;
        margin: 18px 0 24px 0;
        list-style: none;
        width: 100%;
    }

    /* Streamlit's markdown container pins direct-child <ol>/<ul> to
       `width: fit-content`, which makes the ranked list shrink-wrap to
       its longest row and leaves a ragged strip of dead space on the
       right of the Top Regions / Top Wines columns. It also overrides
       each <li>'s margin and padding (indenting the cards out of line
       with the heading and collapsing the padding so the score hugs the
       right border). Re-assert a full-width grid and the item's own
       margin/padding with selectors specific enough to beat those
       (emotion-cache) rules. */
    [data-testid="stMarkdownContainer"] ol.ranked-list {
        display: grid;
        width: 100%;
    }
    [data-testid="stMarkdownContainer"] li.ranked-item {
        margin: 0;
        padding: 12px 18px;
    }

    .ranked-item {
        display: grid;
        grid-template-columns: 40px minmax(0, 1fr) 82px;
        align-items: center;
        gap: 12px;
        min-height: 64px;
        padding: 12px 18px;
        border: 1px solid var(--card-border);
        border-left: 2px solid var(--wine);
        border-radius: var(--radius-card);
        background: rgba(255, 252, 246, 0.9);
        box-shadow: 0 1px 0 rgba(33, 26, 22, 0.04);
    }

    .ranked-index {
        font-family: var(--font-body);
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1;
        color: var(--wine);
        font-variant-numeric: tabular-nums;
    }

    .ranked-title {
        font-family: var(--font-body);
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.25;
    }

    .ranked-meta {
        margin-top: 4px;
        font-family: var(--font-body);
        font-size: 0.86rem;
        color: var(--text-secondary);
    }

    .ranked-value {
        text-align: right;
        min-width: 82px;
    }

    .ranked-value span {
        display: block;
        font-family: var(--font-body);
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }

    .ranked-value small {
        display: block;
        margin-top: 4px;
        font-family: var(--font-body);
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
    }

    .gallery-result-line {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin: 12px 0 18px 0;
        padding: 8px 0;
        border-top: 1px solid var(--text-primary);
        border-bottom: 1px solid var(--card-border);
        width: 100%;
    }

    .gallery-result-line span {
        font-family: var(--font-body);
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1;
        color: var(--text-primary);
    }

    .gallery-result-line p {
        margin: 0 !important;
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
    }

    /* ---- Tabs ------------------------------------------------------ */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0;
        border-top: 1px solid var(--card-border);
        border-bottom: 1px solid var(--card-border);
        background: transparent;
        padding: 2px 0;
    }

    [data-testid="stTabs"] [data-baseweb="tab"],
    [data-testid="stTabs"] [role="tab"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
        color: var(--text-secondary) !important;
        padding: 14px 18px !important;
        background: transparent !important;
        border-radius: 0;
        transition: color 0.15s ease, background 0.15s ease;
        letter-spacing: 0;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] p,
    [data-testid="stTabs"] [role="tab"] p {
        margin: 0 !important;
        color: inherit !important;
        font-family: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        line-height: 1.3 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab"]:hover,
    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--terracotta) !important;
        background: rgba(138, 31, 61, 0.06) !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--terracotta) !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        border-bottom: 2px solid var(--terracotta) !important;
    }

    /* ---- Buttons --------------------------------------------------- */
    .stButton > button,
    [data-testid="stPopover"] button,
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0 !important;
    }

    .stButton > button,
    [data-testid="stPopover"] button,
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"] {
        border-radius: var(--radius-button) !important;
        padding: 10px 18px !important;
        transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        border: 1px solid var(--card-border) !important;
        box-shadow: none !important;
    }

    .stButton > button p,
    [data-testid="stPopover"] button p,
    [data-testid="baseButton-primary"] p,
    [data-testid="baseButton-secondary"] p {
        margin: 0 !important;
        font-family: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        color: inherit !important;
    }

    /* Streamlit renders chevrons / icons inside buttons via
       <span class="material-icons-..."> elements using the Material
       Symbols font. Our font-family override on the button cascades
       to those spans, breaking the icon font and showing raw text
       like "expand_more". Reset font-family on any material-icons
       span so Streamlit's icon font wins. */
    [class*="material-icons"],
    [class*="material-symbols"],
    [data-testid="stIconMaterial"],
    button span[aria-hidden="true"],
    [role="button"] span[aria-hidden="true"],
    [data-testid="stExpander"] summary span[aria-hidden="true"],
    [data-testid="stPopover"] button span[aria-hidden="true"],
    [data-baseweb="select"] span[aria-hidden="true"] {
        font-family: 'Material Symbols Rounded',
                     'Material Symbols Outlined',
                     'Material Icons' !important;
        font-style: normal !important;
        font-weight: normal !important;
        letter-spacing: normal !important;
        line-height: 1 !important;
        text-transform: none !important;
    }

    /* Primary buttons - bordeaux */
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stPopover"] button[kind="primary"],
    [data-testid="stPopover"] [data-testid="baseButton-primary"] {
        background: var(--terracotta) !important;
        color: var(--text-on-accent) !important;
        border-color: var(--terracotta) !important;
    }
    .stButton > button[kind="primary"] *,
    [data-testid="baseButton-primary"] *,
    [data-testid="stPopover"] button[kind="primary"] *,
    [data-testid="stPopover"] [data-testid="baseButton-primary"] * {
        color: var(--text-on-accent) !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="stPopover"] button[kind="primary"]:hover {
        background: var(--terracotta-dark) !important;
        border-color: var(--terracotta-dark) !important;
        box-shadow: none !important;
    }

    /* Secondary buttons - outlined parchment */
    .stButton > button[kind="secondary"],
    [data-testid="baseButton-secondary"] {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }
    .stButton > button[kind="secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover {
        background: var(--terracotta-soft) !important;
        border-color: var(--terracotta) !important;
        color: var(--terracotta) !important;
    }

    /* Download / link buttons - Streamlit's testid for the inner
       button varies; cover the section and the link variant too. */
    [data-testid="stDownloadButton"] > button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stLinkButton"] > a,
    [data-testid="stLinkButton"] a,
    a[download],
    button[data-testid*="Download"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: var(--radius-button) !important;
        background: var(--card-bg) !important;
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
        padding: 10px 20px !important;
        text-decoration: none !important;
    }
    [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stLinkButton"] > a:hover,
    a[download]:hover {
        background: var(--terracotta-soft) !important;
        background-color: var(--terracotta-soft) !important;
        border-color: var(--terracotta) !important;
        color: var(--terracotta) !important;
    }

    /* ---- Popover content panel ----------------------------------- */
    /* BaseWeb renders popovers, menus, and tooltips in a portal layer
       at the bottom of <body>, NOT inside the trigger element. Selectors
       must be global, not scoped to the trigger. */

    /* The portal layer wrapper itself. Force parchment on every dropdown
       menu, sign-in popover, and any other BaseWeb popover that
       Streamlit renders. */
    [data-baseweb="layer"] [data-baseweb="popover"],
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] [role="dialog"],
    [data-baseweb="popover"] [data-baseweb="block"] {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }
    [data-baseweb="popover"] {
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius-card) !important;
        box-shadow: var(--shadow-card-hover) !important;
    }

    /* Force text-primary inside popovers, even on labels/captions
       that would otherwise inherit secondary colour against the
       old dark background. */
    [data-baseweb="popover"] p,
    [data-baseweb="popover"] label,
    [data-baseweb="popover"] [data-testid="stWidgetLabel"],
    [data-baseweb="popover"] h1,
    [data-baseweb="popover"] h2,
    [data-baseweb="popover"] h3,
    [data-baseweb="popover"] h4 {
        color: var(--text-primary) !important;
    }

    [data-baseweb="popover"] [data-testid="stButton"] button,
    [data-baseweb="popover"] [data-testid="stButton"] button p {
        white-space: nowrap !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
    }

    /* Dropdown menus (selectbox open state). The menu is portaled,
       so we target it globally - every menu role across the app
       inherits parchment. */
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul,
    [data-baseweb="menu"] li,
    ul[role="listbox"],
    [role="listbox"] {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius-button) !important;
        font-family: var(--font-body) !important;
    }
    [data-baseweb="menu"] [role="option"],
    ul[role="listbox"] [role="option"],
    [role="listbox"] li {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        padding: 8px 12px !important;
    }
    [data-baseweb="menu"] [role="option"][aria-selected="true"],
    [data-baseweb="menu"] [role="option"]:hover,
    ul[role="listbox"] [role="option"]:hover,
    [role="listbox"] li:hover {
        background: var(--terracotta-soft) !important;
        color: var(--terracotta) !important;
    }

    /* Tooltips - BaseWeb renders these in the portal layer too. */
    [data-baseweb="tooltip"],
    [role="tooltip"] {
        background: var(--text-primary) !important;
        color: var(--card-bg) !important;
        border-radius: var(--radius-button) !important;
        font-family: var(--font-body) !important;
        font-size: 0.85rem !important;
        padding: 6px 10px !important;
        box-shadow: var(--shadow-card-hover) !important;
    }

    /* Signed-in user pill - the popover trigger renders with a dark
       chip background by default. Override to a soft mineral tint so
       it reads as "you're signed in" without competing with the
       primary Sign-in button. */
    [data-testid="stPopover"] button:not([kind="primary"]) {
        background: var(--olive-soft) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--olive) !important;
        font-weight: 600 !important;
    }
    [data-testid="stPopover"] button:not([kind="primary"]):hover {
        background: var(--olive) !important;
        color: var(--text-on-accent) !important;
    }

    /* ---- File uploader ------------------------------------------- */
    /* Streamlit's drag-and-drop zone renders as a dark box by default.
       Override the dropzone background, border, and the inner text. */
    [data-testid="stFileUploader"] > section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: var(--card-bg) !important;
        border: 2px dashed var(--card-border) !important;
        border-radius: var(--radius-card) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stFileUploader"] > section:hover,
    [data-testid="stFileUploaderDropzone"]:hover {
        background: var(--terracotta-soft) !important;
        border-color: var(--terracotta) !important;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        color: var(--text-secondary) !important;
    }
    /* "Browse files" button inside the dropzone */
    [data-testid="stFileUploader"] button {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 600 !important;
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: var(--terracotta) !important;
        color: var(--text-on-accent) !important;
        border-color: var(--terracotta) !important;
    }

    /* ---- Radio buttons (used for input mode picker) -------------- */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-size: 1.05rem !important;
    }
    /* The unchecked circle */
    [data-testid="stRadio"] [role="radio"] {
        border-color: var(--card-border) !important;
    }
    /* The checked circle dot */
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        border-color: var(--terracotta) !important;
        background: var(--terracotta) !important;
    }

    /* ---- Toggle switch (Liked toggle) ---------------------------- */
    [data-testid="stToggle"] [role="switch"][aria-checked="true"] {
        background: var(--terracotta) !important;
    }

    /* ---- Sliders ------------------------------------------------- */
    [data-testid="stSlider"] [role="slider"] {
        background: var(--terracotta) !important;
        border-color: var(--terracotta) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
        background: var(--terracotta) !important;
    }

    /* ---- Image previews (st.image) -------------------------------- */
    /* st.image with width='stretch' fills its column at natural
       resolution, which makes uploaded wine photos enormous on
       wide screens. Cap to a reasonable preview width. */
    [data-testid="stImage"] {
        max-width: 320px !important;
        margin: 0 auto;
    }
    [data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: var(--radius-card) !important;
        border: 1px solid var(--card-border);
    }

    /* ---- Header auth right-alignment ----------------------------- */
    /* The header is a 4-to-1 column split with the auth popover in
       the right column. Streamlit places column content flush left
       by default; this makes Sign-in float in the middle of its
       column instead of hugging the page edge. Push content right. */
    [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stColumn"]:last-child [data-testid="stPopover"] {
        display: flex;
        justify-content: flex-end;
    }
    /* Fallback for browsers without :has() - apply to any popover
       inside a horizontal block's last column. */
    [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child {
        text-align: right;
    }

    /* ---- Code blocks (for any debug output) ---------------------- */
    code, pre, [data-testid="stCodeBlock"] {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-radius: 6px;
    }

    /* ---- Form controls -------------------------------------------- */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius-button) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--terracotta) !important;
        outline: none !important;
    }

    /* Placeholder text - the default is too pale on ivory. Bump to
       a readable mid-brown. */
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;  /* Firefox lowers placeholder opacity by default */
    }

    /* Selectbox - Streamlit nests multiple divs under data-testid;
       override the BaseWeb control wrapper specifically. */
    [data-testid="stSelectbox"] > div,
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] [role="combobox"] {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        letter-spacing: 0 !important;
    }

    [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p,
    [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        color: inherit !important;
        font-family: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
    }

    [data-testid="stSelectbox"] > div,
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] [role="combobox"] {
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius-button) !important;
    }

    /* Hide the text caret AND any value-container separator in
       selectboxes. Streamlit selectboxes are comboboxes; depending
       on the Streamlit version the trailing vertical bar is either
       the text input's caret or a BaseWeb separator element. Suppress
       both so the picker doesn't show a stray "All Regions|".
       The input stays functional - clicking still opens the menu. */
    [data-testid="stSelectbox"] input,
    [data-baseweb="select"] input {
        caret-color: transparent !important;
    }
    /* BaseWeb sometimes renders a vertical separator between the
       value and the dropdown arrow. Hide it. */
    [data-baseweb="select"] [data-baseweb="select-dropdown-indicator"] + div,
    [data-baseweb="select"] div[role="presentation"] {
        display: none !important;
    }
    /* Belt-and-suspenders: kill any 1px-wide pseudo-bar in the
       value container. */
    [data-baseweb="select"] [class*="Separator"],
    [data-baseweb="select"] hr {
        display: none !important;
    }

    /* Labels above inputs */
    [data-testid="stWidgetLabel"],
    [data-testid="InputInstructions"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        letter-spacing: 0 !important;
    }

    [data-testid="stWidgetLabel"] {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stWidgetLabel"] p {
        margin: 0 !important;
        color: inherit !important;
        font-family: inherit !important;
        font-weight: inherit !important;
    }

    [data-testid="InputInstructions"] {
        font-weight: 400 !important;
        color: var(--text-muted) !important;
    }

    /* ---- Expanders ----------------------------------------------- */
    [data-testid="stExpander"],
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary {
        background: var(--card-bg) !important;
        border-color: var(--card-border) !important;
        border-radius: var(--radius-card) !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        color: var(--text-primary) !important;
        letter-spacing: 0 !important;
    }

    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] summary p {
        margin: 0 !important;
        color: inherit !important;
        font-family: inherit !important;
        font-weight: inherit !important;
    }

    /* ---- Alerts / info / warning / error -------------------------- */
    [data-testid="stAlert"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stAlert"] {
        border-radius: var(--radius-card) !important;
        border: 1px solid var(--card-border) !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] li,
    [data-testid="stAlert"] strong,
    [data-testid="stAlert"] em {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        letter-spacing: 0 !important;
        color: var(--text-primary) !important;
    }

    /* ---- Metric (st.metric) --------------------------------------- */
    [data-testid="stMetric"] {
        background: rgba(255, 252, 246, 0.84) !important;
        border: 1px solid var(--card-border) !important;
        border-top: 2px solid rgba(33, 26, 22, 0.72) !important;
        border-radius: var(--radius-card) !important;
        padding: 14px 16px !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease, background 0.15s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(138, 31, 61, 0.36) !important;
        background: var(--card-bg) !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.05rem !important;
        line-height: 1.05 !important;
    }

    /* ---- Plotly charts ------------------------------------------- */
    [data-testid="stPlotlyChart"] {
        background:
            linear-gradient(180deg, rgba(255, 252, 246, 0.96), rgba(250, 247, 239, 0.9)) !important;
        border: 1px solid var(--card-border) !important;
        border-top: 3px solid var(--wine) !important;
        border-radius: var(--radius-card) !important;
        padding: 16px 16px 8px 16px !important;
        box-shadow: var(--shadow-card) !important;
    }

    [data-testid="stPlotlyChart"] .modebar {
        opacity: 0;
        transition: opacity 0.15s ease;
    }

    [data-testid="stPlotlyChart"]:hover .modebar,
    [data-testid="stPlotlyChart"] .modebar:hover {
        opacity: 1;
    }

    /* Re-assert Streamlit's icon font after widget text rules. */
    [class*="material-icons"],
    [class*="material-symbols"],
    [data-testid="stIconMaterial"],
    button span[aria-hidden="true"],
    [role="button"] span[aria-hidden="true"],
    [data-testid="stExpander"] summary span[aria-hidden="true"],
    [data-testid="stPopover"] button span[aria-hidden="true"],
    [data-baseweb="select"] span[aria-hidden="true"],
    [data-testid="stIconMaterial"] * {
        font-family: 'Material Symbols Rounded',
                     'Material Symbols Outlined',
                     'Material Icons' !important;
    }

    /* ---- Decorative horizontal divider --------------------------- */
    /* A plain printed rule. Decorative dividers made the app feel
       more restaurant menu than modern magazine. */
    hr {
        border: none !important;
        height: 1px !important;
        background: var(--card-border) !important;
        margin: 30px 0 !important;
        position: relative;
        overflow: visible !important;
    }

    /* ---- Mobile responsive ---------------------------------------- */
    @media (max-width: 768px) {
        .main-title {
            font-size: 3.1rem !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-size: 0.95rem !important;
            padding: 10px 12px !important;
        }
        .cellar-snapshot,
        .editorial-stat-grid,
        .feature-profile-grid {
            grid-template-columns: 1fr;
        }
        .stat-tile {
            min-height: auto;
        }
        .feature-profile-head {
            display: block;
        }
        .feature-profile-head p {
            margin-top: 4px !important;
            white-space: normal;
        }
        .ranked-item {
            grid-template-columns: 42px minmax(0, 1fr);
        }
        .ranked-value {
            grid-column: 2;
            text-align: left;
            margin-top: 2px;
        }
    }

    /* ---- Streamlit default header overrides ---------------------- */
    [data-testid="stToolbar"] {
        background: transparent !important;
    }
</style>
"""


_GALLERY_STYLES = """\
<style>
/* Remove gaps between Streamlit column containers */
.block-container [data-testid="column"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.block-container .element-container {
    margin-bottom: 0 !important;
}

.wine-card {
    font-family: var(--font-body) !important;
    font-style: normal !important;
    letter-spacing: 0 !important;
}

[data-testid="column"]:has(.wine-card-img),
[data-testid="column"]:has(.wine-card-img-placeholder) {
    background: rgba(255, 252, 246, 0.88);
    border: 1px solid var(--card-border);
    border-top: 3px solid var(--line-strong);
    border-radius: var(--radius-card);
    padding: 14px 14px 18px 14px !important;
    box-shadow: var(--shadow-card);
}

[data-testid="column"]:has(.wine-card-img):hover,
[data-testid="column"]:has(.wine-card-img-placeholder):hover {
    border-color: rgba(138, 31, 61, 0.28);
    box-shadow: var(--shadow-card-hover);
}


/* Wine Gallery Grid - back to 3-per-row with larger photos.
   minmax(280px, 1fr) gives 3 columns at 1100px, 4 at ~1450px. */
.wine-gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    grid-auto-rows: min-content;
    gap: 2rem;
    margin: 24px 0;
}

/* Wine card image - bottle photos should show the full label/bottle,
   not crop like generic product tiles. */
.wine-card-img,
.wine-card-img-placeholder {
    display: block;
    width: 100%;
    max-height: 380px;
    aspect-ratio: 4 / 5;
    object-fit: contain;
    border-radius: var(--radius-card);
    background:
        linear-gradient(160deg, rgba(236, 230, 218, 0.88), rgba(255, 252, 246, 0.96));
    border: 1px solid var(--card-border);
    margin: 0 0 14px 0;
    padding: 10px;
}

.wine-card-img-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    min-height: 280px;
    font-family: var(--font-body);
    color: var(--text-muted);
    position: relative;
    overflow: hidden;
    text-align: center;
}

.wine-card-img-placeholder::before {
    content: "";
    position: absolute;
    inset: 18px;
    border: 1px solid rgba(33, 26, 22, 0.18);
    border-radius: 6px;
}

.placeholder-kicker {
    font-family: var(--font-body);
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--terracotta);
    position: relative;
}

.placeholder-name {
    max-width: 72%;
    font-family: var(--font-body);
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1.05;
    color: var(--text-primary);
    position: relative;
}

.tone-white .placeholder-kicker { color: var(--gold); }
.tone-red .placeholder-kicker { color: var(--wine); }
.tone-rose .placeholder-kicker { color: var(--rose); }
.tone-orange .placeholder-kicker { color: var(--orange); }

.wine-card-title {
    margin: 14px 0 5px 0;
    font-family: var(--font-body);
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.05;
}

.wine-card-meta,
.wine-card-location {
    margin: 0;
    font-family: var(--font-body);
    color: var(--text-secondary);
}

.wine-card-meta {
    font-size: 0.86rem;
    font-weight: 600;
}

.wine-card-location {
    margin-top: 4px;
    font-size: 0.8rem;
}

.wine-card-facts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    margin: 14px 0 10px 0;
    border-top: 1px solid var(--card-border);
    border-bottom: 1px solid var(--card-border);
}

.wine-card-fact {
    padding: 10px 8px 10px 0;
}

.wine-card-fact + .wine-card-fact {
    padding-left: 12px;
    border-left: 1px solid var(--card-border);
}

.wine-card-fact span {
    display: block;
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-secondary);
}

.wine-card-fact strong {
    display: block;
    margin-top: 4px;
    font-family: var(--font-body) !important;
    font-size: 1.18rem;
    font-weight: 700 !important;
    line-height: 1;
    color: var(--text-primary) !important;
    font-variant-numeric: tabular-nums;
}

.wine-card-notes {
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
    font-family: var(--font-body);
}

.icon-row {
    min-height: 24px;
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 8px 0;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 2px 8px;
    border: 1px solid rgba(138, 31, 61, 0.22);
    border-radius: 6px;
    background: rgba(138, 31, 61, 0.07);
    color: var(--terracotta);
    font-family: var(--font-body);
    font-size: 12px;
    font-style: normal;
    font-weight: 600;
    line-height: 1.2;
}

.wine-card-footer {
    margin-top: auto;
}
</style>
"""

_SELECTBOX_CARET_FIX = """
<style>
.stSelectbox div[data-baseweb="select"] input {
    position: absolute !important;
    left: -9999px !important;
    width: 0 !important;
    opacity: 0 !important;
    caret-color: transparent !important;
}
</style>
"""


def apply_global_styles() -> None:
    """Inject the main app theme.

    Call once at app boot, after `st.set_page_config`. Re-calling is
    harmless but wasteful - Streamlit will re-emit the `<style>` tag
    on every rerun, which the browser deduplicates by content.
    """
    st.markdown(_FONT_LINKS, unsafe_allow_html=True)
    st.markdown(_GLOBAL_STYLES, unsafe_allow_html=True)


def apply_gallery_styles() -> None:
    """Inject CSS scoped to the Wine Gallery view.

    Call at the top of the gallery tab body, before rendering wine
    cards. Selectors target the grid layout (`.wine-gallery-grid`)
    and per-card elements (`.wine-card-notes`, `.icon-row`,
    `.wine-card-footer`).
    """
    st.markdown(_GALLERY_STYLES, unsafe_allow_html=True)

    st.markdown(_SELECTBOX_CARET_FIX, unsafe_allow_html=True)
