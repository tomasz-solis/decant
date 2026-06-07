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

Phase 4 - Mediterranean light theme:
- Cream background (#FAF6F0), warm white cards (#FFFDF8)
- Terracotta primary (#C2410C), olive accent (#65733E),
  deep wine red for verdicts (#7C2D12)
- Playfair Display for h1-h4 headings and metric values; DM Sans
  for body text, widgets, cards, labels, and controls
- Restrained shadows, paper-like card surfaces
"""

from __future__ import annotations

import streamlit as st


_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=DM+Sans:wght@400;600;700&'
    'family=Playfair+Display:wght@700&display=swap" '
    'rel="stylesheet">'
)


_GLOBAL_STYLES = """
<style>
    /* ---- FONT LOAD ------------------------------------------------- */
    /* Loaded separately via st.markdown(_FONT_LINKS) before this
       stylesheet. Keep external font loading out of this style block. */

    /* ---- Mediterranean light palette ------------------------------- */
    :root {
        --bg-primary: #FAF6F0;         /* cream - baked paper, not stark white */
        --bg-secondary: #F3EDE3;       /* slightly deeper for tab nav, sidebar */
        --card-bg: #FFFDF8;            /* warm white for raised surfaces */
        --card-border: #E8DFCF;        /* hairline between card and background */

        --terracotta: #C2410C;         /* primary accent - restaurant signage */
        --terracotta-dark: #9A330A;    /* hover state for terracotta controls */
        --terracotta-soft: #FED7AA;    /* terracotta tint for hover/highlight */
        --olive: #65733E;              /* secondary accent - herbal */
        --olive-soft: #DCE3C4;         /* olive tint */
        --wine: #7C2D12;               /* deep red for verdicts, liked badge */
        --wine-fill: rgba(124, 45, 18, 0.4);

        --text-primary: #3D2817;       /* warm dark brown, not pure black */
        --text-secondary: #5C4D3F;     /* darker brown for captions/subtitles - readable on cream */
        --text-muted: #8B7E6D;         /* fainter, for tertiary info only (timestamps, hints) */
        --text-on-accent: #FFFFFF;     /* text on terracotta/olive/wine surfaces */

        --shadow-card: 0 2px 8px rgba(120, 60, 30, 0.06);
        --shadow-card-hover: 0 4px 14px rgba(120, 60, 30, 0.10);

        --radius-card: 12px;
        --radius-button: 8px;

        --font-display: 'Playfair Display', Georgia, 'Times New Roman', serif;
        --font-body: 'DM Sans', system-ui, -apple-system, sans-serif;
    }

    /* ---- Global base ---------------------------------------------- */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 400;
    }

    /* Subtle paper texture - a barely-visible noise pattern gives
       the cream background actual depth instead of feeling flat.
       Inline SVG noise so we don't depend on an external asset. */
    [data-testid="stAppViewContainer"] {
        background-image:
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.24  0 0 0 0 0.16  0 0 0 0 0.09  0 0 0 0.025 0'/></filter><rect width='200' height='200' filter='url(%23n)'/></svg>") !important;
        background-blend-mode: multiply;
        background-size: 200px 200px;
    }

    /* Menu-masthead bar: a thin terracotta-to-olive band at the very
       top of the page. Subtle visual reference to printed menus. */
    .stApp::before {
        content: "";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(to right,
            var(--terracotta) 0%,
            var(--terracotta) 40%,
            var(--olive) 60%,
            var(--olive) 100%);
        z-index: 1000;
        pointer-events: none;
    }

    /* Streamlit injects a top header bar; on light theme we want it
       to blend with the page rather than show as a darker strip. */
    [data-testid="stHeader"] {
        background: var(--bg-primary) !important;
    }

    /* ---- DISPLAY HEADINGS (Playfair) ------------------------------ */
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
        font-size: 3.4rem !important;
        line-height: 1.05 !important;
        margin: 0 !important;
    }

    /* Section titles (h2 in tab bodies) get a soft terracotta
       underline that ties them to the active-tab indicator. */
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
        width: 48px;
        height: 2px;
        background: var(--terracotta);
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

    /* ---- LOWER-LEVEL TITLES (DM Sans) ----------------------------- */
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

    /* ---- STREAMLIT WIDGET TYPOGRAPHY (DM Sans) ------------------- */
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

    /* ---- METRIC VALUE (Playfair big numbers) ---------------------- */
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

    /* ---- METRIC LABEL (DM Sans uppercase) ------------------------- */
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

    /* ---- BODY BASELINE (DM Sans, upright, everything else) -------- */
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

    /* ---- CAPTION (DM Sans, muted, smaller) ------------------------ */
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
        font-weight: 400 !important;
        font-size: 1rem !important;
        color: var(--text-secondary) !important;
        margin: 4px 0 0 0 !important;
    }

    /* ---- Tabs ------------------------------------------------------ */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--card-border);
        background: transparent;
    }

    [data-testid="stTabs"] [data-baseweb="tab"],
    [data-testid="stTabs"] [role="tab"] {
        font-family: var(--font-body) !important;
        font-style: normal !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: var(--text-secondary) !important;
        padding: 12px 18px !important;
        background: transparent !important;
        border-radius: var(--radius-button) var(--radius-button) 0 0;
        transition: color 0.15s ease;
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
    }

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--terracotta) !important;
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
        padding: 10px 20px !important;
        transition: all 0.15s ease !important;
        border: 1px solid var(--card-border) !important;
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

    /* Primary buttons - terracotta */
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stPopover"] button[kind="primary"],
    [data-testid="stPopover"] [data-testid="baseButton-primary"] {
        background: var(--terracotta) !important;
        color: var(--text-on-accent) !important;
        border-color: var(--terracotta) !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="stPopover"] button[kind="primary"]:hover {
        background: var(--terracotta-dark) !important;
        border-color: var(--terracotta-dark) !important;
        box-shadow: var(--shadow-card-hover);
    }

    /* Secondary buttons - outlined cream */
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

    /* The portal layer wrapper itself. Force cream on every dropdown
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
       inherits cream. */
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
       chip background by default. Override to a soft olive tint so
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

    /* Placeholder text - the default is too pale on cream. Bump to
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
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius-card) !important;
        padding: 16px !important;
        box-shadow: var(--shadow-card) !important;
        transition: transform 0.15s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-card-hover) !important;
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
    /* A thin terracotta line with a tiny olive dot in the middle.
       Restaurant menus often have decorated rules between courses;
       this is the restrained version. */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(to right,
            transparent 0%,
            var(--card-border) 20%,
            var(--card-border) 48%,
            var(--olive) 50%,
            var(--card-border) 52%,
            var(--card-border) 80%,
            transparent 100%) !important;
        margin: 32px 0 !important;
        position: relative;
        overflow: visible !important;
    }

    /* ---- Mobile responsive ---------------------------------------- */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.4rem !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-size: 0.95rem !important;
            padding: 10px 12px !important;
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


/* Wine Gallery Grid - back to 3-per-row with larger photos.
   minmax(280px, 1fr) gives 3 columns at 1100px, 4 at ~1450px. */
.wine-gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    grid-auto-rows: min-content;
    gap: 2rem;
    margin: 24px 0;
}

/* Wine card image - bottle photos benefit from the larger format.
   max-height kept generous (400px) so portrait photos display well
   without exploding to natural size. */
.wine-card-img,
.wine-card-img-placeholder {
    display: block;
    width: 100%;
    max-height: 400px;
    aspect-ratio: 3 / 4;
    object-fit: cover;
    border-radius: var(--radius-card);
    background: var(--bg-secondary);
    border: 1px solid var(--card-border);
    margin: 0 0 12px 0;
}

.wine-card-img-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-body);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-muted);
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
    border: 1px solid var(--card-border);
    border-radius: 999px;
    background: var(--card-bg);
    color: var(--text-secondary);
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
