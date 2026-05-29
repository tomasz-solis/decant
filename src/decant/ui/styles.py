"""Inline CSS for the Streamlit UI.

Two functions, two scopes:

- `apply_global_styles()` ships the main theme — typography, colour
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
`unsafe_allow_html=True` here is benign — the markup is a static
string we control, not user-supplied content.

Phase 4 — Mediterranean light theme:
- Cream background (#FAF6F0), warm white cards (#FFFDF8)
- Terracotta primary (#C2410C), olive accent (#65733E),
  deep wine red for verdicts (#7C2D12)
- Playfair Display for headings; DM Sans for body
- Restrained shadows, paper-like card surfaces
"""

from __future__ import annotations

from matplotlib import style
import streamlit as st


_GLOBAL_STYLES = """
<style>
    /* ---- Fonts ----------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');

    /* ---- Mediterranean light palette ------------------------------- */
    :root {
        --bg-primary: #FAF6F0;         /* cream — baked paper, not stark white */
        --bg-secondary: #F3EDE3;       /* slightly deeper for tab nav, sidebar */
        --card-bg: #FFFDF8;            /* warm white for raised surfaces */
        --card-border: #E8DFCF;        /* hairline between card and background */

        --terracotta: #C2410C;         /* primary accent — restaurant signage */
        --terracotta-soft: #FED7AA;    /* terracotta tint for hover/highlight */
        --olive: #65733E;              /* secondary accent — herbal */
        --olive-soft: #DCE3C4;         /* olive tint */
        --wine: #7C2D12;               /* deep red for verdicts, liked badge */

        --text-primary: #3D2817;       /* warm dark brown, not pure black */
        --text-secondary: #5C4D3F;     /* darker brown for captions/subtitles — readable on cream */
        --text-muted: #8B7E6D;         /* fainter, for tertiary info only (timestamps, hints) */

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
    }

    /* Subtle paper texture — a barely-visible noise pattern gives
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

    /* ---- Typography ----------------------------------------------- */
    h1, h2, h3, h4, h5, h6,
    .main-title,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        font-family: var(--font-display) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }

    .main-title {
        font-size: clamp(2.4rem, 5vw, 3.4rem) !important;
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

    /* The italicised kicker line that follows section titles
       (the st.caption right under a heading). */
    [data-testid="stMarkdownContainer"] h2 + p,
    [data-testid="stCaptionContainer"] p {
        font-family: var(--font-display) !important;
        font-style: italic;
        color: var(--text-secondary) !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
    }

    .subtitle {
        font-family: var(--font-display) !important;
        font-style: italic;
        font-weight: 400 !important;
        color: var(--text-secondary) !important;
        font-size: clamp(0.95rem, 2vw, 1.1rem) !important;
        margin: 4px 0 0 0 !important;
    }

    /* Body text — DM Sans inherited; make sure Streamlit's own
       paragraph styling doesn't pull in default fonts. */
    p, span, div, label, .stMarkdown, [data-testid="stMarkdownContainer"] p {
        font-family: var(--font-body) !important;
        color: var(--text-primary);
    }

    .stCaption, [data-testid="stCaptionContainer"] p, small {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    /* ---- Tabs ------------------------------------------------------ */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--card-border);
        background: transparent;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-family: var(--font-display) !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: var(--text-secondary) !important;
        padding: 12px 18px !important;
        background: transparent !important;
        border-radius: var(--radius-button) var(--radius-button) 0 0;
        transition: color 0.15s ease;
    }

    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: var(--terracotta) !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color: var(--terracotta) !important;
        border-bottom: 2px solid var(--terracotta) !important;
    }

    /* ---- Buttons --------------------------------------------------- */
    .stButton > button,
    [data-testid="stPopover"] button[data-testid="stPopoverButton"],
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"] {
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-button) !important;
        padding: 10px 20px !important;
        transition: all 0.15s ease !important;
        border: 1px solid var(--card-border) !important;
    }

    /* CRITICAL: Streamlit renders chevrons / icons inside buttons via
       <span class="material-icons-..."> elements using the Material
       Symbols font. Our font-family override on the button cascades
       to those spans, breaking the icon font and showing raw text
       like "expand_more". Reset font-family on any material-icons
       span so Streamlit's icon font wins. */
    [class*="material-icons"],
    [class*="material-symbols"],
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded',
                     'Material Symbols Outlined',
                     'Material Icons' !important;
    }

    /* Primary buttons — terracotta */
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stPopover"] button[kind="primary"],
    [data-testid="stPopover"] [data-testid="baseButton-primary"] {
        background: var(--terracotta) !important;
        color: #FFFFFF !important;
        border-color: var(--terracotta) !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="stPopover"] button[kind="primary"]:hover {
        background: #9A330A !important;  /* darker terracotta */
        border-color: #9A330A !important;
        box-shadow: var(--shadow-card-hover);
    }

    /* Secondary buttons — outlined cream */
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

    /* Download / link buttons — Streamlit's testid for the inner
       button varies; cover the section and the link variant too. */
    [data-testid="stDownloadButton"] > button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stLinkButton"] > a,
    [data-testid="stLinkButton"] a,
    a[download],
    button[data-testid*="Download"] {
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
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

    /* Dropdown menus (selectbox open state). The menu is portaled,
       so we target it globally — every menu role across the app
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

    /* Tooltips — BaseWeb renders these in the portal layer too. */
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

    /* Signed-in user pill — the popover trigger renders with a dark
       chip background by default. Override to a soft olive tint so
       it reads as "you're signed in" without competing with the
       primary Sign-in button. */
    [data-testid="stPopover"] button:not([kind="primary"]) {
        background: var(--olive-soft) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--olive) !important;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
    }
    [data-testid="stPopover"] button:not([kind="primary"]):hover {
        background: var(--olive) !important;
        color: #FFFFFF !important;
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
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text-secondary) !important;
    }
    /* "Browse files" button inside the dropzone */
    [data-testid="stFileUploader"] button {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: var(--terracotta) !important;
        color: #FFFFFF !important;
        border-color: var(--terracotta) !important;
    }

    /* ---- Radio buttons (used for input mode picker) -------------- */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
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
    /* Fallback for browsers without :has() — apply to any popover
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
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--terracotta) !important;
        outline: none !important;
    }

    /* Placeholder text — the default is too pale on cream. Bump to
       a readable mid-brown. */
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;  /* Firefox lowers placeholder opacity by default */
    }

    /* Selectbox — Streamlit nests multiple divs under data-testid;
       override the BaseWeb control wrapper specifically. */
    [data-testid="stSelectbox"] > div,
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] [role="combobox"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius-button) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }

    /* Hide the text caret AND any value-container separator in
       selectboxes. Streamlit selectboxes are comboboxes; depending
       on the Streamlit version the trailing vertical bar is either
       the text input's caret or a BaseWeb separator element. Suppress
       both so the picker doesn't show a stray "All Regions|".
       The input stays functional — clicking still opens the menu. */
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
    [data-testid="stWidgetLabel"] {
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* ---- Alerts / info / warning / error -------------------------- */
    [data-testid="stAlert"] {
        border-radius: var(--radius-card) !important;
        border: 1px solid var(--card-border) !important;
        font-family: var(--font-body) !important;
    }

    /* ---- Metric (st.metric) --------------------------------------- */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-card);
        padding: 16px;
        box-shadow: var(--shadow-card);
        transition: transform 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-card-hover);
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-body) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
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


/* Wine Gallery Grid — back to 3-per-row with larger photos.
   minmax(280px, 1fr) gives 3 columns at 1100px, 4 at ~1450px. */
.wine-gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    grid-auto-rows: min-content;
    gap: 2rem;
    margin: 24px 0;
}

/* Wine card image — bottle photos benefit from the larger format.
   max-height kept generous (400px) so portrait photos display well
   without exploding to natural size. */
.wine-card-img,
.wine-card-img-placeholder {
    display: block;
    width: 100%;
    max-height: 400px;
    aspect-ratio: 3 / 4;
    object-fit: cover;
    border-radius: 12px;
    background: #F3EDE3;
    border: 1px solid #E8DFCF;
    margin: 0 0 12px 0;
}

.wine-card-img-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4.5rem;
    color: #C2410C;
    opacity: 0.5;
}

.wine-card-notes {
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    color: #78716C;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
    font-family: 'DM Sans', system-ui, sans-serif;
}

.icon-row {
    min-height: 24px;
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 8px 0;
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
    harmless but wasteful — Streamlit will re-emit the `<style>` tag
    on every rerun, which the browser deduplicates by content.
    """
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
