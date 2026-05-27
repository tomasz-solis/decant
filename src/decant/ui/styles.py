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

A future Phase 4 will likely rework large parts of `_GLOBAL_STYLES`
(visual identity change). The split into two functions keeps Phase
4 surgical: change the theme constants in `_GLOBAL_STYLES`, leave
the gallery layout alone.
"""

from __future__ import annotations

import streamlit as st


_GLOBAL_STYLES = """\
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800;900&display=swap');

    /* 2026 Bento Dark Mode Color System */
    :root {
        --bg-primary: #0F0F12;
        --bg-secondary: #1A1A1E;
        --card-bg: rgba(255, 255, 255, 0.05);
        --wine-red: #800020;
        --accent-red: #8B0000;
        --accent-red-glow: rgba(128, 0, 32, 0.5);
        --text-primary: #E8E8EB;
        --text-secondary: #A0A0A8;
        --border-subtle: rgba(255, 255, 255, 0.1);
        --border-radius: 16px;
    }

    /* Global Typography - Geist with Inter fallback */
    * {
        font-family: 'Geist', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Global Background */
    body, .main, [data-testid="stAppViewContainer"] {
        background-color: #0F0F12 !important;
    }

    /* Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container */
    .main {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* 2026 Bento Glassmorphic Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }

    /* Hero Card with Wine Red Glow */
    .glass-card.glow {
        box-shadow: 0 8px 32px rgba(128, 0, 32, 0.5),
                    0 0 80px rgba(128, 0, 32, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* Radial Gradient Text Effect for Match Likelihood */
    .match-score-gradient {
        background: radial-gradient(circle at 30% 50%, #FF1744 0%, #800020 50%, #4A0012 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        letter-spacing: -3px;
        text-shadow: 0 0 40px rgba(255, 23, 68, 0.3);
    }

    /* Fallback for browsers without backdrop-filter */
    @supports not (backdrop-filter: blur(10px)) {
        .glass-card {
            background: rgba(26, 26, 30, 0.95);
        }
    }

    /* Header Styling */
    .main-title {
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        color: var(--text-primary);
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1em;
        margin-bottom: 32px;
        font-weight: 500;
    }

    /* 2026 Bento Button Styling */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.1em;
        font-weight: 600;
        border-radius: 12px;
        margin: 10px 0;
        background: linear-gradient(135deg, var(--wine-red) 0%, var(--accent-red) 100%);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 16px rgba(128, 0, 32, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #A60028 0%, #D00000 100%);
        box-shadow: 0 6px 24px rgba(128, 0, 32, 0.6),
                    0 0 40px rgba(128, 0, 32, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed var(--accent-red);
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        background: var(--card-bg);
    }

    /* Dark Mode Overrides for Streamlit Components */
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        color: var(--text-primary);
    }

    .stSelectbox > div > div {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
    }

    /* Plotly Chart Container */
    .js-plotly-plot {
        background-color: transparent !important;
    }

    /* Bento Grid Layout */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 24px 0;
    }

    /* Wine Gallery Card - Flex Container for Sticky Footer */
    .wine-card {
        padding: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .wine-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 16px 48px rgba(128, 0, 32, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .wine-card h4 {
        font-weight: 600;
        line-height: 1.3;
    }

    .wine-card .badge {
        display: inline-block;
        transition: transform 0.2s;
    }

    .wine-card:hover .badge {
        transform: scale(1.05);
    }

    /* Wine Card Image - Strict 350px Container */
    .wine-card-img {
        height: 350px;
        width: 100%;
        object-fit: contain;
        background: #0a0a0a;
        border-radius: 8px;
    }

    .wine-card-img-placeholder {
        height: 350px;
        width: 100%;
        background: rgba(139, 0, 0, 0.1);
        border: 2px solid rgba(139, 0, 0, 0.3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 60px;
    }

    /* Wine Card Content Wrapper - Full Height Flex */
    .wine-card-content {
        display: flex;
        flex-direction: column;
        height: 100%;
        gap: 12px;
    }

    /* Sticky Footer - Pushes Bottom Elements to Same Level */
    .wine-card-footer {
        margin-top: auto;
    }

    /* Icon Row */
    .icon-row {
        display: flex;
        gap: 8px;
        margin: 8px 0;
    }

    /* Seal of Approval */
    .seal-of-approval {
        position: relative;
        overflow: hidden;
    }

    .seal-of-approval::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(139, 0, 0, 0.2), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0%, 100% { left: -100%; }
        50% { left: 100%; }
    }

    /* Responsive Breakpoints - Mobile */
    @media (max-width: 768px) {
        /* Compact header for more screen real estate */
        .main-title {
            font-size: 1.5em;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        }

        .subtitle {
            font-size: 0.85em;
            margin-bottom: 16px;
        }

        /* Tighter spacing on mobile */
        .glass-card {
            padding: 12px;
            margin: 8px 0;
            border-radius: 12px;
        }

        /* Larger buttons for touch targets */
        .stButton > button {
            height: 56px !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            border-radius: 12px;
        }

        /* Wine images - optimized for mobile viewing */
        .wine-card-img, .wine-card-img-placeholder {
            height: 240px;  /* Smaller on mobile to see more content */
        }

        /* Single column layout - essential for shop browsing */
        .bento-grid {
            grid-template-columns: 1fr !important;
            gap: 12px;
        }

        /* Touch targets (44px minimum per Apple HIG) */
        .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
            min-height: 48px !important;
        }

        /* File uploader - make it huge and obvious for quick photo capture */
        .stFileUploader {
            padding: 24px !important;
            margin: 16px 0 !important;
        }

        .stFileUploader label {
            font-size: 1.2em !important;
            font-weight: 600 !important;
        }

        /* Sidebar collapsed by default on mobile */
        [data-testid="stSidebar"] {
            min-width: 0;
        }

        [data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: -21rem;
        }

        /* Main content full width on mobile */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }

        /* Columns stack vertically on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }

        /* Force stacking for multi-column layouts (3+ columns) */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 8px !important;
        }

        /* Metrics more compact on mobile */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            padding: 8px;
            border-radius: 8px;
        }

        /* Tab navigation bigger for easier tapping */
        .stTabs [data-baseweb="tab-list"] button {
            min-height: 48px !important;
            font-size: 1em !important;
        }

        /* Image preview on mobile - contain to screen */
        img {
            max-width: 100% !important;
            height: auto !important;
        }
    }

    @media (min-width: 769px) and (max-width: 1200px) {
        .bento-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .wine-card-img, .wine-card-img-placeholder {
            height: 320px;  /* Medium height for tablets */
        }
    }

    @media (min-width: 1201px) {
        .bento-grid {
            grid-template-columns: repeat(4, 1fr);
        }
    }

    /* Landscape Mobile Optimization - For horizontal phone in shop */
    @media (max-width: 900px) and (orientation: landscape) {
        .main-title {
            font-size: 1.2em;
            margin-bottom: 2px;
        }
        .subtitle {
            font-size: 0.8em;
            margin-bottom: 8px;
        }
        .glass-card {
            padding: 8px;
            margin: 6px 0;
        }
        .stButton > button {
            height: 44px !important;
            font-size: 0.95em !important;
        }
        /* Compact metrics in landscape */
        [data-testid="stMetric"] {
            padding: 6px;
        }
    }

    /* Small phones (iPhone SE, etc) - Extra compact */
    @media (max-width: 375px) {
        .main-title {
            font-size: 1.3em;
        }
        .glass-card {
            padding: 10px;
        }
        .stButton > button {
            height: 52px !important;
            font-size: 1em !important;
        }
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

/* Wine Gallery Grid */
.wine-gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    grid-auto-rows: min-content;
    gap: 2.5rem;
    margin: 24px 0;
}

.wine-card-notes {
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    color: #A0A0A8;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
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
