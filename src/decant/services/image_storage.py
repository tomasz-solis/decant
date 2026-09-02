"""Image storage helpers: save uploaded wine images to disk and look them up later.

These functions operate on the local filesystem (`data/wine_images/`). They
do not touch Supabase Storage; the original code that did is in
`decant.database` and is dead in the live app.

`save_wine_image` calls `st.error` on failure to surface the error in the
UI. This is a slight smell (a service touching the UI layer) but is
preserved from the original code to keep behaviour identical during the
Phase 3 refactor. A later pass should refactor it to return a
result/error tuple and let the caller render.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image


WINE_IMAGE_DIR = Path("data/wine_images")
SUPPORTED_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
MAX_IMAGE_WIDTH = 800
JPEG_QUALITY = 85


def _safe_filename(wine_name: str) -> str:
    """Slugify a wine name for filesystem use."""
    safe = re.sub(r"[^\w\s-]", "", wine_name.lower())
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe


def get_wine_image_url(wine_name: str, producer: str) -> Optional[str]:
    """Build a Vivino search URL for a wine.

    Returns a search-page URL, not a direct image URL. The original
    code is named `_url` because the intent was to fetch a thumbnail
    in future; for now it's just a deep link to Vivino's search.
    """
    try:
        search_query = f"{producer} {wine_name}".replace(" ", "+")
        return f"https://www.vivino.com/search/wines?q={search_query}"
    except Exception:
        return None


def get_wine_image_path(wine_name: str) -> Optional[str]:
    """Return the local path to a wine's saved image, or None if absent.

    Looks for files matching the slugified name with any of the
    supported extensions.
    """
    safe_name = _safe_filename(wine_name)
    for ext in SUPPORTED_EXTENSIONS:
        image_path = WINE_IMAGE_DIR / f"{safe_name}.{ext}"
        if image_path.exists():
            return str(image_path)
    return None


def save_wine_image(uploaded_file, wine_name: str) -> Optional[str]:
    """Save an uploaded image under a slugified wine name.

    Resizes images wider than `MAX_IMAGE_WIDTH` and converts RGBA to
    RGB-with-white-background so JPEG saves work. Returns the saved
    path on success, None on failure (and shows the error in the UI).
    """
    try:
        safe_name = _safe_filename(wine_name)

        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            file_ext = "jpg"

        WINE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        image_path = WINE_IMAGE_DIR / f"{safe_name}.{file_ext}"

        image = Image.open(uploaded_file)

        if image.width > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / image.width
            new_height = int(image.height * ratio)
            image = image.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        image.save(image_path, quality=JPEG_QUALITY, optimize=True)
        return str(image_path)
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None
