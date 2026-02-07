"""
PostgreSQL database module for Decant wine storage.

Provides persistent storage on Supabase for wine tasting data,
replacing ephemeral CSV storage on Streamlit Cloud.
"""

import os
import psycopg
import pandas as pd
from typing import Optional, Dict, Any, BinaryIO
import streamlit as st
from supabase import create_client, Client
from io import BytesIO
import re


def get_database_url() -> Optional[str]:
    """Get database URL from Streamlit secrets or environment."""
    try:
        return st.secrets["DATABASE_URL"]
    except (FileNotFoundError, KeyError):
        return os.getenv("DATABASE_URL")


def get_connection():
    """Create database connection with error handling."""
    database_url = get_database_url()

    if not database_url:
        raise ValueError("DATABASE_URL not found in secrets or environment")

    try:
        conn = psycopg.connect(database_url)
        return conn
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}")


def init_database():
    """Initialize database schema (create wines table if not exists)."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Create wines table with full schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wines (
                    id SERIAL PRIMARY KEY,
                    wine_name TEXT NOT NULL,
                    producer TEXT,
                    vintage FLOAT,
                    notes TEXT,
                    score FLOAT NOT NULL,
                    liked BOOLEAN NOT NULL,
                    price FLOAT,
                    country TEXT,
                    region TEXT,
                    wine_color TEXT,
                    is_sparkling BOOLEAN,
                    is_natural BOOLEAN,
                    sweetness TEXT,
                    acidity FLOAT,
                    minerality FLOAT,
                    fruitiness FLOAT,
                    tannin FLOAT,
                    body FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create index on wine_name for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_wine_name ON wines(wine_name);
            """)

            # Create index on liked for filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_liked ON wines(liked);
            """)

            conn.commit()


def add_wine(wine_data: Dict[str, Any]) -> bool:
    """Add a wine to the database."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO wines (
                    wine_name, producer, vintage, notes, score, liked, price,
                    country, region, wine_color, is_sparkling, is_natural, sweetness,
                    acidity, minerality, fruitiness, tannin, body
                ) VALUES (
                    %(wine_name)s, %(producer)s, %(vintage)s, %(notes)s, %(score)s, %(liked)s, %(price)s,
                    %(country)s, %(region)s, %(wine_color)s, %(is_sparkling)s, %(is_natural)s, %(sweetness)s,
                    %(acidity)s, %(minerality)s, %(fruitiness)s, %(tannin)s, %(body)s
                )
            """, wine_data)

            conn.commit()
            return True


def get_all_wines() -> pd.DataFrame:
    """Retrieve all wines as a pandas DataFrame."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    wine_name, producer, vintage, notes, score, liked, price,
                    country, region, wine_color, is_sparkling, is_natural, sweetness,
                    acidity, minerality, fruitiness, tannin, body
                FROM wines
                ORDER BY created_at DESC
            """)

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                # Return empty DataFrame with correct schema
                return pd.DataFrame(columns=[
                    'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
                    'country', 'region', 'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
                    'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
                ])

            df = pd.DataFrame(rows, columns=columns)

            # Convert types to match CSV format
            df['liked'] = df['liked'].astype(bool)
            df['price'] = df['price'].astype(float)
            df['score'] = df['score'].astype(float)

            return df


def delete_wine(wine_name: str, vintage: Optional[float] = None) -> bool:
    """Delete a wine from the database."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            if vintage:
                cursor.execute("""
                    DELETE FROM wines
                    WHERE wine_name = %s AND vintage = %s
                """, (wine_name, vintage))
            else:
                cursor.execute("""
                    DELETE FROM wines
                    WHERE wine_name = %s
                """, (wine_name,))

            conn.commit()
            return True


def wine_exists(wine_name: str, vintage: Optional[float] = None) -> bool:
    """Check if a wine already exists in the database."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            if vintage:
                cursor.execute("""
                    SELECT COUNT(*) FROM wines
                    WHERE wine_name = %s AND vintage = %s
                """, (wine_name, vintage))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM wines
                    WHERE wine_name = %s
                """, (wine_name,))

            count = cursor.fetchone()[0]
            return count > 0


def get_wine_count() -> int:
    """Get total number of wines in database."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM wines")
            count = cursor.fetchone()[0]
            return count


# ===== Supabase Storage Functions =====

def get_supabase_client() -> Optional[Client]:
    """Get Supabase client for storage operations."""
    try:
        # Get credentials from secrets or environment
        try:
            supabase_url = st.secrets.get("SUPABASE_URL")
            supabase_key = st.secrets.get("SUPABASE_KEY")
        except (FileNotFoundError, KeyError, AttributeError):
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            # Extract from DATABASE_URL if available
            database_url = get_database_url()
            if database_url and "supabase.co" in database_url:
                # Extract project ref from URL: postgres.PROJECT_REF or db.PROJECT_REF
                match = re.search(r'@(?:db|aws-\d+-[^.]+\.pooler)\.([a-z0-9]+)\.supabase\.co', database_url)
                if match:
                    project_ref = match.group(1)
                    supabase_url = f"https://{project_ref}.supabase.co"
                    # For storage, we need the anon key which should be in secrets
                    return None  # Require explicit SUPABASE_KEY

        if supabase_url and supabase_key:
            return create_client(supabase_url, supabase_key)
        return None
    except Exception:
        return None


def sanitize_filename(name: str) -> str:
    """Sanitize wine name for use as filename."""
    # Remove special characters, replace spaces with underscores
    sanitized = re.sub(r'[^\w\s-]', '', name.lower())
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.strip('_')


def upload_wine_image(wine_name: str, image_data: bytes, content_type: str = "image/png") -> Optional[str]:
    """
    Upload wine image to Supabase Storage.

    Args:
        wine_name: Name of the wine
        image_data: Image bytes
        content_type: MIME type (default: image/png)

    Returns:
        Public URL of uploaded image, or None if upload failed
    """
    try:
        client = get_supabase_client()
        if not client:
            return None

        # Generate filename
        safe_name = sanitize_filename(wine_name)
        extension = content_type.split('/')[-1]
        filename = f"wines/{safe_name}.{extension}"

        # Upload to 'wine-images' bucket
        try:
            response = client.storage.from_('wine-images').upload(
                filename,
                image_data,
                {
                    'content-type': content_type,
                    'upsert': 'true'  # Overwrite if exists
                }
            )
        except Exception as e:
            # If bucket doesn't exist, return None
            if "not found" in str(e).lower():
                return None
            raise

        # Get public URL
        public_url = client.storage.from_('wine-images').get_public_url(filename)
        return public_url

    except Exception as e:
        # Fail gracefully - storage is optional
        return None


def get_wine_image_url(wine_name: str) -> Optional[str]:
    """
    Get public URL for wine image from Supabase Storage.

    Args:
        wine_name: Name of the wine

    Returns:
        Public URL of the image, or None if not found
    """
    try:
        client = get_supabase_client()
        if not client:
            return None

        safe_name = sanitize_filename(wine_name)

        # Try common extensions
        for ext in ['png', 'jpg', 'jpeg', 'webp']:
            filename = f"wines/{safe_name}.{ext}"
            try:
                # Check if file exists
                files = client.storage.from_('wine-images').list('wines', {
                    'search': f'{safe_name}.{ext}'
                })
                if files:
                    public_url = client.storage.from_('wine-images').get_public_url(filename)
                    return public_url
            except Exception:
                continue

        return None
    except Exception:
        return None


def delete_wine_image(wine_name: str) -> bool:
    """
    Delete wine image from Supabase Storage.

    Args:
        wine_name: Name of the wine

    Returns:
        True if deleted, False otherwise
    """
    try:
        client = get_supabase_client()
        if not client:
            return False

        safe_name = sanitize_filename(wine_name)

        # Try to delete all possible extensions
        deleted = False
        for ext in ['png', 'jpg', 'jpeg', 'webp']:
            filename = f"wines/{safe_name}.{ext}"
            try:
                client.storage.from_('wine-images').remove([filename])
                deleted = True
            except Exception:
                continue

        return deleted
    except Exception:
        return False
