"""
PostgreSQL database module for Decant wine storage.

Provides persistent storage on Supabase for wine tasting data,
replacing ephemeral CSV storage on Streamlit Cloud.
"""

import os
import psycopg
import pandas as pd
from typing import Optional, Dict, Any
import streamlit as st


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
