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

try:
    from psycopg_pool import ConnectionPool
    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False
    ConnectionPool = None

# Global connection pool
_connection_pool: Optional[ConnectionPool] = None


def get_cellar_id() -> str:
    """Get shared cellar ID from Streamlit secrets or environment."""
    try:
        cellar_id = st.secrets["CELLAR_ID"]
    except (FileNotFoundError, KeyError):
        cellar_id = os.getenv("CELLAR_ID")

    if not cellar_id:
        raise ValueError("CELLAR_ID not found in secrets or environment")
    return cellar_id


def get_database_url() -> Optional[str]:
    """Get database URL from Streamlit secrets or environment."""
    try:
        return st.secrets["DATABASE_URL"]
    except (FileNotFoundError, KeyError):
        return os.getenv("DATABASE_URL")


def get_connection_pool():
    """
    Get or create connection pool.

    Uses connection pooling to prevent resource exhaustion.
    Pool configuration: min=1, max=5 connections.
    Falls back to direct connection if pool not available.
    """
    global _connection_pool

    if not POOL_AVAILABLE:
        # Fallback: return None, will use direct connections
        return None

    if _connection_pool is None:
        database_url = get_database_url()
        if not database_url:
            raise ValueError("DATABASE_URL not found in secrets or environment")

        # Create and open pool with 1-5 connections
        _connection_pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=5
        )
        _connection_pool.open()  # Open pool immediately

    return _connection_pool


def get_connection():
    """Get database connection (from pool if available, otherwise direct)."""
    pool = get_connection_pool()

    if pool is not None:
        # Use pool
        return pool.getconn()
    else:
        # Direct connection fallback
        database_url = get_database_url()
        if not database_url:
            raise ValueError("DATABASE_URL not found in secrets or environment")
        try:
            return psycopg.connect(database_url)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")


def return_connection(conn):
    """Return connection to pool (or close if no pool)."""
    pool = get_connection_pool()
    if pool is not None:
        pool.putconn(conn)
    else:
        conn.close()


def init_database():
    """Initialize database schema (create wines table if not exists)."""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Create wines table with full schema + user_id
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wines (
                    id SERIAL PRIMARY KEY,
                    cellar_id TEXT NOT NULL DEFAULT 'default_cellar',
                    user_id TEXT NOT NULL DEFAULT 'admin',
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Add user_id column if table already exists (migration)
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'wines' AND column_name = 'user_id'
                    ) THEN
                        ALTER TABLE wines ADD COLUMN user_id TEXT NOT NULL DEFAULT 'admin';
                    END IF;
                END $$;
            """)

            # Add cellar_id column if table already exists (migration)
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'wines' AND column_name = 'cellar_id'
                    ) THEN
                        ALTER TABLE wines ADD COLUMN cellar_id TEXT NOT NULL DEFAULT 'default_cellar';
                    END IF;
                END $$;
            """)

            # Add updated_at column if table already exists (migration)
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'wines' AND column_name = 'updated_at'
                    ) THEN
                        ALTER TABLE wines ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                    END IF;
                END $$;
            """)

            # Create UNIQUE constraint to prevent duplicates
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'unique_user_wine_vintage'
                    ) THEN
                        ALTER TABLE wines
                        ADD CONSTRAINT unique_user_wine_vintage
                        UNIQUE(user_id, wine_name, vintage);
                    END IF;
                END $$;
            """)

            # Create index on user_id for filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON wines(user_id);
            """)

            # Create index on cellar_id for filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cellar_id ON wines(cellar_id);
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
    finally:
        return_connection(conn)


def add_wine(wine_data: Dict[str, Any]) -> bool:
    """Add a wine to the database with user attribution."""
    row_data = dict(wine_data)
    row_data["cellar_id"] = get_cellar_id()

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO wines (
                    cellar_id, user_id, wine_name, producer, vintage, notes, score, liked, price,
                    country, region, wine_color, is_sparkling, is_natural, sweetness,
                    acidity, minerality, fruitiness, tannin, body
                ) VALUES (
                    %(cellar_id)s, %(user_id)s, %(wine_name)s, %(producer)s, %(vintage)s, %(notes)s, %(score)s, %(liked)s, %(price)s,
                    %(country)s, %(region)s, %(wine_color)s, %(is_sparkling)s, %(is_natural)s, %(sweetness)s,
                    %(acidity)s, %(minerality)s, %(fruitiness)s, %(tannin)s, %(body)s
                )
            """, row_data)

            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        raise
    finally:
        return_connection(conn)


def get_user_group(user_id: str) -> list:
    """
    Get list of user IDs in the same group (for shared collections).

    Groups are defined in SHARED_GROUPS setting. If user not in any group,
    returns just the user_id.

    Args:
        user_id: User ID to check

    Returns:
        List of user IDs that share the same wine collection
    """
    # Define shared groups here (can be moved to secrets later)
    # Format: list of groups, where each group is a list of usernames
    try:
        import streamlit as st
        # Try to get from secrets first
        shared_groups = st.secrets.get("SHARED_GROUPS", [])
    except (FileNotFoundError, KeyError, AttributeError):
        # Default: tomasz and karolina share collection
        shared_groups = [["tomasz", "karolina"]]

    # Find which group the user belongs to
    for group in shared_groups:
        if user_id in group:
            return group

    # Not in any group, return just the user
    return [user_id]


def get_all_wines(user_id: str) -> pd.DataFrame:
    """
    Retrieve wines as a pandas DataFrame filtered by user group.

    If user belongs to a shared group (e.g., couple), returns wines
    from all users in that group. Otherwise returns only user's wines.

    Args:
        user_id: User ID to filter wines

    Returns:
        DataFrame with wine data for the user's group
    """
    if not user_id:
        raise ValueError("user_id is required for data isolation")

    # Get all users in the same group
    group_users = get_user_group(user_id)
    cellar_id = get_cellar_id()

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Build WHERE clause for multiple users
            placeholders = ','.join(['%s'] * len(group_users))
            cursor.execute(f"""
                SELECT
                    user_id, wine_name, producer, vintage, notes, score, liked, price,
                    country, region, wine_color, is_sparkling, is_natural, sweetness,
                    acidity, minerality, fruitiness, tannin, body
                FROM wines
                WHERE cellar_id = %s
                  AND user_id IN ({placeholders})
                ORDER BY created_at DESC
            """, tuple([cellar_id] + group_users))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                # Return empty DataFrame with correct schema
                return pd.DataFrame(columns=[
                    'user_id', 'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
                    'country', 'region', 'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
                    'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
                ])

            df = pd.DataFrame(rows, columns=columns)

            # Convert types to match CSV format
            df['liked'] = df['liked'].astype(bool)
            df['price'] = df['price'].astype(float)
            df['score'] = df['score'].astype(float)

            return df
    finally:
        return_connection(conn)


def delete_wine(wine_name: str, user_id: str, vintage: Optional[float] = None) -> bool:
    """Delete a wine from the database for a specific user."""
    cellar_id = get_cellar_id()
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if vintage:
                cursor.execute("""
                    DELETE FROM wines
                    WHERE cellar_id = %s AND user_id = %s AND wine_name = %s AND vintage = %s
                """, (cellar_id, user_id, wine_name, vintage))
            else:
                cursor.execute("""
                    DELETE FROM wines
                    WHERE cellar_id = %s AND user_id = %s AND wine_name = %s
                """, (cellar_id, user_id, wine_name))

            conn.commit()
            return True
    finally:
        return_connection(conn)


def wine_exists(wine_name: str, user_id: str, vintage: Optional[float] = None) -> bool:
    """Check if a wine already exists in the database for a specific user."""
    cellar_id = get_cellar_id()
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if vintage:
                cursor.execute("""
                    SELECT COUNT(*) FROM wines
                    WHERE cellar_id = %s AND user_id = %s AND wine_name = %s AND vintage = %s
                """, (cellar_id, user_id, wine_name, vintage))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM wines
                    WHERE cellar_id = %s AND user_id = %s AND wine_name = %s
                """, (cellar_id, user_id, wine_name))

            count = cursor.fetchone()[0]
            return count > 0
    finally:
        return_connection(conn)


def get_wine_count(user_id: Optional[str] = None) -> int:
    """
    Get total number of wines in database.

    Args:
        user_id: Optional user ID to filter wines. If None, counts all wines.

    Returns:
        Count of wines
    """
    cellar_id = get_cellar_id()
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if user_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM wines WHERE cellar_id = %s AND user_id = %s",
                    (cellar_id, user_id),
                )
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM wines WHERE cellar_id = %s",
                    (cellar_id,),
                )
            count = cursor.fetchone()[0]
            return count
    finally:
        return_connection(conn)


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
    import unicodedata

    # Normalize Unicode characters to ASCII equivalents
    # NFD = decompose accented chars (ñ -> n + ~), then filter out combining marks
    normalized = unicodedata.normalize('NFKD', name)
    ascii_str = normalized.encode('ascii', 'ignore').decode('ascii')

    # Remove special characters, replace spaces with underscores
    sanitized = re.sub(r'[^\w\s-]', '', ascii_str.lower())
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
