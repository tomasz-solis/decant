#!/usr/bin/env python3
"""
Migrate wine images from local storage to Supabase Storage.

This script uploads all existing wine photos from data/wine_images/
to Supabase Storage bucket 'wine-images'.

Usage:
    python scripts/migrate_images_to_supabase.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decant.database import get_supabase_client, upload_wine_image, sanitize_filename


def migrate_images_to_supabase():
    """Migrate local wine images to Supabase Storage."""
    images_dir = Path("data/wine_images")

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    # Check Supabase connection
    client = get_supabase_client()
    if not client:
        print("❌ Supabase client not configured")
        print("\n💡 Add to .streamlit/secrets.toml:")
        print('SUPABASE_URL = "https://rageghyliafgxublfljb.supabase.co"')
        print('SUPABASE_KEY = "your-anon-key-here"')
        print("\nGet your anon key from:")
        print("https://supabase.com/dashboard/project/rageghyliafgxublfljb/settings/api")
        return

    # Check if bucket exists
    try:
        buckets = client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        if 'wine-images' not in bucket_names:
            print("📦 Creating 'wine-images' bucket...")
            client.storage.create_bucket(
                'wine-images',
                {
                    'public': True,
                    'file_size_limit': 5242880  # 5MB
                }
            )
            print("✅ Bucket created")
    except Exception as e:
        print(f"⚠️  Could not check/create bucket: {e}")
        print("   Continuing anyway...")

    print(f"\n🚀 Starting image migration from {images_dir}...\n")

    # Get all image files
    image_files = list(images_dir.glob("*.png")) + \
                  list(images_dir.glob("*.jpg")) + \
                  list(images_dir.glob("*.jpeg")) + \
                  list(images_dir.glob("*.webp"))

    if not image_files:
        print("❌ No images found to migrate")
        return

    success_count = 0
    error_count = 0

    for image_path in image_files:
        # Extract wine name from filename
        wine_name = image_path.stem.replace('_', ' ').title()

        try:
            # Read image data
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Determine content type
            ext = image_path.suffix.lower()
            content_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp'
            }.get(ext, 'image/png')

            # Upload to Supabase
            public_url = upload_wine_image(wine_name, image_data, content_type)

            if public_url:
                print(f"   ✓ {image_path.name} → {wine_name}")
                success_count += 1
            else:
                print(f"   ✗ {image_path.name} - Upload failed")
                error_count += 1

        except Exception as e:
            print(f"   ✗ {image_path.name} - Error: {e}")
            error_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"📈 Migration Summary:")
    print(f"   ✅ Successfully uploaded: {success_count} images")
    if error_count > 0:
        print(f"   ❌ Errors: {error_count} images")
    print(f"{'='*60}")

    if success_count > 0:
        print("\n🎉 Migration complete!")
        print("\n💡 Next steps:")
        print("   1. Images are now in Supabase Storage")
        print("   2. The app will automatically use Supabase for new images")
        print("   3. You can safely delete local images (optional)")
    else:
        print("\n⚠️  No images were migrated successfully")


if __name__ == "__main__":
    migrate_images_to_supabase()
