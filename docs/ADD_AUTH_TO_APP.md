# Adding Authentication to app.py

To enable password protection, add these lines to the TOP of your `app.py` file:

## Step 1: Add import at the top

```python
# Add this with other imports at the top of app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from decant.auth import setup_authentication
```

## Step 2: Add authentication check after imports

```python
# Add this RIGHT AFTER all imports, BEFORE any Streamlit UI code
# This must be the FIRST Streamlit command that runs
username = setup_authentication()

# Now the rest of your app code...
st.title("🍷 Decant - Wine Recommendation System")
# ... rest of app ...
```

## Complete Example

```python
#!/usr/bin/env python3
"""
Decant - AI-powered wine recommendation app
"""

import streamlit as st
import pandas as pd
# ... other imports ...

# Add auth imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from decant.auth import setup_authentication

# AUTHENTICATION - Must be first!
username = setup_authentication()

# Now your app continues normally...
st.set_page_config(page_title="Decant", page_icon="🍷")
st.title("🍷 Decant - Wine Recommendation System")

# ... rest of your app code ...
```

## Default Credentials

- **Username**: `admin`
- **Password**: `wine123`

**IMPORTANT**: Change the password after first login!

## Changing the Password

1. Generate a new hash:
   ```bash
   python -m decant.auth YOUR_NEW_PASSWORD
   ```

2. Copy the hash and update `.streamlit/secrets.toml`:
   ```toml
   [passwords]
   admin = "PASTE_NEW_HASH_HERE"
   ```

3. Restart the app

## Adding More Users

In `.streamlit/secrets.toml`, add more usernames:

```toml
[passwords]
admin = "$2b$12$..."  # First user
partner = "$2b$12$..."  # Second user  (generate hash first)
```

Then both can log in with their own credentials!

---

**Note**: This provides password protection but does NOT separate data by user. All users see the same wines and recommendations.
