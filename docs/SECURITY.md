# Security Guidelines for Decant

> Security documentation created with Claude AI assistance

## ⚠️ API Key Management

### DO NOT:
- ❌ Share API keys in chat, emails, or messages
- ❌ Commit API keys to git (even in private repos)
- ❌ Include API keys in screenshots
- ❌ Hardcode API keys in source code

### DO:
- ✅ Store API keys in `.env` file (already in `.gitignore`)
- ✅ Revoke compromised keys immediately
- ✅ Use environment variables for sensitive data
- ✅ Rotate API keys periodically

## Setting Up Your API Key Securely

### 1. Create/Revoke Keys
- Go to: https://platform.openai.com/api-keys
- Delete any compromised keys
- Create a new key

### 2. Store in .env File
```bash
# Edit the .env file
nano .env

# Add your key (no spaces, no quotes)
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### 3. Verify .env is in .gitignore
```bash
# Check that .env is ignored
grep "^\.env$" .gitignore
# Should output: .env
```

### 4. Never Commit .env
```bash
# Before committing, always check:
git status

# .env should appear under "Untracked files" or not at all
# NEVER under "Changes to be committed"
```

## Using the .env File

The `scripts/extract_features.py` script automatically loads from `.env`:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

api_key = os.getenv("OPENAI_API_KEY")  # Reads from .env
```

## What to Do If You Exposed a Key

1. **Immediately revoke** at https://platform.openai.com/api-keys
2. **Create a new key**
3. **Update your `.env` file** with the new key
4. **Check git history** - if committed, consider it permanently compromised
5. **Rotate any other keys** that might have been exposed

## Best Practices

- Store `.env` file ONLY on your local machine
- Use different API keys for dev/staging/production
- Set spending limits on API keys
- Monitor usage regularly
- Use read-only keys when possible
- Enable MFA on your OpenAI account

## Best Practices for Data Projects

Demonstrating security awareness is critical:
- Always use environment variables for secrets
- Implement proper secret management
- Document security practices
- Never compromise on security for convenience

---

**Built with Claude AI** - This security documentation was created with assistance from [Claude Code](https://claude.ai/code)
