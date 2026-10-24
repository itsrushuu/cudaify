"""List available Nemotron models from Nebius Token Factory."""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEBIUS_API_KEY")
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")

if not API_KEY:
    print("ERROR: NEBIUS_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

# Try common model listing endpoints
endpoints = [
    f"{BASE_URL}/models",
    f"{BASE_URL}/models/",
]

for url in endpoints:
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
        if resp.ok:
            data = resp.json()
            models = data.get("data", data if isinstance(data, list) else [])
            print(f"Models from {url}:")
            for m in models:
                mid = m.get("id", m) if isinstance(m, dict) else m
                print(f"  - {mid}")
            sys.exit(0)
        else:
            print(f"{url} returned {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        print(f"{url} error: {exc}")

print("\nCould not list models. Common Nemotron model IDs on Nebius Token Factory:")
print("  - nvidia/Nemotron-3.5-Super-VL")
print("  - nvidia/Nemotron-3.5-Nano")
print("  - nvidia/Nemotron-3.5-Ultra")
print("\nSet NEBIUS_MODEL in .env to your preferred model ID.")