"""Translation history routes."""

import json
from fastapi import APIRouter, Depends, HTTPException
import sqlite3

from app.models import get_translations, translation_to_dict
from app.routes.auth import get_current_user

router = APIRouter()


@router.get("")
def list_history(current_user: sqlite3.Row = Depends(get_current_user)):
    """Return recent translation history for the authenticated user."""
    user_id = current_user["id"]
    rows = get_translations(user_id)
    results = []
    for row in rows:
        item = translation_to_dict(row)
        if item.get("sections_json"):
            try:
                item["sections"] = json.loads(item["sections_json"])
            except (json.JSONDecodeError, TypeError):
                item["sections"] = {}
        if item.get("ir_json"):
            try:
                item["ir"] = json.loads(item["ir_json"])
            except (json.JSONDecodeError, TypeError):
                item["ir"] = {}
        results.append(item)
    return results