# utils.py
import json
import os
from datetime import datetime, timezone
from typing import Any

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def iso_to_datetime(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))

def iso_to_datetime(iso_str: str) -> datetime:

    dt_naive = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    
    if dt_naive.tzinfo is None:
        return dt_naive.replace(tzinfo=timezone.utc)
    
    return dt_naive