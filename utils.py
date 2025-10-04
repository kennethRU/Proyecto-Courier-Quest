# utils.py
import json
import os
from datetime import datetime, timezone
from typing import Any

def ensure_dir(path: str) -> None:
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Any:
    """Lee un archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data: Any) -> None:
    """Escribe datos a un archivo JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clamp(val: float, lo: float, hi: float) -> float:
    """Limita un valor entre un mínimo y un máximo."""
    return max(lo, min(hi, val))

def iso_to_datetime(iso_str: str) -> datetime:
    """Convierte una cadena ISO a objeto datetime, manejando el formato 'Z'."""
    # Reemplaza 'Z' por '+00:00' para compatibilidad con fromisoformat
    return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))

def iso_to_datetime(iso_str: str) -> datetime:
    """
    Convierte una cadena ISO a objeto datetime, forzando la conciencia de zona 
    horaria (aware) en UTC.
    """
    # 1. Reemplaza 'Z' por '+00:00' para que fromisoformat funcione
    dt_naive = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    
    # 2. Hace que la fecha sea aware, asignándole la zona horaria UTC
    if dt_naive.tzinfo is None:
        return dt_naive.replace(tzinfo=timezone.utc)
    
    return dt_naive