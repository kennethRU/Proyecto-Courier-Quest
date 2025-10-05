# persistence.py
import pickle
import os
from typing import Optional, List, Dict, Any
from config import SAVE_PATH, SCORES_PATH
from models import GameState
from utils import read_json, write_json

def save_binary(state: GameState) -> None:
    os.makedirs(os.path.dirname(SAVE_PATH) or '.', exist_ok=True)
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(state, f)

def load_binary() -> Optional[GameState]:
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error cargando partida: {e}")
            return None
    return None

def load_scores() -> List[Dict[str, Any]]:
    if os.path.exists(SCORES_PATH):
        try:
            return read_json(SCORES_PATH)
        except Exception:
            return []
    return []

def save_scores(scores: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(SCORES_PATH) or '.', exist_ok=True)
    write_json(SCORES_PATH, scores)