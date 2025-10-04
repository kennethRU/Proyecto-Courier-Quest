# weather.py
import random
from typing import Tuple, Dict
from config import WEATHER_STATES, WEATHER_TRANSITION, WEATHER_MULT, WEATHER_BURST_MIN, WEATHER_BURST_MAX

def pick_next_condition(current: str) -> str:
    """Selecciona la siguiente condición climática basada en la matriz de Markov."""
    idx = WEATHER_STATES.index(current)
    probs = WEATHER_TRANSITION[idx]
    r = random.random()
    acc = 0.0
    for i, prob in enumerate(probs):
        acc += prob
        if r <= acc:
            return WEATHER_STATES[i]
    return current # Fallback si algo sale mal

def pick_burst_duration() -> float:
    """Retorna una duración aleatoria para el burst de clima."""
    return random.uniform(WEATHER_BURST_MIN, WEATHER_BURST_MAX)

def weather_multiplier(condition: str) -> float:
    """Retorna el multiplicador de velocidad para una condición climática."""
    return WEATHER_MULT.get(condition, 1.0)

def interpolate(start: float, end: float, t: float) -> float:
    """Interpola linealmente entre start y end usando t (0.0 a 1.0)."""
    return start + (end - start) * t