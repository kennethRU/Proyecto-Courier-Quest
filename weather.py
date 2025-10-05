# weather.py
import random
from typing import Tuple, Dict
from config import WEATHER_STATES, WEATHER_TRANSITION, WEATHER_MULT, WEATHER_BURST_MIN, WEATHER_BURST_MAX

def pick_next_condition(current: str) -> str:
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
    return random.uniform(WEATHER_BURST_MIN, WEATHER_BURST_MAX)

def weather_multiplier(condition: str) -> float:
    return WEATHER_MULT.get(condition, 1.0)

def interpolate(start: float, end: float, t: float) -> float:
    return start + (end - start) * t