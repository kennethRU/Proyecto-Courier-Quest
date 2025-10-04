# models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

from inventory import DoublyLinkedList # Necesaria para GameState

@dataclass
class CityMap:
    """Estructura de datos para el mapa de la ciudad."""
    version: str
    width: int
    height: int
    tiles: List[List[str]]
    legend: Dict[str, Dict[str, Any]]
    goal: int

    def is_blocked(self, x: int, y: int) -> bool:
        """Verifica si una celda está bloqueada."""
        if not (0 <= y < self.height and 0 <= x < self.width):
            return True # Fuera de límites
        code = self.tiles[y][x]
        return self.legend.get(code, {}).get("blocked", False)

    def surface_weight(self, x: int, y: int) -> float:
        """Retorna el peso de la superficie (costo de movimiento)."""
        if not (0 <= y < self.height and 0 <= x < self.width):
            return 1.0
        code = self.tiles[y][x]
        return self.legend.get(code, {}).get("surface_weight", 1.0)


@dataclass
class Job:
    """Estructura de datos para un trabajo/pedido."""
    id: str
    pickup: Tuple[int, int]
    dropoff: Tuple[int, int]
    payout: int
    deadline: datetime
    weight: int
    priority: int
    release_time: int = 0
    accepted: bool = False
    delivered: bool = False
    canceled: bool = False

    @property
    def deadline_seconds(self) -> float:
        # Se asume que este valor es calculado o ajustado externamente
        # o es solo un placeholder para el código de game.py que lo usa.
        # Por simplicidad para este archivo, se deja un placeholder.
        return 900.0


@dataclass
class Player:
    """Estructura de datos para el jugador (el mensajero)."""
    x: int
    y: int
    stamina: float
    reputation: int
    money: int = 0
    weight_carried: float = 0.0
    deliveries_in_row: int = 0


@dataclass
class WeatherState:
    """Estado actual y futuro del clima, con soporte para secuencia de bursts de la API."""
    current: str
    target: str
    intensity: float
    burst_time_left: float
    transition_time_left: float
    bursts: Optional[List[Dict[str, Any]]] = field(default_factory=list)  # lista de bursts de la API
    burst_index: int = 0  # índice del burst actual



@dataclass
class GameState:
    """Estado global del juego."""
    city: CityMap
    player: Player
    jobs_all: List[Job]
    jobs_active: List[Job]
    inventory: DoublyLinkedList
    inv_cursor: int
    weather: WeatherState
    elapsed: float
    game_over: bool
    victory: bool
    message: str
    current_path: List[Tuple[int, int]] = field(default_factory=list)