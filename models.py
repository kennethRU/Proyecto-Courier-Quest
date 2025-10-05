# models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

from inventory import DoublyLinkedList # Necesaria para GameState

@dataclass
class CityMap:
    version: str
    width: int
    height: int
    tiles: List[List[str]]
    legend: Dict[str, Dict[str, Any]]
    goal: int

    def is_blocked(self, x: int, y: int) -> bool:
        if not (0 <= y < self.height and 0 <= x < self.width):
            return True 
        code = self.tiles[y][x]
        return self.legend.get(code, {}).get("blocked", False)

    def surface_weight(self, x: int, y: int) -> float:
        if not (0 <= y < self.height and 0 <= x < self.width):
            return 1.0
        code = self.tiles[y][x]
        return self.legend.get(code, {}).get("surface_weight", 1.0)


@dataclass
class Job:
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
        return 900.0


@dataclass
class Player:
    x: int
    y: int
    stamina: float
    reputation: int
    money: int = 0
    weight_carried: float = 0.0
    deliveries_in_row: int = 0


@dataclass
class WeatherState:
    current: str
    target: str
    intensity: float
    burst_time_left: float
    transition_time_left: float
    bursts: Optional[List[Dict[str, Any]]] = field(default_factory=list) 
    burst_index: int = 0  


@dataclass
class GameState:
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