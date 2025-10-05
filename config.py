
from datetime import datetime, timezone
# Ventana
SCREEN_WIDTH = 1300
SCREEN_HEIGHT = 1000
TILE_SIZE = 32

# Juego
TARGET_FPS = 60
GAME_DURATION_SECONDS = 15 * 60  # 15 minutos
UNDO_HISTORY = 20  # cantidad máxima de pasos a deshacer
SAVE_PATH = "saves/slot1.sav"
SCORES_PATH = "data/puntajes.json"

# API y caché
API_BASE = "https://tigerds-api.kindflower-ccaf48b6.eastus.azurecontainerapps.io"
CACHE_DIR = "api_cache"
DATA_DIR = "data"
CITY_ENDPOINT = "/city/map"
JOBS_ENDPOINT = "/city/jobs"
WEATHER_ENDPOINT = "/city/weather?mode=seed"

# Inventario
MAX_WEIGHT = 8  # peso máximo que puede cargar el jugador

# Velocidad y estados del jugador
V0_CELLS_PER_SEC = 3.0
STAMINA_MAX = 100.0
STAMINA_RECOVERY_IDLE = 5.0  # por segundo
STAMINA_RECOVERY_REST_POINT = 10.0  # por segundo (si implementas puntos de descanso)
STAMINA_BASE_COST_PER_CELL = 0.5

# Pesos y clima (consumo extra por celda)
STAMINA_WEIGHT_THRESHOLD = 3.0
STAMINA_WEIGHT_EXTRA_PER_UNIT = 0.2
STAMINA_WEATHER_EXTRA = {
    "rain": 0.1,
    "wind": 0.1,
    "storm": 0.3,
    "heat": 0.2
}

# Reputación
REPUTATION_START = 70
REPUTATION_FAIL = 20
REPUTATION_BONUS_THRESHOLD = 90
REPUTATION_PAYOUT_BONUS = 0.05

# Bonificaciones/penalizaciones reputación
REP_DELIVERY_ON_TIME = +3
REP_DELIVERY_EARLY = +5  # >=20% antes del deadline
REP_LATE_30 = -2
REP_LATE_120 = -5
REP_LATE_OVER = -10
REP_CANCEL_ACCEPTED = -4
REP_EXPIRE_LOST = -6
REP_STREAK_3 = +2

# Matriz de Markov del clima (ejemplo ampliado)
WEATHER_STATES = ["clear", "clouds", "rain_light", "rain", "storm", "fog", "wind", "heat", "cold"]
# Matriz cuadrada con probabilidades por fila (suman ~1.0)
WEATHER_TRANSITION = [
    # clear
    [0.55, 0.20, 0.07, 0.05, 0.02, 0.03, 0.04, 0.02, 0.02],
    # clouds
    [0.30, 0.40, 0.08, 0.08, 0.03, 0.03, 0.04, 0.02, 0.02],
    # rain_light
    [0.25, 0.25, 0.20, 0.15, 0.04, 0.02, 0.05, 0.02, 0.02],
    # rain
    [0.15, 0.25, 0.20, 0.25, 0.08, 0.02, 0.03, 0.01, 0.01],
    # storm
    [0.10, 0.20, 0.15, 0.25, 0.25, 0.01, 0.02, 0.01, 0.01],
    # fog
    [0.20, 0.35, 0.05, 0.05, 0.02, 0.25, 0.05, 0.02, 0.01],
    # wind
    [0.20, 0.25, 0.05, 0.05, 0.02, 0.03, 0.35, 0.03, 0.02],
    # heat
    [0.25, 0.20, 0.05, 0.05, 0.01, 0.02, 0.02, 0.35, 0.05],
    # cold
    [0.25, 0.25, 0.05, 0.05, 0.01, 0.03, 0.03, 0.05, 0.28],
]
# Multiplicadores base de velocidad por clima
WEATHER_MULT = {
    "clear": 1.00,
    "clouds": 0.98,
    "rain_light": 0.90,
    "rain": 0.85,
    "storm": 0.75,
    "fog": 0.88,
    "wind": 0.92,
    "heat": 0.90,
    "cold": 0.92,
}
WEATHER_BURST_MIN = 45
WEATHER_BURST_MAX = 60
WEATHER_TRANSITION_TIME = 3.0  # segundos para interpolación suave


# ...
# Si su API devuelve 'start_time': '2025-09-01T12:00:00Z'
START_TIME_STR = "2025-09-01T12:00:00Z" # O la que use la API
GAME_CLOCK_START = datetime.fromisoformat(START_TIME_STR.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
# ...
