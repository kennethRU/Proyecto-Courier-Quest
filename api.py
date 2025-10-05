# api.py
import os
import time
from typing import Any, Dict, List

try:
    import requests  
except Exception:
    requests = None

from config import (
    API_BASE, CACHE_DIR, DATA_DIR,
    CITY_ENDPOINT, JOBS_ENDPOINT, WEATHER_ENDPOINT
)
from utils import ensure_dir, read_json, write_json


class ApiClient:

    def __init__(self) -> None:
        ensure_dir(CACHE_DIR)
        ensure_dir(DATA_DIR)
        self._use_requests = requests is not None
        if self._use_requests:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": "CourierQuest/1.0"})

    def _cache_path(self, key: str) -> str:
        return os.path.join(CACHE_DIR, key)

    def _fetch(self, url: str) -> Any:
        if self._use_requests:
            resp = self.session.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        import json
        from urllib.request import urlopen, Request
        from urllib.error import URLError, HTTPError
        req = Request(url, headers={"User-Agent": "CourierQuest/1.0"})
        try:
            with urlopen(req, timeout=5) as r:
                data = r.read().decode("utf-8")
                return json.loads(data)
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"HTTP error: {e}")

    def _get_with_cache(self, url: str, cache_key: str, local_fallback: str) -> Any:
        cache_path = self._cache_path(cache_key)
        try:
            data = self._fetch(url)
            write_json(cache_path, {"fetched_at": time.time(), "data": data})
            write_json(os.path.join(DATA_DIR, local_fallback), data)
            return data
        except Exception:
            if os.path.exists(cache_path):
                cached = read_json(cache_path)
                return cached["data"]
            local_path = os.path.join(DATA_DIR, local_fallback)
            return read_json(local_path)

    def get_city_map(self) -> Dict[str, Any]:
        return self._get_with_cache(API_BASE + CITY_ENDPOINT, "city_map.json", "ciudad.json")

    def get_jobs(self) -> List[Dict[str, Any]]:
        return self._get_with_cache(API_BASE + JOBS_ENDPOINT, "jobs.json", "pedidos.json")

    def get_weather(self) -> Dict[str, Any]:
        return self._get_with_cache(API_BASE + WEATHER_ENDPOINT, "weather.json", "weather.json")
