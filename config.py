"""
Application configuration: API endpoints, cache TTLs, city presets, and color palettes.

Keep this module dependency-free so any other module can import it without side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_BASE_URL: Final[str] = "https://open-bus-stride-api.hasadna.org.il"

# Per-request safety net. The Stride API can be slow on wide bulk fetches.
REQUEST_TIMEOUT_SEC: Final[int] = 90

# Auto-paginated list calls: max number of pages we'll ever pull before bailing,
# regardless of what the server claims. Hard ceiling = HARD_PAGE_LIMIT * BATCH_SIZE rows.
#
# The Stride API rejects `limit > 15000` with HTTP 500 ("due to abuse..."), so
# we page at 10000 — enough to fetch typical reference datasets in 1–4 hops.
BATCH_SIZE: Final[int] = 10_000
HARD_PAGE_LIMIT: Final[int] = 50

# ---------------------------------------------------------------------------
# Caching strategy
# ---------------------------------------------------------------------------
# Static reference data (routes/agencies/stops) refreshes once a day at most.
# SIRI vehicle positions move every few seconds — short TTL.
# Route shapes are stable for the operating day.

CACHE_TTL_REFERENCE_SEC: Final[int] = 60 * 60 * 12  # 12h — routes/agencies/stops
CACHE_TTL_SHAPE_SEC: Final[int] = 60 * 60          # 1h  — route shapes/timetables
CACHE_TTL_LIVE_SEC: Final[int] = 20                 # 20s — SIRI vehicle locations

DISK_CACHE_DIR: Final[str] = "data"

# ---------------------------------------------------------------------------
# Geography (Israel)
# ---------------------------------------------------------------------------

ISRAEL_TIMEZONE: Final[str] = "Asia/Jerusalem"

# Validation envelope; SIRI points outside this box are dropped as junk.
ISRAEL_LAT_RANGE: Final[tuple[float, float]] = (29.0, 34.0)
ISRAEL_LON_RANGE: Final[tuple[float, float]] = (34.0, 36.0)


@dataclass(frozen=True)
class CityPreset:
    name: str
    lat: float
    lon: float
    zoom: int = 13


CITY_PRESETS: Final[dict[str, CityPreset]] = {
    "Tel Aviv":   CityPreset("Tel Aviv",   32.0853, 34.7818, zoom=13),
    "Jerusalem":  CityPreset("Jerusalem",  31.7683, 35.2137, zoom=13),
    "Haifa":      CityPreset("Haifa",      32.7940, 34.9896, zoom=13),
    "Be'er Sheva": CityPreset("Be'er Sheva", 31.2529, 34.7914, zoom=13),
}

DEFAULT_CITY: Final[str] = "Tel Aviv"

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Twemoji directional arrows; deck.gl IconLayer URLs.
TWEMOJI_BASE: Final[str] = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72"
BEARING_ICONS: Final[dict[str, str]] = {
    "N":  f"{TWEMOJI_BASE}/2b06.png",
    "NE": f"{TWEMOJI_BASE}/2197.png",
    "E":  f"{TWEMOJI_BASE}/27a1.png",
    "SE": f"{TWEMOJI_BASE}/2198.png",
    "S":  f"{TWEMOJI_BASE}/2b07.png",
    "SW": f"{TWEMOJI_BASE}/2199.png",
    "W":  f"{TWEMOJI_BASE}/2b05.png",
    "NW": f"{TWEMOJI_BASE}/2196.png",
}

# Categorical color palette for distinguishing routes on the map.
ROUTE_PALETTE: Final[list[tuple[int, int, int]]] = [
    (0, 255, 150),    # bright green
    (255, 0, 100),    # pink
    (0, 200, 255),    # cyan
    (255, 200, 0),    # gold
    (200, 100, 255),  # purple
    (255, 100, 50),   # orange
    (50, 50, 255),    # blue
    (255, 255, 255),  # white
]


# ---------------------------------------------------------------------------
# Application defaults
# ---------------------------------------------------------------------------

@dataclass
class VizSettings:
    """Mutable viz settings driven by the sidebar."""
    dot_radius: int = 50
    path_width: int = 30
    arrow_size: int = 45
    pin_radius: int = 150
    max_paths: int = 50


@dataclass
class LiveSettings:
    """User-facing settings for the live-traffic mode."""
    city: str = DEFAULT_CITY
    lat: float = field(default=CITY_PRESETS[DEFAULT_CITY].lat)
    lon: float = field(default=CITY_PRESETS[DEFAULT_CITY].lon)
    radius_km: float = 2.0
    lookback_min: int = 10
    auto_refresh: bool = False
