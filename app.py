"""
OceanData — app.py (Groq Edition)
===================================
Single-command server. Run with:   python app.py
Then open:                          http://localhost:5000

Uses Groq API (completely free, no credit card) instead of Gemini.
Model: llama-3.3-70b-versatile — fast, accurate, generous free quota.

Requirements:
    pip install flask flask-cors requests python-dotenv
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ── Load .env from the same folder as this script ────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

NEWS_API_KEY        = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── In-memory cache — Groq is free but we still cache to be polite ────────────
_cache = {
    "summary":        None,
    "threat":         None,
    "source_count":   0,
    "fetched_at":     None,
    "fetched_at_iso": None,
}
CACHE_TTL_SECONDS = 30 * 60   # re-fetch every 30 minutes


# ─────────────────────────────────────────────────────────────────────────────
#  Route: /   →  serves index.html
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: fetch latest CO2 / pollution headlines from NewsAPI
# ─────────────────────────────────────────────────────────────────────────────
def fetch_headlines(n: int = 12) -> list[dict]:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is missing — add it to your .env file.")

    params = {
        "q": (
            "ocean acidification OR ocean pollution "
            "OR CO2 emissions OR marine pollution "
            "OR sea temperature rise OR climate ocean"
        ),
        "sortBy":   "publishedAt",
        "pageSize": n,
        "language": "en",
        "apiKey":   NEWS_API_KEY,
    }

    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

    articles = data.get("articles", [])
    log.info("NewsAPI → %d articles fetched", len(articles))

    return [
        {
            "title":  a.get("title", ""),
            "desc":   a.get("description", "") or "",
            "source": a.get("source", {}).get("name", ""),
        }
        for a in articles
        if a.get("title")
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: send headlines to Groq → marine science analysis
# ─────────────────────────────────────────────────────────────────────────────
def run_groq_analysis(headlines: list[dict]) -> dict:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing — add it to your .env file.")

    headline_block = "\n".join(
        f"[{i+1}] ({h['source']}) {h['title']}. {h['desc']}"
        for i, h in enumerate(headlines)
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 300,
        "temperature": 0.4,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior marine environmental scientist. "
                    "Given recent global news headlines about CO2 emissions, "
                    "ocean pollution, and climate change, write a 3-sentence "
                    "analysis of the likely current impact on ocean health — "
                    "covering acidification risk, sea surface temperature stress, "
                    "and ecosystem or biodiversity threat level. "
                    "Be factual, direct, and scientific — no filler phrases. "
                    "On the very last line output exactly: THREAT_LEVEL: <level> "
                    "where <level> is one of LOW, MODERATE, HIGH, or CRITICAL. "
                    "Nothing else after the threat level line."
                ),
            },
            {
                "role": "user",
                "content": f"Recent headlines:\n{headline_block}",
            },
        ],
    }

    log.info("Sending request to Groq (llama-3.3-70b-versatile)...")

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    log.info("Groq response received (%d chars)", len(raw))

    # Parse out THREAT_LEVEL from the last line
    lines, threat, body_lines = raw.splitlines(), "MODERATE", []
    for line in lines:
        s = line.strip()
        if s.startswith("THREAT_LEVEL:"):
            c = s.replace("THREAT_LEVEL:", "").strip().upper()
            if c in {"LOW", "MODERATE", "HIGH", "CRITICAL"}:
                threat = c
        else:
            body_lines.append(line)

    return {
        "summary": " ".join(body_lines).strip(),
        "threat":  threat,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Route: /ocean-summary  →  JSON
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/ocean-summary")
def ocean_summary():
    global _cache

    # Serve from cache if still fresh
    if _cache["fetched_at"]:
        age = time.time() - _cache["fetched_at"]
        if age < CACHE_TTL_SECONDS:
            log.info("Cache hit — age %.0fs", age)
            return jsonify({
                "summary":      _cache["summary"],
                "threat":       _cache["threat"],
                "source_count": _cache["source_count"],
                "fetched_at":   _cache["fetched_at_iso"],
                "cached":       True,
            })

    log.info("Cache miss — fetching fresh data")

    try:
        headlines = fetch_headlines(n=12)
        analysis  = run_groq_analysis(headlines)
        now_iso   = datetime.now(timezone.utc).isoformat()

        _cache = {
            "summary":        analysis["summary"],
            "threat":         analysis["threat"],
            "source_count":   len(headlines),
            "fetched_at":     time.time(),
            "fetched_at_iso": now_iso,
        }

        return jsonify({
            "summary":      analysis["summary"],
            "threat":       analysis["threat"],
            "source_count": len(headlines),
            "fetched_at":   now_iso,
            "cached":       False,
        })

    except ValueError as e:
        log.error("Config error: %s", e)
        return jsonify({"error": str(e)}), 500

    except requests.RequestException as e:
        log.error("Network error: %s", e)
        return jsonify({"error": f"External API call failed: {e}"}), 502

    except Exception as e:
        log.error("Unexpected error: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  Route: /weather  →  atmospheric pressure from Open-Meteo (free, no key)
#
#  Buoy coordinates — update LAT/LON to your buoy's actual location
#  Default: Pacific Ocean near Vancouver Island
# ─────────────────────────────────────────────────────────────────────────────
BUOY_LAT = float(os.getenv("BUOY_LAT", "48.5"))
BUOY_LON = float(os.getenv("BUOY_LON", "-124.6"))

_weather_cache: dict = {}
WEATHER_TTL = 10 * 60   # refresh every 10 minutes


def _fetch_openweather() -> dict:
    """Current conditions from OpenWeatherMap (requires API key)."""
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "lat":   BUOY_LAT,
            "lon":   BUOY_LON,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=8,
    )
    resp.raise_for_status()
    d = resp.json()
    return {
        "temp_air":     round(d["main"]["temp"], 1),
        "humidity":     d["main"]["humidity"],
        "pressure":     round(d["main"]["pressure"], 1),
        "wind_speed":   round(d["wind"]["speed"], 1),
        "wind_deg":     d["wind"].get("deg", 0),
        "description":  d["weather"][0]["description"].title() if d.get("weather") else "—",
        "visibility_km": round(d.get("visibility", 0) / 1000, 1),
    }


def _fetch_open_meteo() -> dict:
    """Atmospheric + marine data from Open-Meteo (free, no key)."""
    # Atmospheric
    atm_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":        BUOY_LAT,
            "longitude":       BUOY_LON,
            "current":         "surface_pressure,wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m",
            "wind_speed_unit": "ms",
            "timezone":        "UTC",
        },
        timeout=8,
    )
    atm_resp.raise_for_status()
    atm = atm_resp.json()["current"]

    result = {
        "pressure":    round(atm["surface_pressure"], 1),
        "wind_speed":  round(atm["wind_speed_10m"], 1),
        "wind_deg":    atm["wind_direction_10m"],
        "temp_air":    round(atm["temperature_2m"], 1),
        "humidity":    atm["relative_humidity_2m"],
        "description": "—",
        "visibility_km": None,
    }

    # Marine (wave height) — separate endpoint
    try:
        mar_resp = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude":  BUOY_LAT,
                "longitude": BUOY_LON,
                "current":   "wave_height,wave_direction,wave_period",
                "timezone":  "UTC",
            },
            timeout=8,
        )
        mar_resp.raise_for_status()
        mar = mar_resp.json()["current"]
        result["wave_height"]    = round(mar["wave_height"], 2)
        result["wave_direction"] = mar["wave_direction"]
        result["wave_period"]    = round(mar["wave_period"], 1)
    except Exception as e:
        log.warning("Marine API skipped: %s", e)
        result["wave_height"]    = None
        result["wave_direction"] = None
        result["wave_period"]    = None

    return result


@app.route("/weather")
def weather():
    global _weather_cache

    if _weather_cache.get("fetched_at"):
        age = time.time() - _weather_cache["fetched_at"]
        if age < WEATHER_TTL:
            log.info("Weather cache hit — age %.0fs", age)
            return jsonify({**_weather_cache, "cached": True})

    try:
        if OPENWEATHER_API_KEY:
            log.info("Fetching weather via OpenWeatherMap at (%.2f, %.2f)", BUOY_LAT, BUOY_LON)
            data = _fetch_openweather()
            # Also grab wave data from Open-Meteo marine (free)
            try:
                mar_resp = requests.get(
                    "https://marine-api.open-meteo.com/v1/marine",
                    params={
                        "latitude":  BUOY_LAT,
                        "longitude": BUOY_LON,
                        "current":   "wave_height,wave_direction,wave_period",
                        "timezone":  "UTC",
                    },
                    timeout=8,
                )
                mar_resp.raise_for_status()
                mar = mar_resp.json()["current"]
                data["wave_height"]    = round(mar["wave_height"], 2)
                data["wave_direction"] = mar["wave_direction"]
                data["wave_period"]    = round(mar["wave_period"], 1)
            except Exception as e:
                log.warning("Marine API skipped: %s", e)
                data["wave_height"] = data["wave_direction"] = data["wave_period"] = None
        else:
            log.info("OPENWEATHER_API_KEY not set — falling back to Open-Meteo")
            data = _fetch_open_meteo()

        _weather_cache = {
            **data,
            "lat":        BUOY_LAT,
            "lon":        BUOY_LON,
            "fetched_at": time.time(),
            "fetched_at_iso": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "Weather → %.1f°C  %.1f hPa  wind %.1f m/s  wave %s m",
            data["temp_air"], data["pressure"], data["wind_speed"],
            data.get("wave_height", "—"),
        )
        return jsonify({**_weather_cache, "cached": False})

    except requests.RequestException as e:
        log.error("Weather fetch failed: %s", e)
        return jsonify({"error": f"Weather API failed: {e}"}), 502

    except (KeyError, TypeError) as e:
        log.error("Weather parse error: %s", e, exc_info=True)
        return jsonify({"error": "Unexpected weather response format"}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  Route: /health
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    age = round(time.time() - _cache["fetched_at"], 1) if _cache["fetched_at"] else None
    return jsonify({
        "status":               "ok",
        "news_key_set":         bool(NEWS_API_KEY),
        "groq_key_set":         bool(GROQ_API_KEY),
        "openweather_key_set":  bool(OPENWEATHER_API_KEY),
        "buoy_lat":             BUOY_LAT,
        "buoy_lon":             BUOY_LON,
        "cache_age_s":          age,
        "weather_route":        "/weather",
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 52)
    print("  🌊  OceanData Server  (Groq Edition)")
    print("═" * 52)
    print("  Dashboard  →  http://localhost:5000")
    print("  API        →  http://localhost:5000/ocean-summary")
    print("  Weather    →  http://localhost:5000/weather")
    print("  Health     →  http://localhost:5000/health")
    print("═" * 52)

    if not NEWS_API_KEY:
        print("  ⚠  NEWS_API_KEY not set in .env")
    else:
        print("  ✓  NewsAPI key loaded")

    if not GROQ_API_KEY:
        print("  ⚠  GROQ_API_KEY not set in .env")
    else:
        print("  ✓  Groq API key loaded")

    if not OPENWEATHER_API_KEY:
        print("  ⚠  OPENWEATHER_API_KEY not set — falling back to Open-Meteo")
    else:
        print("  ✓  OpenWeatherMap key loaded")

    print("═" * 52 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)