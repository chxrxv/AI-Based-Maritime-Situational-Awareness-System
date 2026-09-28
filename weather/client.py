"""Weather data client. Provider: Open-Meteo (free, no API key required).

Swap providers later by replacing fetch_weather()'s implementation --
the return shape (WeatherSnapshot) is what the dashboard depends on.
"""

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODE_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def describe_weather_code(code):
    return WMO_CODE_DESCRIPTIONS.get(code, f"Unknown ({code})")


def fetch_weather(latitude, longitude, timeout=10):
    """Fetch current conditions + hourly (24h) + daily forecast for a location.

    Returns a plain dict:
      {
        "current": {temperature_c, humidity_pct, apparent_temperature_c,
                    condition, wind_speed_kmh, wind_direction_deg, visibility_m},
        "hourly": [{time, temperature_c, condition, visibility_m}, ...],  # next 24h
        "daily": {date, condition, temp_max_c, temp_min_c, precipitation_mm, wind_speed_max_kmh},
      }

    Raises requests.RequestException on network/API failure -- callers decide
    how to degrade (e.g. show "weather unavailable" in the dashboard).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,wind_direction_10m,visibility",
        "hourly": "temperature_2m,weather_code,visibility",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                 "precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "forecast_days": 1,
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    current = data["current"]
    hourly = data["hourly"]
    daily = data["daily"]

    return {
        "current": {
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "apparent_temperature_c": current["apparent_temperature"],
            "condition": describe_weather_code(current["weather_code"]),
            "wind_speed_kmh": current["wind_speed_10m"],
            "wind_direction_deg": current["wind_direction_10m"],
            "visibility_m": current["visibility"],
        },
        "hourly": [
            {
                "time": hourly["time"][i],
                "temperature_c": hourly["temperature_2m"][i],
                "condition": describe_weather_code(hourly["weather_code"][i]),
                "visibility_m": hourly["visibility"][i],
            }
            for i in range(len(hourly["time"]))
        ],
        "daily": {
            "date": daily["time"][0],
            "condition": describe_weather_code(daily["weather_code"][0]),
            "temp_max_c": daily["temperature_2m_max"][0],
            "temp_min_c": daily["temperature_2m_min"][0],
            "precipitation_mm": daily["precipitation_sum"][0],
            "wind_speed_max_kmh": daily["wind_speed_10m_max"][0],
        },
    }
