"""Time and current-weather direct-action services."""
from __future__ import annotations
from datetime import datetime
import json
import urllib.parse
import urllib.request

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 80: "rain showers",
    81: "moderate rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


class TimeService:
    """Fetches the current time from TimeAPI.io, falling back to the Pi's clock."""
    @staticmethod
    def current_time(time_zone: str = "Asia/Kolkata", timeout: float = 2.0) -> str:
        import logging
        log = logging.getLogger(__name__)
        try:
            query = urllib.parse.urlencode({"timeZone": time_zone})
            request = urllib.request.Request(
                "https://timeapi.io/api/Time/current/zone?" + query,
                headers={"Accept": "application/json", "User-Agent": "EMO-Robot/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            year, month, day = int(data["year"]), int(data["month"]), int(data["day"])
            hour, minute, second = int(data["hour"]), int(data["minute"]), int(data.get("seconds", 0))
            now = datetime(year, month, day, hour, minute, second)
            clock = now.strftime("%I:%M %p").lstrip("0")
            day_name = str(data.get("dayOfWeek") or now.strftime("%A"))
            month_name = now.strftime("%B")
            zone = str(data.get("timeZone") or time_zone)
            return f"It is {clock} on {day_name}, {month_name} {day}, in {zone}."
        except Exception as exc:
            log.debug("Time API unavailable; using local Pi clock: %s", exc)
            now = datetime.now().astimezone()
            return f"It is {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B')} {now.day}."


class WeatherService:
    """Uses Open-Meteo's free geocoding and forecast endpoints; an internet connection is required."""
    def __init__(self, timeout: int = 6) -> None:
        self.timeout = timeout

    def current(self, city: str) -> str:
        city = (city or "").strip()
        if not city:
            raise ValueError("Set WEATHER_CITY in .env or ask for weather in a named city.")
        query = urllib.parse.urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
        geo = self._get_json("https://geocoding-api.open-meteo.com/v1/search?" + query)
        results = geo.get("results") or []
        if not results:
            raise ValueError(f"I couldn't find a place called {city}.")
        place = results[0]
        label = ", ".join(part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part)
        params = urllib.parse.urlencode({
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto",
        })
        data = self._get_json("https://api.open-meteo.com/v1/forecast?" + params)
        current = data.get("current") or {}
        if not current:
            raise RuntimeError("The weather service did not return current conditions.")
        code = current.get("weather_code")
        description = WEATHER_CODES.get(code, "conditions not classified") if code is not None else "conditions not classified"
        return (
            f"Current weather for {label}: {description}, {current.get('temperature_2m', '?')} degrees Celsius, "
            f"feels like {current.get('apparent_temperature', '?')} degrees, humidity "
            f"{current.get('relative_humidity_2m', '?')} percent, wind "
            f"{current.get('wind_speed_10m', '?')} kilometres per hour."
        )

    def _get_json(self, url: str) -> dict:
        request = urllib.request.Request(url, headers={"User-Agent": "EMO-Robot/1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
