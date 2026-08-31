"""
Ingestion module for Weather Data.
Fetches live or historical weather conditions from OpenWeatherMap API,
with offline fallback mock generators for robust pipeline testing.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_current_weather(
    city: str = "Boston",
    api_key: str = None,
    output_dir: str = "data/raw/weather/openweather_raw"
) -> dict:
    """
    Fetch current weather metrics from OpenWeatherMap API.
    If API key is unavailable or request fails, returns realistic mock payload.
    """
    key = api_key or os.getenv("OPENWEATHER_API_KEY")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"weather_{city.lower()}_{timestamp_str}.json"
    
    if key and key.strip() and key != "YOUR_API_KEY":
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
        try:
            logger.info(f"Calling OpenWeatherMap API for city: {city}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved real weather observation to {output_file}")
                return data
            else:
                logger.warning(f"OpenWeatherMap returned code {response.status_code}: {response.text}. Using mock fallback.")
        except Exception as e:
            logger.warning(f"Failed to connect to OpenWeatherMap: {e}. Using mock fallback.")
            
    # Mock fallback payload
    logger.info(f"Generating realistic mock weather data for {city}...")
    mock_data = {
        "city": city,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weather": [{"main": "Rain", "description": "light rain"}],
        "main": {
            "temp": 14.5,
            "feels_like": 13.2,
            "humidity": 82,
            "pressure": 1012
        },
        "wind": {"speed": 4.8, "deg": 180},
        "rain": {"1h": 1.2},
        "is_mock": True
    }
    
    with open(output_file, "w") as f:
        json.dump(mock_data, f, indent=2)
        
    logger.info(f"Saved mock weather data to {output_file}")
    return mock_data


if __name__ == "__main__":
    city_name = sys.argv[1] if len(sys.argv) > 1 else "Boston"
    fetch_current_weather(city=city_name)
