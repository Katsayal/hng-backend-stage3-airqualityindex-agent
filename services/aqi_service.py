import requests
import os
import urllib.parse


def fetch_aqi(location: str):
    WAQI_TOKEN = os.getenv("WAQI_TOKEN")
    if not WAQI_TOKEN:
        print("WAQI_TOKEN not set in environment")
        return None

    location_encoded = urllib.parse.quote(location)
    url = f"https://api.waqi.info/feed/{location_encoded}/?token={WAQI_TOKEN}"
    
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        print(f"WAQI API response for {location}: {data}")  # Debug
        if data.get("status") == "ok":
            return data["data"]["aqi"]
    except Exception as e:
        print(f"Error fetching AQI: {e}")
    return None
