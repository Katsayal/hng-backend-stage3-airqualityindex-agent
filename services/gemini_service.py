import google.generativeai as genai
import os

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in environment")
genai.configure(api_key=api_key)

def summarize_aqi(aqi_data: dict) -> str:
    """
    Use Gemini to summarize AQI data in natural language.
    """
    if "error" in aqi_data:
        return f"Sorry, I couldn't fetch the air quality for {aqi_data.get('location', 'your area')}."

    aqi = aqi_data["aqi"]
    location = aqi_data["location"]
    dom_pollutant = aqi_data.get("dominant_pollutant", "Unknown")

    prompt = f"""
    Summarize air quality for {location} where AQI is {aqi} and dominant pollutant is {dom_pollutant}.
    Classify air safety (Good, Moderate, Unhealthy, etc.) and provide a short health advisory.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()
