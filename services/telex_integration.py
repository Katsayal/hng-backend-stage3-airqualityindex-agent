from dotenv import load_dotenv
load_dotenv()

import os
import re
import logging
from typing import Optional, Dict
import google.generativeai as genai

from models import TelexEvent, TelexResponse, TelexResponseData
from utils.validators import (
    validate_location,
    validate_aqi,
    validate_summary,
    build_telex_response,
)
from .aqi_service import fetch_aqi

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in environment")
genai.configure(api_key=api_key)

# === Logging setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# === Simple in-memory conversation memory ===
SESSION_MEMORY: Dict[str, Dict[str, str]] = {}

# === Helper: extract location using Gemini or regex ===
async def extract_location_gemini(text: str) -> Optional[str]:
    """
    Extracts a location name from free-form text using Gemini API,
    with regex and fallback handling.
    """
    text = text.strip()

    # Single word input — likely a city
    if len(text.split()) == 1 and text.isalpha():
        return validate_location(text)

    prompt = (
        f"Extract ONLY the city or location name from this message: '{text}'. "
        f"Respond with only the location, no extra text or punctuation."
    )

    location = None
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        location = response.text.strip()
        location = validate_location(location)
        logger.info(f"Gemini extracted location: {location}")
    except Exception as e:
        logger.warning(f"Gemini extraction error: {e}")

    # Fallback regex if Gemini fails or returns empty
    if not location:
        fallback = re.search(r"in\s+([A-Za-z\s]+)[\?\.]?", text)
        if fallback:
            location = fallback.group(1).strip()
            location = re.sub(r"\b(right now|today|currently|please|now)\b", "", location, flags=re.I)
            location = validate_location(location)
            logger.info(f"Regex fallback extracted: {location}")
            
    return location


# === Handle incoming events ===
async def handle_telex_event(event: TelexEvent, session_id: Optional[str] = "default") -> TelexResponse:
    """
    Handles incoming Telex events:
    - Extracts or remembers location
    - Fetches AQI
    - Builds validated response
    """

    try:
        text = event.data.text.strip()
        logger.info(f"Received message: '{text}'")

        # Retrieve or create session memory
        session = SESSION_MEMORY.get(session_id or "default", {})

        # Try to extract location
        location = await extract_location_gemini(text)
        if not location:
            # If user said something like "What about there?" use last known
            last_location = session.get("last_location")
            if last_location:
                logger.info(f"Using remembered location: {last_location}")
                location = last_location
            else:
                summary = "Sorry, I couldn't determine the location from your message."
                logger.warning(summary)
                return TelexResponse(
                    type="message",
                    data=TelexResponseData(location=None, aqi=None, summary=summary),
                )

        # Remember this location for the session
        session["last_location"] = location
        SESSION_MEMORY[session_id or "default"] = session

        # Fetch AQI
        aqi = fetch_aqi(location)
        if aqi is None:
            summary = f"Sorry, I couldn't fetch the air quality for {location}."
            logger.error(f"AQI fetch failed for {location}")
        else:
            # Simple classification
            if aqi <= 50:
                level = "good"
            elif aqi <= 100:
                level = "moderate"
            elif aqi <= 150:
                level = "unhealthy for sensitive groups"
            elif aqi <= 200:
                level = "unhealthy"
            elif aqi <= 300:
                level = "very unhealthy"
            else:
                level = "hazardous"
            summary = f"Air quality in {location} is {level} (AQI {aqi})."
            logger.info(f"Generated summary: {summary}")

        # Build validated response
        response_dict = build_telex_response(location, aqi, summary)
        return TelexResponse(**response_dict)

    except Exception as e:
        logger.exception(f"Error handling event: {e}")
        return TelexResponse(
            type="error",
            data=TelexResponseData(location=None, aqi=None, summary=str(e)),
        )
