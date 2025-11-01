import os
import uuid
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from models import TelexEvent
from services.telex_integration import handle_telex_event
from utils.errors import handle_exception

# === Load environment variables ===
load_dotenv()

WAQI_TOKEN = os.getenv("WAQI_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Setup logging ===
logger = logging.getLogger("telex_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

if not WAQI_TOKEN:
    logger.warning("⚠️ WAQI_TOKEN not set in environment")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not set in environment")

# === Initialize FastAPI ===
app = FastAPI(title="Telex Air Quality Index Agent")

@app.get("/")
async def home():
    return {
        "message": "🌍 Telex AQI Agent is running!",
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook",
            "websocket": "/ws"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AQI Agent is running with WebSocket support"}


# === Webhook Endpoint ===
@app.post("/webhook")
async def webhook_listener(request: Request):
    """
    Handles incoming webhook events (HTTP).
    Supports optional session_id in query or headers.
    """
    try:
        body = await request.json()
        event = TelexEvent(**body)

        # Optional: allow clients to send ?session_id=xxx or Header
        session_id = (
            request.query_params.get("session_id")
            or request.headers.get("X-Session-ID")
            or "default"
        )

        logger.info(f"Webhook event received | session={session_id} | text='{event.data.text}'")

        response = await handle_telex_event(event, session_id=session_id)

        logger.info(f"Response: {response.data.summary}")
        return JSONResponse(response.model_dump(), status_code=200)

    except Exception as e:
        logger.exception("Error processing webhook event")
        return handle_exception(e)


# === WebSocket Endpoint (Persistent Conversation) ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time, stateful conversations.
    Each connected client maintains its own session memory.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())  # unique session ID per connection
    logger.info(f"New WebSocket connection: {session_id}")

    await websocket.send_json({
        "type": "status",
        "data": {"message": "Connected to AQI Agent", "session_id": session_id}
    })

    try:
        while True:
            data = await websocket.receive_json()
            event = TelexEvent(**data)
            logger.info(f"WS message from session={session_id}: {event.data.text}")

            response = await handle_telex_event(event, session_id=session_id)
            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        logger.info(f"WebSocket {session_id} disconnected.")
    except Exception as e:
        logger.exception("Error in WebSocket session")
        await websocket.send_json({"type": "error", "data": {"message": str(e)}})
