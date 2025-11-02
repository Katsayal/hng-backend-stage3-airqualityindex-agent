import os
import uuid
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from models import TelexEvent, TelexEventData
from services.telex_integration import handle_telex_event
from utils.errors import handle_exception
from modelss.a2a import (
    JSONRPCRequest, JSONRPCResponse, TaskResult,
    TaskStatus, A2AMessage, MessagePart, Artifact
)


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
        

@app.post("/a2a/aqi")
async def a2a_aqi_endpoint(request: Request):
    """
    A2A-compliant endpoint for Air Quality Index Agent.
    Accepts JSON-RPC 2.0 payloads and returns TaskResult responses.
    """

    def extract_message_text(message) -> str:
        """
        Safely extract all text from a message object,
        handling parts with kind="text" and kind="data".
        """
        if not message or not getattr(message, "parts", None):
            return ""

        texts = []
        for part in message.parts:
            if getattr(part, "kind", None) == "text" and getattr(part, "text", None):
                texts.append(part.text)
            elif getattr(part, "kind", None) == "data" and isinstance(getattr(part, "data", None), list):
                for d in part.data:
                    if isinstance(d, dict) and d.get("text"):
                        texts.append(d["text"])
        return " ".join(texts).strip()

    try:
        body = await request.json()
        print("Received body:", body)

        # Validate JSON-RPC base structure
        if body.get("jsonrpc") != "2.0" or "id" not in body:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: jsonrpc must be '2.0' and id is required"
                    }
                }
            )

        rpc_request = JSONRPCRequest(**body)

        # Extract the message object depending on method
        message = None
        if rpc_request.method == "message/send":
            message = getattr(rpc_request.params, "message", None)
            if message is None:
                messages = getattr(rpc_request.params, "messages", None)
                if isinstance(messages, list) and messages:
                    message = messages[-1]
        elif rpc_request.method == "execute":
            params = rpc_request.params
            messages = getattr(params, "messages", None) or getattr(params, "message", None)
            if isinstance(messages, list):
                message = messages[-1]
            else:
                message = messages
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": rpc_request.id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported method: {rpc_request.method}"
                    }
                }
            )

        # Safely extract text from the message
        text = extract_message_text(message)

        # Build a Telex-style event so we can reuse existing logic
        from models import TelexEvent
        telex_event = TelexEvent(type="message", data=TelexEventData(text=text))
        response = await handle_telex_event(telex_event)

        # Build A2A-compliant message
        agent_message = A2AMessage(
            role="agent",
            parts=[MessagePart(kind="text", text=response.data.summary)]
        )

        task_result = TaskResult(
            id=response.data.location or "task-default",
            contextId="context-1",
            status=TaskStatus(state="completed", message=agent_message),
            artifacts=[],
            history=[agent_message]
        )

        rpc_response = JSONRPCResponse(id=rpc_request.id, result=task_result)
        return rpc_response.model_dump()

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": body.get("id") if "body" in locals() else None,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": {"details": str(e)}
                }
            }
        )
