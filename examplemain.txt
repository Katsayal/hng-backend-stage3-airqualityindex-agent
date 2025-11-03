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
    try:
        body = await request.json()
        event = TelexEvent(**body)

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

# === WebSocket Endpoint ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
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

# === A2A Endpoint ===
@app.post("/a2a/aqi")
async def a2a_aqi_endpoint(request: Request):
    """
    A2A-compliant endpoint for Air Quality Index Agent.
    Accepts JSON-RPC 2.0 payloads and returns TaskResult responses.
    Only processes the latest meaningful user text from messages.
    Fully guarded with logging for debugging.
    """
    def extract_latest_text(message) -> str:
        """
        Extracts the last meaningful user text from a message.
        Ignores HTML, JSON, empty strings, and safely handles nested 'data' arrays.
        """
        if not message or not getattr(message, "parts", None):
            return ""

        all_texts = []

        def collect_texts(parts):
            for part in parts:
                kind = getattr(part, "kind", None)

                if kind == "text":
                    text = getattr(part, "text", "").strip()
                    if text and not text.startswith("<") and not text.startswith("{"):
                        all_texts.append(text)

                elif kind == "data" and isinstance(getattr(part, "data", None), list):
                    for d in part.data:
                        # Guard against unexpected dict structure
                        if isinstance(d, dict):
                            t = d.get("text", "").strip()
                            if t and not t.startswith("<") and not t.startswith("{"):
                                all_texts.append(t)

        collect_texts(message.parts)

        return all_texts[-1] if all_texts else "No valid message found."

    try:
        body = await request.json()
        logger.info(f"Received A2A body: {body}")

        # Validate basic JSON-RPC structure
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

        # Parse into Pydantic model
        rpc_request = JSONRPCRequest(**body)
        logger.info(f"RPC request parsed. Method: {rpc_request.method}, ID: {rpc_request.id}")

        # Extract message depending on method
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
                    "error": {"code": -32601, "message": f"Unsupported method: {rpc_request.method}"}
                }
            )

        logger.info(f"Message extracted. Parts: {len(getattr(message, 'parts', [])) if message else 0}")

        # Safely extract the latest text
        text = extract_latest_text(message)
        logger.info(f"Latest text extracted: '{text}'")

        # Build Telex-style event
        from models import TelexEvent, TelexEventData
        telex_event = TelexEvent(type="message", data=TelexEventData(text=text))

        # Safely call handle_telex_event
        try:
            response = await handle_telex_event(telex_event)
            summary_text = getattr(response.data, "summary", "No summary returned")
        except Exception as e:
            logger.exception("handle_telex_event failed")
            summary_text = f"Error handling event: {str(e)}"

        logger.info(f"Response summary: '{summary_text}'")

        # Build A2A-compliant response
        agent_message = A2AMessage(
            role="agent",
            parts=[MessagePart(kind="text", text=summary_text)]
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
        logger.exception("A2A endpoint failed")
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
