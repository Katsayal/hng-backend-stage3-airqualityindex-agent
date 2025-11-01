# 🌍 Telex Air Quality Index (AQI) AI Agent

An intelligent FastAPI-based agent that provides **real-time Air Quality Index (AQI)** information for any city worldwide.  
It integrates with **Telex**, **Google Gemini**, and the **World Air Quality Index (WAQI)** API to understand natural language and respond conversationally — via both **webhooks** and **WebSocket**.



## 🧠 Overview

This AI agent:
- Understands **free-form messages** like  
  > “What’s the air quality in Kaduna right now?”  
  > “AQI for New York please”  
  > “And what about Abuja?”  
- Extracts the **city name** using **Gemini AI** (with regex fallback).  
- Fetches real-time **AQI data** from [WAQI](https://waqi.info).  
- Generates natural summaries (e.g., “Air quality in London is good (AQI 45).”)  
- Supports **persistent conversation sessions** through **WebSockets** or `session_id`s in the webhook URL.  
- Has full **logging**, **error handling**, and **validation**.


## 🏗️ Project Structure

```

airqualityindex_agent/
│
├── main.py                     # FastAPI entry point
├── models.py                   # Pydantic models for requests/responses
├── requirements.txt             # Dependencies
├── Procfile                     # Deployment process file
├── .env                         # Environment variables
│
├── services/
│   ├── aqi_service.py           # Fetches AQI data from WAQI API
│   ├── gemini_service.py        # Summarizes air quality using Gemini
│   └── telex_integration.py     # Handles Telex events + location extraction
│
├── utils/
│   ├── validators.py            # Input/output validation & normalization
│   └── errors.py                # Centralized exception handler
│
└── **pycache**/

````


## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Katsayal/hng-backend-stage3-airqualityindex-agent.git
cd hng-backend-stage3-airqualityindex-agent
````


### 2️⃣ Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```


### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```


### 4️⃣ Create Your `.env` File

Create a `.env` file in the root directory with the following contents:

```ini
# .env
GEMINI_API_KEY=your_google_gemini_api_key_here
WAQI_TOKEN=your_waqi_token_here
TELEX_WEBHOOK_URL=https://api.telex.im/agents/webhook
ENV=development
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

> 🔑 You can get your WAQI token from [https://aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/)
> 🔑 You can generate your Gemini API key from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)


### 5️⃣ Run the Application

```bash
uvicorn main:app --reload
```

The agent will start on:

```
http://localhost:8000
```

---

## ⚡ API Endpoints

### ✅ **GET /**

Health & info check.

**Response:**

```json
{
  "message": "🌍 Telex AQI Agent is running!",
  "endpoints": {
    "health": "/health",
    "webhook": "/webhook",
    "websocket": "/ws"
  }
}
```


### ✅ **POST /webhook**

Handles Telex-style messages via HTTP.

**Request Example:**

```json
{
  "type": "message",
  "data": { "text": "What's the air quality in London right now?" }
}
```

**Response Example:**

```json
{
  "type": "message",
  "data": {
    "location": "London",
    "aqi": 45,
    "summary": "Air quality in London is good (AQI 45)."
  }
}
```

**Optional Query or Header for Sessions:**

```
POST /webhook?session_id=user123
```

or

```
Header: X-Session-ID: user123
```

This allows the agent to remember the last city mentioned in the same session.

---

### ✅ **WebSocket /ws**

Interactive, real-time conversation endpoint.

**Example flow:**

```
Client → {"type": "message", "data": {"text": "What's the AQI in Lagos?"}}
Agent  → {"type": "message", "data": {"location": "Lagos", "aqi": 67, "summary": "Air quality in Lagos is moderate (AQI 67)."}}

Client → {"type": "message", "data": {"text": "And what about Abuja?"}}
Agent  → {"type": "message", "data": {"location": "Abuja", "aqi": 82, "summary": "Air quality in Abuja is moderate (AQI 82)."}}
```

Each WebSocket connection has its own **session memory**.

---

## 🧩 Core Features

| Feature                               | Description                                      |
| ------------------------------------- | ------------------------------------------------ |
| 🌐 **Natural Language Understanding** | Uses Gemini + regex to extract cities from text. |
| 💨 **Real-time AQI Data**             | Fetches from WAQI API with live results.         |
| 🧠 **Memory**                         | Remembers last city per session or user.         |
| 🔐 **Validation**                     | All outputs validated via `validators.py`.       |
| 📜 **Logging**                        | Timestamped structured logs for all events.      |
| 🧯 **Error Handling**                 | Graceful fallback responses for all exceptions.  |

---

## 🧰 Tech Stack

* **Python 3.11+**
* **FastAPI** (async web framework)
* **Google Gemini API**
* **WAQI (World Air Quality Index) API**
* **Pydantic** (data models & validation)
* **dotenv** (environment management)
* **Uvicorn** (ASGI server)

---

## 🔍 Logs and Debugging

Logs are shown in your terminal in structured format:

```
[2025-11-01 20:03:58] INFO - Received message: "What's the air quality in London right now?"
[2025-11-01 20:04:02] INFO - Generated summary: Air quality in London is good (AQI 45).
```

---

## 🧪 Testing Locally

You can test the webhook endpoint using `curl` or Postman:

```bash
curl -X POST http://127.0.0.1:8000/webhook \
-H "Content-Type: application/json" \
-d '{"type": "message", "data": {"text": "What’s the air quality in Lagos?"}}'
```

## ⚠️ Troubleshooting

| Issue                                                              | Cause                              | Fix                                                                        |
| ------------------------------------------------------------------ | ---------------------------------- | -------------------------------------------------------------------------- |
| `ValueError: GEMINI_API_KEY not set in environment`                | `.env` not loaded before import    | Add `load_dotenv()` at top of `telex_integration.py`                       |
| `Gemini extraction error: Your default credentials were not found` | Gemini SDK trying to use ADC       | Ensure `genai.configure(api_key=...)` runs after `.env` loaded             |
| `"Unknown station"` from WAQI                                      | City not recognized                | Try using broader location (e.g., "London" instead of "London right now")  |
| WebSocket disconnects quickly                                      | Missing `await` or invalid payload | Make sure to send full `{"type": "message", "data": {"text": "..."}}` JSON |

