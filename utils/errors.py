from fastapi.responses import JSONResponse

def handle_exception(e: Exception):
    return JSONResponse(
        status_code=500,
        content={"type": "error", "data": {"message": str(e)}}
    )
