from fastapi import APIRouter
from starlette.responses import JSONResponse

router = APIRouter()


@router.get("/")
async def test_connection():
    return JSONResponse(content="OK", status_code=200)
