from fastapi import FastAPI

from routes.connection import router as connection
import nest_asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# app.include_router(login)
app.include_router(connection)

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="localhost", port=8000)

