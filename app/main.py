from typing import Dict

from fastapi import FastAPI

from src.router import router

app = FastAPI()


@app.get("/", tags=["Root"])
def root() -> Dict[str, str]:
    return {"message": "Welcome to the service for recognizing water meter readings!"}


app.include_router(router)
