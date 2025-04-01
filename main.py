from fastapi import FastAPI

from chat.server import router

app = FastAPI()

app.include_router(router)
