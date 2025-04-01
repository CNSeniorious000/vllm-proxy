from os import getenv

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from chat.server import router

app = FastAPI(title=getenv("TITLE", "vLLM Proxy"))

app.include_router(router)

app.get("/", include_in_schema=False)(lambda: RedirectResponse("/docs"))
