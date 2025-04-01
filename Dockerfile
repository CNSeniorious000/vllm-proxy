FROM ghcr.io/astral-sh/uv:python3.13-alpine AS py
WORKDIR /app
COPY pyproject.toml .
RUN uv sync --compile-bytecode --no-cache
COPY . .

CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0"]
