# tripora-agent

An AI travel-planner agent built to learn FastAPI and the LangChain ecosystem.
See the [roadmap](docs/ROADMAP.md) and the per-milestone [study notes](docs/learn/).

## Run

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY, and LANGSMITH_API_KEY for tracing
uv run uvicorn app.main:app --reload --port 8001
```

API docs: http://localhost:8001/docs

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check |
| `POST /itinerary` | Day-by-day itinerary from trip details (M1) |
| `POST /chat` | Travel assistant agent with tools (M2), per-`thread_id` memory (M3), trip-brief approval before building a plan (M4) |
| `POST /chat/stream` | Same agent, streamed as server-sent events: tool calls, tool results, tokens (M3) |
