# Tripora Agent: learning roadmap

Goal: learn FastAPI and the LangChain ecosystem by growing one travel-planner agent,
one concept at a time. Each milestone has a study note in [`docs/learn/`](learn/).

| # | Milestone | Ecosystem concept | FastAPI concept | Status |
|---|---|---|---|---|
| 0 | Skeleton | none | app, routers | ✅ |
| 1 | Quick itinerary: `POST /itinerary` | Chat models, prompts, LCEL, structured output, LangSmith Observability | Pydantic request and response models | ✅ [notes](learn/m1-quick-itinerary.md) |
| 2 | Agent with tools: weather, geocoding, currency | `@tool`, `create_agent`, the agent loop | A second router | ✅ [notes](learn/m2-agent-with-tools.md) |
| 3 | Chat that remembers and streams | Checkpointers, thread IDs, streaming | Server-sent events (`EventSourceResponse`) | ✅ [notes](learn/m3-memory-and-streaming.md) |
| 4 | Approve the trip brief: the agent pauses with what it understood, and builds the real itinerary only after you approve | Human in the loop: `HumanInTheLoopMiddleware` (uses `interrupt()`), resuming with `Command(resume=...)`; M1's chain becomes an agent tool | The same chat endpoints take a message or an approve decision | ✅ [notes](learn/m4-human-approval.md) |
| 5 | Is the agent any good? | LangSmith Evaluation: datasets, evaluators, experiments | | ✅ [notes](learn/m5-evaluation.md) |
| 6 | Deep trip planner with subagents | Deep Agents | | |
| 7 | Agent runs code (budget math, charts) | LangSmith Sandboxes | | |
| 8 | Ship it | LangSmith Deployment vs serving it yourself with FastAPI | Deployment trade-offs | |
| 9 | No-code comparison | LangSmith Fleet | | |
| ★ | Side quest, any time after M6 | Deep Agents Code (`dcode`) on this repo | | |

Possible extra after M3: RAG over destination guides (embeddings, vector store, retriever tool).
Cost note: M1–M6 need only OpenAI usage, since LangSmith's free tier covers tracing and evals.
Check LangSmith pricing before starting M7–M9.

Style: code stays minimal and readable rather than production-ready, and tests are optional.
