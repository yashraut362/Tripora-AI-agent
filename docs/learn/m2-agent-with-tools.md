# M2: Agent with tools

`POST /chat` takes a free-text message. Unlike M1, *the model decides what to do*: it can
look up a place, fetch a weather forecast, or convert currency, as many times as it needs,
before answering.

Read in this order:

1. [`app/tools.py`](../../app/tools.py): three plain Python functions with `@tool`
2. [`app/agent.py`](../../app/agent.py): `create_agent(...)` and the `/chat` endpoint

## M1 chain vs M2 agent

|  | M1 chain | M2 agent |
|---|---|---|
| Who decides the steps? | You: always prompt → model, once | The model, at runtime |
| LLM calls per request | Exactly 1 | 1 + one per round of tool use |
| Can use live data? | No, only what the model already knows | Yes, through tools |

## A real run

`"I am going to Jaipur for the next 3 days with a budget of 200 USD. What will the weather be like, and how much is my budget in rupees?"`

```
[human] I am going to Jaipur for the next 3 days ...
[ai]    calls -> get_coordinates(place="Jaipur"), convert_currency(200, "USD", "INR")   ← LLM call 1
[tool]  get_coordinates  -> {"latitude": 26.92, "longitude": 75.79, ...}
[tool]  convert_currency -> {"rate": 96.73, "converted": 19346.0, "currency": "INR"}
[ai]    calls -> get_weather_forecast(latitude=26.92, longitude=75.79, days=3)       ← LLM call 2
[tool]  get_weather_forecast -> {"time": [...], "temperature_2m_max": [34.3, ...], ...}
[ai]    "Jaipur for the next 3 days: Oct 10: 34.3°C / 24.5°C ..."                    ← LLM call 3
```

Three things happened that nobody coded:
- **Parallel tool calls.** Location and currency don't depend on each other, so the model
  asked for both in one turn.
- **Chaining.** The weather tool needs coordinates, so the model waited for
  `get_coordinates` and passed its output into the next call. It learned that from the
  docstring ("call get_coordinates first").
- **Stopping.** Once it had everything, it replied with text instead of a tool call, which
  ends the loop.

## The concepts

**Tool (`@tool`).** A normal function. The decorator turns its **name, type hints and
docstring** into a schema sent to the model with every request (roughly
`{"name": "convert_currency", "description": "...", "parameters": {"amount": number, ...}}`).
The model never runs your code. It only *asks* for a call by name with JSON arguments, and
LangChain runs the function. So:
- **The docstring is a prompt.** A vague docstring gives you a confused agent.
- **Type hints are validation.** The model's arguments are checked against them.
- **The return value goes back to the model as text** (dicts become JSON), in a `ToolMessage`.
- Returning `{"error": ...}` instead of raising lets the model read the problem and try
  something else, like a different spelling of the place.

**Messages, extended.** M1 had system, human, and ai messages. Agents add two things:
- `AIMessage.tool_calls`: the model's request, e.g. `[{"name": "get_coordinates", "args": {...}, "id": "call_1"}]`
- `ToolMessage`: a tool's result, linked to the request by that `id`

The `steps` field in the `/chat` response is exactly this list. The agent's whole "memory"
during a run is this growing list of messages.

**`create_agent(model, tools, system_prompt)`.** LangChain's prebuilt agent loop:

```
         ┌─────────────────────────────┐
         ▼                             │
  messages ──► model ──► tool_calls? ──yes──► run the tools, append ToolMessages
                              │
                              no
                              ▼
                       final answer (last message)
```

You call it like any Runnable: `agent.ainvoke({"messages": [...]})`. It returns the state:
`{"messages": [every message from the run]}`. Underneath, `create_agent` builds a
**LangGraph** graph with a "model" node and a "tools" node. In M4 you'll build that graph
yourself.

**`model="openai:gpt-5.4-mini"`.** `create_agent` takes a `provider:model` string and calls
`init_chat_model` for you (or you can pass a model object, like in M1).

**Sync tools in an async endpoint.** The tools use plain `httpx.get` (blocking). When the
agent runs with `ainvoke`, LangChain runs sync tools in a thread pool, so the server isn't
blocked. You could write `async def` tools with `httpx.AsyncClient`, and they'd work the
same way.

## In LangSmith

One `/chat` request now shows up as a tree: the agent at the top, with a `model` step and
a `tools` step for each round of the loop. Open the `ChatOpenAI` runs and look at the
**tools** sent with each request: that's your docstrings, converted to JSON schema. Also
compare tokens: every round re-sends the whole message list, so agents cost more than M1.

## Try it

```bash
curl -s -X POST localhost:8001/chat -H 'content-type: application/json' -d '{"message": "Is it a good week for a beach trip to Goa? And what is 500 EUR in INR?"}'
```

Or use `/docs` → `POST /chat`.

## Play with it

1. Ask something that needs no tools ("What's Jaipur famous for?") and look at `steps`. How many LLM calls?
2. Delete "call get_coordinates first" from the weather docstring. Does the agent still figure it out?
3. Ask about a place that doesn't exist and watch it handle the `{"error": ...}`.
4. Add a tool, e.g. `get_local_time(timezone: str)` using `datetime` + `zoneinfo`, and add it to `tools=[...]`.
5. Ask a follow-up question in a second request ("and the day after?"). It has no memory, which is what M3 fixes.
