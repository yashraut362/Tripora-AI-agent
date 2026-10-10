# M3: Memory and streaming

Two upgrades to the M2 agent, both in [`app/agent.py`](../../app/agent.py):

- **Memory.** Requests with the same `thread_id` are one conversation, so follow-ups like "and there?" work.
- **Streaming.** `POST /chat/stream` sends status updates and answer tokens *while* the agent works, instead of one JSON blob at the end.

## Memory: checkpointer + thread_id

```python
agent = create_agent(..., checkpointer=InMemorySaver())
await agent.ainvoke({"messages": [new_message]}, {"configurable": {"thread_id": "abc"}})
```

**What a checkpointer does.** The agent is a LangGraph graph, and its *state* is the
message list. After every step (each model call, each tool round) LangGraph saves a
**checkpoint**, a snapshot of that state, under the `thread_id`. On the next request with
the same `thread_id`, it loads the latest checkpoint and appends your new message to it.

```
request 1 (thread test-1): "Weather in Goa?"
   state: [human, ai→tool_call, tool, ai answer]                   ← saved
request 2 (thread test-1): "How much is 100 USD there?"
   state: [human, ai, tool, ai answer,  human "100 USD there?", …]  ← loaded + appended
```

That's why you send **only the new message**. The `steps` count in the `/chat` response
grows with every turn on a thread.

**A real run** shows why memory matters:

| Thread | Message | Answer |
|---|---|---|
| `test-1` | "What's the weather in Goa for the next 2 days?" | Goa forecast |
| `test-1` | "And how much is 100 USD there in local currency?" | ₹9,673 ✅ knew "there" = Goa |
| `test-2` | same follow-up, new thread | "500.50 BRL in Teresópolis" ❌ no context, so the model made up a place |

**Things to know:**
- **Memory is just the message list.** The LLM itself remembers nothing. Every request
  re-sends the whole conversation, so long threads cost more tokens per turn.
- **`InMemorySaver` lives in RAM.** A server restart wipes it, and **so does every
  `--reload`** when you edit a file. Real apps use a persistent saver
  (`langgraph-checkpoint-sqlite` / `-postgres`). It's the same one-line swap.
- **`thread_id` is chosen by the client.** Here it defaults to `"default"`, so requests
  without one all share a single conversation. A real app would create one per chat.
- Checkpoints are also what make "pause and resume" possible: M4's human approval step
  and LangSmith Deployment are both built on them.

## Streaming: `agent.astream` + server-sent events

```python
async for chunk in agent.astream(input, config, stream_mode=["messages", "updates"], version="v2"):
    chunk["type"]   # "messages" or "updates"
    chunk["data"]   # the payload
```

**Stream modes** (what you get from LangGraph):

| Mode | You get | Used here for |
|---|---|---|
| `messages` | `(token, metadata)` for every piece of LLM output as it's generated. `metadata["langgraph_node"]` says which node produced it | The answer's text, token by token |
| `updates` | Each node's output once that node finishes, e.g. `{"model": {"messages": [...]}}` | Spotting tool calls (from `model`) to send a status line |
| `values` | The full state after every step | (not used) |
| `custom` | Anything a tool sends with `get_stream_writer()` | (not used, but handy for "fetching weather…" progress) |

`version="v2"` makes every chunk the same shape (`{"type", "ns", "data"}`), so there's no
tuple unpacking that changes with the number of modes.

**`token.text` filter.** When the model decides to call a tool, its streamed output is
tool-call pieces, not text, so `token.text` is empty and they're skipped. The `updates`
event reports the finished tool call instead.

**What the client gets (production-style).** The first version streamed raw `tool_call`
and `tool_result` events (exact arguments and API JSON). That's a debugging view. Real apps
send only what the user should see:

| Event | Data | Why |
|---|---|---|
| `status` | `"Looking up Udaipur…"` (from the `TOOL_STATUS` map) | Shows progress without exposing tool names, arguments or raw data |
| `token` | the next bit of answer text | The answer appears as it's written |
| `error` | a generic message | Once the stream has started, the HTTP status (200) is already sent, so errors must be events |
| `done` | empty | Tells the client the answer is complete |

Raw tool arguments and results still exist, in **LangSmith traces**, which is the
developer's view.

**Server-sent events (SSE)** (the FastAPI side): a plain HTTP response that stays open,
where the server writes events as they happen:

```
event: status
data: "Looking up Jaipur…"

event: token
data: "Ja"

event: done
data: ""
```

FastAPI (0.135+) supports this directly: `response_class=EventSourceResponse`, then `yield
ServerSentEvent(event=..., data=...)` from an `async def` endpoint. `data` is JSON-encoded
for you. A browser can read SSE with `fetch` + a stream reader (or `EventSource` for GET
endpoints).

**A real stream** for "Weather in Udaipur tomorrow and what is 100 USD in INR?":

```
status   "Looking up Udaipur…"
status   "Converting USD to INR…"           ← same model turn as above (parallel tool calls)
status   "Checking the weather forecast…"   ← next turn, needed the coordinates first
token ×32 "Udaipur" " tomorrow" ":" " 33.7°C" ...
done
```

## Try it

```bash
# memory: run these two in order
curl -s -X POST localhost:8001/chat -H 'content-type: application/json' -d '{"message": "What is the weather in Goa this week?", "thread_id": "me-1"}' | jq .answer
curl -s -X POST localhost:8001/chat -H 'content-type: application/json' -d '{"message": "And what is 50 EUR there in local money?", "thread_id": "me-1"}' | jq .answer

# streaming (-N turns off curl's buffering so you see events arrive)
curl -N -X POST localhost:8001/chat/stream -H 'content-type: application/json' -d '{"message": "Plan a quick rainy-day idea for Jaipur tomorrow", "thread_id": "me-1"}'
```

## Play with it

1. Send the follow-up with a *different* `thread_id` and watch it lose the context.
2. Have a long conversation on one thread, then compare token counts of the first and last request in LangSmith.
3. Edit any `.py` file while the server runs with `--reload`, then ask a follow-up. Memory gone, which is why persistent checkpointers exist.
4. Add `"values"` to `stream_mode` and print what arrives. How is it different from `updates`?
5. In `get_weather_forecast`, add `from langgraph.config import get_stream_writer` and `get_stream_writer()({"status": "fetching weather…"})`, then stream the `custom` mode as a `status` event.
