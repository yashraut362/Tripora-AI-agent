# M4: Approve the trip brief before the plan is built (human in the loop)

The M3 chat agent gets one new tool, `create_itinerary`. When you ask for a trip plan, the
agent **pauses** and shows the brief it understood. Only after you approve does it run the
big itinerary call (the M1 chain) and return a real day-by-day plan.

```
you:   "Plan a trip to Goa for me, 3 days, I love beaches and food"

       ┌────────────────────────────────┐
       │ 📝 Here's the trip I'll plan    │
       │ Goa · 3 days · mid budget       │
       │ beaches, food · 1 traveler      │
       │   [ Change ]    [ Build plan ]  │
       └────────────────────────────────┘
          │                 │
          │ type a change   │ approve
          ▼                 ▼
       new brief card     M1 chain runs → real Itinerary → agent writes the plan
```

Real apps do the same: Claude Code's plan mode (approve the plan before it edits files) and
ChatGPT Deep Research (confirm the scope before it starts). The rule is **confirm before slow,
expensive or important work**, so a misunderstanding doesn't cost a full run.

## The diff against M3

`git diff` shows only additions:

| File | Change |
|---|---|
| `tools.py` | `+ create_itinerary`: a plain tool that runs the M1 `chain` |
| `schemas.py` | `+ decision: "approve"` on `ChatRequest`; `message` now defaults to `""` |
| `agent.py` | `+ HumanInTheLoopMiddleware`, `+ agent_input()` (resume or new message), `+ approval_required` in both endpoints, `+ create_itinerary` in the tools, one prompt sentence |

## Why code and not the prompt?

You could write "confirm the details before planning" in the system prompt. That's a
**suggestion**: the model usually follows it, but it can skip it, or take "sure, but 4 days" as a yes.
With the middleware, the tool physically can't run until the request with `decision: "approve"` arrives.

| | Prompt: "confirm first" | `HumanInTheLoopMiddleware` |
|---|---|---|
| Can the model skip it? | Yes | No |
| What the user answers | Free text that the model interprets | A button (`approve`), or typed text for changes |
| What the app gets | A chat bubble | The brief as data, to draw the card |

## The tool: just the work, no approval logic

```python
@tool(args_schema=TripRequest)
async def create_itinerary(**trip) -> dict:
    itinerary = await chain.ainvoke(TripRequest(**trip).model_dump(mode="json"))   # the M1 chain
    return itinerary.model_dump()
```

`args_schema=TripRequest` reuses the M1 request model, so the model must fill the same fields with
the same rules (days 1–14, budget is budget/mid/luxury). M1 and M2 are now connected.

## The approval rule: middleware

```python
create_agent(
    ...,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"create_itinerary": {"allowed_decisions": ["approve", "reject"]}})
    ],
    checkpointer=InMemorySaver(),
)
```

**Middleware** is code that plugs into the agent loop. This built-in one runs **after every model
turn** and looks at the tool calls the model asked for:

```
model asks for convert_currency(...)    → not in interrupt_on → runs normally
model asks for create_itinerary(Goa, 3) → in interrupt_on → interrupt()  ⏸  (the tool hasn't run)
```

What happens:

1. `interrupt()` stops the run **before** the tool executes. The checkpointer saves the state,
   including the pending tool call, under the `thread_id` (the same M3 memory).
2. The HTTP request ends. The paused run is just saved data.
3. The next request sends `Command(resume={"decisions": [...]})` on the same thread, and
   LangGraph **continues the same run**:
   - `{"type": "approve"}`: the tool runs with the model's arguments, and the plan comes back.
   - `{"type": "reject", "message": "make it 2 days"}`: the tool **doesn't run**. The model gets a
     tool result like *"User rejected the tool call for `create_itinerary` with reason: make it 2 days"*
     and calls it again with the change, which gives a new card.

The pause happens before the tool, so the expensive chain runs exactly once, after approval.

What the middleware hands to `interrupt()` (and what we read back from `state.interrupts[0].value`):

```python
{
  "action_requests": [{"name": "create_itinerary",
                       "args": {"destination": "Goa", "days": 3, ...},
                       "description": "Tool execution requires approval ..."}],
  "review_configs": [{"action_name": "create_itinerary", "allowed_decisions": ["approve", "reject"]}],
}
```

The card only needs `action_requests[0]["args"]`, which is the brief.

## Resuming: `agent_input()`

```python
async def agent_input(req):
    state = await agent.aget_state(thread_config(req))
    if state.interrupts:                                       # a brief is waiting
        if req.decision == "approve":
            decision = {"type": "approve"}
        else:
            decision = {"type": "reject", "message": req.message}
        return Command(resume={"decisions": [decision]})
    return {"messages": [{"role": "user", "content": req.message}]}   # same as M3
```

| Thread state | Request | Sent to the agent | What happens |
|---|---|---|---|
| Nothing pending | `message` | A new user message | Same as M3 |
| Brief pending | `decision: "approve"` | `decisions: [{"type": "approve"}]` | The tool runs, and the plan comes back |
| Brief pending | Typed `message` ("make it 2 days") | `decisions: [{"type": "reject", "message": "make it 2 days"}]` | The tool doesn't run; the model retries with the change, giving a new card |

`aget_state()` reads the checkpoint (M3 memory) without running anything.

The endpoints read pending interrupts through LangGraph's typed API:

```python
result = await agent.ainvoke(input, config, version="v2")   # /chat
result.value["messages"]; result.interrupts

state = await agent.aget_state(config)                       # /chat/stream, after the stream ends
state.interrupts
```

New in the response: `approval_required` (the brief). In the stream it's a new event:

```
event: approval_required
data: {"destination": "Goa", "days": 3, "budget": "mid", "interests": ["beaches", "food"], "travelers": 1}
```

## A real run

```
→ "Plan a trip to Goa for me, 3 days, I love beaches and food"
   status: Preparing a brief for your 3-day trip to Goa…
   approval_required: {destination: Goa, days: 3, budget: mid, interests: [beaches, food], travelers: 1}
   done

→ "make it 2 days and budget"                       (typed while the card was showing)
   approval_required: {…, days: 2, budget: budget, …}
   done

→ {"decision": "approve"}
   token…: Here's your 2-day budget Goa trip focused on beaches and food:
           Day 1: North Goa: Calangute, Baga + beach-shack lunch, Anjuna sunset …
           Day 2: South Goa: Colva, Benaulim + Goan lunch, Palolem sunset …
   done
```

In `/chat`, the `steps` show the tool result is the real typed `Itinerary`
(`destination, summary, days, tips`), which the agent then writes up in chat.

## Lessons (from earlier tries)

1. **The prompt and the guardrail have to agree.** Without *"call create_itinerary right away:
   the app shows them the brief to confirm, so don't ask for confirmation in chat"*, the model
   asks "Shall I go ahead?" in chat **and** the card appears, so the user confirms twice.
2. **Tool results are prompts too.** On reject, the reason becomes the tool result the model reads,
   so a clear reason ("make it 2 days") is what makes it retry with the change.

## Other ways to do this

- **`interrupt()` inside the tool**: call `answer = interrupt(brief)` at the top of the tool and branch
  on it. It's more manual, and the tool **re-runs from the top** on resume, so any work before
  `interrupt()` runs twice. It's useful when the pause depends on logic inside the tool
  (e.g. only ask when the price is above X).
- **Prompt only**: fine when a mistake is cheap.
- The middleware also supports `"edit"` (the app sends changed args, e.g. from an editable form) and
  `"respond"` (answer in place of the tool). Add them to `allowed_decisions` to use them.

## What's still not production

| Gap | Production fix |
|---|---|
| `InMemorySaver`: a pending brief disappears on restart, and doesn't work across 2+ server instances | Postgres/Mongo checkpointer (parked in the roadmap) |
| Anyone who knows a `thread_id` can approve | Auth: tie threads to a logged-in user |
| A stale Approve tap (nothing pending) goes to the model as an empty message | Return 409 when `decision` is set but `state.interrupts` is empty, and disable the button after one tap |
| Reopening the app mid-approval: there's no way to redraw the card | `GET /chat/{thread_id}` returning `state.interrupts` |

## Try it

```bash
curl -N localhost:8001/chat/stream -H 'content-type: application/json' \
  -d '{"thread_id":"t1","message":"Plan a trip to Goa for me, 3 days, I love beaches and food"}'

curl -N localhost:8001/chat/stream -H 'content-type: application/json' \
  -d '{"thread_id":"t1","message":"make it 2 days and budget"}'

curl -N localhost:8001/chat/stream -H 'content-type: application/json' \
  -d '{"thread_id":"t1","decision":"approve"}'
```

## Play with it

1. Add `print("create_itinerary running")` as the first line of the tool. Ask, change, approve:
   it prints **once**, because rejected calls never reach the tool.
2. Add `convert_currency` to `interrupt_on`, and ask "plan Goa for 3 days and convert 500 USD to INR".
   Look at `action_requests`: there are two actions, so `decisions` needs two entries.
3. Add `"edit"` to `allowed_decisions`, then resume with
   `{"type": "edit", "edited_action": {"name": "create_itinerary", "args": {...the brief with "days": 4}}}`.
   The tool runs with your args, not the model's.
4. Remove the "don't ask for confirmation in chat" sentence and watch the double confirmation.
5. Restart the server while a brief is pending, then approve. The pending run is gone, because it was only in RAM.
