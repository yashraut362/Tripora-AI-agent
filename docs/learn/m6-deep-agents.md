# M6: Deep planner for big trips (Deep Agents)

Code: [`app/deep_planner.py`](../../app/deep_planner.py), endpoint `POST /deep-plan/stream`.
New dependency: `deepagents` (`uv add deepagents`).

## What

- A "deep plan" mode for big, multi-city trips, like ChatGPT's Deep Research button: it takes ~30s,
  shows progress, and produces a full document.
- It runs next to the M2–M4 chat agent, which still handles quick questions.

```
"Plan a 7-day Japan trip: Tokyo, Kyoto, Osaka. Food and temples. 200000 INR for 2."

main agent
 ├─ write_todos         → checklist: convert budget, research 3 cities, write plan
 ├─ convert_currency    → 200,000 INR ≈ 327,200 JPY
 ├─ task(city-researcher, "Tokyo …") ┐
 ├─ task(city-researcher, "Kyoto …") ├─ subagents, in parallel, each with its own clean context
 ├─ task(city-researcher, "Osaka …") ┘
 ├─ write_file /trip_plan.md   → route, nights per city, budget split, day-by-day
 └─ reply: 2-line summary
```

## Why a deep agent (vs the M2 `create_agent`)

A plain agent loop breaks down on long tasks: it forgets steps, and every tool result piles into one
context until it's huge and confused. A deep agent is the **same loop** plus 3 things:

| Piece | Tool | What it fixes |
|---|---|---|
| **Planning** | `write_todos` (`TodoListMiddleware`) | Keeps a checklist, so it doesn't lose track over many steps. The UI shows it as progress |
| **Subagents** | `task` | Hands a sub-job (one city) to a helper agent with a **fresh context**. Only its short summary comes back, so the main context stays small. Several run in parallel |
| **Filesystem** | `write_file`, `read_file`, `edit_file`, `ls`, … | Long output goes to a file, not the chat. Follow-ups **edit** the file instead of regenerating |

The same pattern is behind Claude Code, Deep Research and Manus.

## The code

```python
city_researcher = {                              # a subagent = name + description + prompt + tools
    "name": "city-researcher",
    "description": "Researches ONE city for a trip ...",     # the main agent reads this to decide when to delegate
    "system_prompt": "Research the one city you are given ...",
    "tools": [get_coordinates, get_weather_forecast],         # the M2 tools, reused
}

deep_planner = create_deep_agent(
    model="openai:gpt-5.4-mini",
    tools=[convert_currency],
    system_prompt="1. write_todos ... 2. delegate each city ... 4. write /trip_plan.md ...",
    subagents=[city_researcher],
    middleware=[TodoListMiddleware()],          # planning tool (not in the deepagents 0.7 defaults)
    checkpointer=InMemorySaver(),               # M3 memory: follow-ups on the same thread_id
)
```

- `create_deep_agent` = `create_agent` + filesystem + `task`/subagents + summarization, all as **middleware** (M4's idea).
- **Files live in the agent's state** (`state["files"]`), saved by the checkpointer. It's a virtual filesystem, not your disk.
- The subagent is **isolated**: it only sees the task description the main agent wrote, not the whole chat.

## The endpoint: progress events

It's long-running, so it's stream-only. It reuses the M3 pattern: `astream(stream_mode="updates")` and
look at the tool calls the model asks for.

| Event | When | Data |
|---|---|---|
| `todos` | Every `write_todos` call | The checklist, with `pending` / `in_progress` / `completed` per item |
| `status` | `task` / `write_file` / `edit_file` | "Researching: Tokyo …", "Writing your trip plan…" |
| `plan` | At the end | The content of `/trip_plan.md`, read from `aget_state()` |
| `summary` | At the end | The agent's short final reply |
| `error` / `done` | | Same as M3 |

## A real run

```
todos:   [Convert budget (in_progress), Research Tokyo, Research Kyoto, Research Osaka, Write plan]
status:  Researching: Research Tokyo for a 7-day Japan trip focused on food and temples…
status:  Researching: … Kyoto …
status:  Researching: … Osaka …
todos:   [… all completed except Write plan]
status:  Writing your trip plan…
plan:    # 7-Day Japan Trip Plan: Tokyo, Kyoto, Osaka
         Tokyo 3 nights · Kyoto 2 · Osaka 1 · budget split in JPY · day-by-day …
summary: Your 7-day Japan plan is ready … 200,000 INR is about 327,200 JPY …
                                                           (~30 seconds)

→ follow-up on the same thread: "Change it to 2 nights in Tokyo and 2 nights in Osaka"
status:  Writing your trip plan…  (×5: five edit_file calls, no new research)
plan:    Tokyo: 2 nights · Kyoto: 2 nights · Osaka: 2 nights …
```

## Try it

```bash
curl -N localhost:8001/deep-plan/stream -H 'content-type: application/json' \
  -d '{"thread_id":"d1","message":"Plan a 7-day Japan trip next week: Tokyo, Kyoto and Osaka. I love food and temples. Budget is 200000 INR for 2 people."}'

curl -N localhost:8001/deep-plan/stream -H 'content-type: application/json' \
  -d '{"thread_id":"d1","message":"Change it to 2 nights in Tokyo and 2 nights in Osaka"}'
```

Then open the trace in LangSmith: the main agent run, with each `task` as a nested subagent run inside it.

## What's still not production

| Gap | Production fix |
|---|---|
| Files and threads are in RAM (`InMemorySaver`) | A Postgres checkpointer; or `StoreBackend` / `CompositeBackend` to keep files across threads |
| One HTTP request for the whole job: if the client disconnects, it's gone | Run it as a background job (queue / LangSmith Deployment background runs) and let the client reconnect to the stream |
| No cost cap | `ModelCallLimitMiddleware`, and limits on how many subagents run |

## Play with it

1. Ask for a 1-city trip ("3 days in Goa"). Does it still use todos and a subagent? (Deep agents are overkill for small tasks, which is why `/chat` exists.)
2. Add a second subagent, `budget-checker`, with only `convert_currency`, and tell the main agent to use it.
3. Remove `TodoListMiddleware()` and compare the trace: does it still finish all 3 cities?
4. Change the subagent prompt to "under 50 words". Is the final plan worse?
5. Add `interrupt_on={"write_file": True}` (M4!) to approve the plan before it's written.
