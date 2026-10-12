# Extra: Loop and hang protection

Code: `app/agent.py`, `app/deep_planner.py`, `app/tools.py`.

## The problem
- An agent decides its own next step, so it can loop: the same tool again and again, or subagent after subagent.
- `create_agent` and `create_deep_agent` set LangGraph's `recursion_limit` to **9,999 steps**, so by default
  a stuck run could make thousands of paid model calls.
- A hung API call with no timeout can also block a request forever.

## The layers we added

| Layer | Where | What it does |
|---|---|---|
| `ModelCallLimitMiddleware(run_limit=10, exit_behavior="end")` | `/chat` agent | Max 10 model calls per request, then stop **cleanly** |
| `ModelCallLimitMiddleware(run_limit=25, ...)` | Deep planner | Max 25 model calls per request |
| `ToolCallLimitMiddleware(tool_name="task", run_limit=6)` | Deep planner | Max 6 subagent tasks. Extra calls get a "limit reached" tool result instead of running |
| `ModelCallLimitMiddleware(run_limit=6, ...)` in the subagent spec | `city-researcher` | Subagents run their own loop, so they need their own limit |
| `agent.step_timeout = 60` / `180` | Both agents | Any single step (one model call or one round of tools) taking longer raises `TimeoutError` |
| `httpx.get(..., timeout=10)` | All 3 tools | A slow or dead API fails after 10s instead of hanging |

## What the user sees
- Limit hit → `error` event: *"This took too many steps, so I stopped. Try asking for less at once."*
  (The middleware's own message, "Model call limits exceeded: run limit (10/10)", is internal, so the stream replaces it.)
- Timeout → `error` event: *"This is taking too long, please try again"*.
- Tested by forcing `run_limit=1` on the real `/chat/stream` code: status, status, error, done.

## Key ideas
- `exit_behavior="end"` stops gracefully. `"error"` raises an exception. For `ToolCallLimitMiddleware`, the default
  `"continue"` only blocks that tool, and the model has to finish without it.
- `run_limit` = per request. `thread_limit` = across the whole conversation (a cost cap).
- Set limits **above normal usage**: a 3-city deep plan uses ~10 model calls, so 25 leaves room.
  Too low and real requests get cut off.
- `recursion_limit` (LangGraph steps) is the last-resort net. Middleware limits are the friendly, controlled stop.

## Also fix the cause
- Clear tool errors ("No place found, ask the user to check the spelling") stop blind retries.
- A prompt with stop conditions ("research each city once").
- LangSmith traces show loops (many identical tool calls). Alert on run length and cost.
- Evals can assert a max number of model calls, so a prompt change that causes loops fails before release.
