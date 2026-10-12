import os
from collections.abc import AsyncIterable

from deepagents import create_deep_agent
from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware, ToolCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from app.guardrails import travel_only
from app.schemas import ChatRequest
from app.tools import convert_currency, get_coordinates, get_weather_forecast

# M6: a deep agent for big, multi-city trips. Same agent loop as M2, plus:
#   - planning: write_todos keeps a checklist of steps (TodoListMiddleware)
#   - subagents: the `task` tool hands one city to city-researcher, which works in its own
#     clean context and returns only a short summary (so the main context stays small)
#   - a virtual filesystem: write_file / read_file / edit_file, stored in the agent's state
city_researcher = {
    "name": "city-researcher",
    "description": "Researches ONE city for a trip: weather forecast and the best sights, food and areas. "
                   "Give it the city with its country, the dates and the traveler's interests.",
    "system_prompt": "Research the one city you are given. Use your tools for the location and the weather. "
                     "Reply with a short summary (under 200 words): weather, must-see sights grouped by area, "
                     "local food, and getting around.",
    "tools": [get_coordinates, get_weather_forecast],
    # Subagents run their own loop, so they get their own limit.
    "middleware": [ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")],
}

deep_planner = create_deep_agent(
    model=f"openai:{os.getenv('MODEL_NAME', 'gpt-5.4-mini')}",
    tools=[convert_currency],
    system_prompt=(
        "You are Tripora's planner for big, multi-city trips. Work like this:\n"
        "1. write_todos: plan your steps.\n"
        "2. Delegate each city to city-researcher (one task per city, in parallel).\n"
        "3. If the traveler gives a budget in another currency, use convert_currency.\n"
        "4. Write the full plan to /trip_plan.md: route, days per city, a day-by-day plan, budget.\n"
        "5. Reply with a 2-line summary. For changes later, edit /trip_plan.md instead of starting over."
    ),
    subagents=[city_researcher],
    middleware=[
        travel_only,
        TodoListMiddleware(),
        # Loop protection: at most 25 model calls and 6 subagent tasks per request.
        ModelCallLimitMiddleware(run_limit=25, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="task", run_limit=6),
    ],
    checkpointer=InMemorySaver(),
)

# Hang protection: one step over 180s raises TimeoutError (a step can include several subagents).
deep_planner.step_timeout = 180

router = APIRouter()


@router.post("/deep-plan/stream", response_class=EventSourceResponse)
async def deep_plan_stream(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    # A deep plan takes minutes, so the client gets progress events (like a "Deep Research" UI):
    #   todos  = the agent's checklist, every time it changes
    #   status = what it's doing right now (e.g. which city is being researched)
    #   plan   = the finished /trip_plan.md
    config = {"configurable": {"thread_id": req.thread_id}}
    try:
        async for chunk in deep_planner.astream(
            {"messages": [{"role": "user", "content": req.message}]}, config,
            stream_mode="updates", version="v2",
        ):
            if chunk["data"].get("ModelCallLimitMiddleware.before_model"):
                yield ServerSentEvent(event="error", data="This plan took too many steps, so I stopped. Try fewer cities.")
                return
            if "model" not in chunk["data"]:
                continue
            for call in chunk["data"]["model"]["messages"][-1].tool_calls:
                if call["name"] == "write_todos":
                    yield ServerSentEvent(event="todos", data=call["args"]["todos"])
                elif call["name"] == "task":
                    yield ServerSentEvent(event="status", data=f"Researching: {call['args']['description'][:80]}…")
                elif call["name"] in ("write_file", "edit_file"):
                    yield ServerSentEvent(event="status", data="Writing your trip plan…")

        state = await deep_planner.aget_state(config)
        plan = state.values.get("files", {}).get("/trip_plan.md")
        if plan:
            yield ServerSentEvent(event="plan", data=plan["content"])
        yield ServerSentEvent(event="summary", data=state.values["messages"][-1].text)
    except TimeoutError:
        yield ServerSentEvent(event="error", data="This is taking too long, please try again")
        raise
    except Exception:
        yield ServerSentEvent(event="error", data="Something went wrong, please try again")
        raise

    yield ServerSentEvent(event="done", data="")
