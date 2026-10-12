import os
from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.guardrails import travel_only
from app.schemas import ChatRequest
from app.tools import convert_currency, create_itinerary, get_coordinates, get_weather_forecast

# An agent = a model + tools + a loop:
#   1. send the messages (and the tool descriptions) to the model
#   2. if the model asks for tool calls -> run them, append the results, go to 1
#   3. if it answers with plain text -> stop
agent = create_agent(
    model=f"openai:{os.getenv('MODEL_NAME', 'gpt-5.4-mini')}",
    tools=[get_coordinates, get_weather_forecast, convert_currency, create_itinerary],
    system_prompt=(
        "You are Tripora, a friendly travel assistant. Use your tools for live facts "
        "(weather, exchange rates) instead of guessing, and keep answers short. "
        "When the traveler wants a full trip plan, call create_itinerary right away: the app "
        "shows them the brief to confirm, so don't ask for confirmation in chat. "
        "If they haven't said how many days, ask them first instead of guessing."
    ),
    # M4: after each model turn, if the model asked for create_itinerary, pause BEFORE the tool
    # runs (interrupt() underneath) until the traveler approves or rejects it.
    middleware=[
        travel_only,  # guardrail: off-topic messages get a fixed reply, the agent doesn't run
        HumanInTheLoopMiddleware(interrupt_on={"create_itinerary": {"allowed_decisions": ["approve", "reject"]}})
    ],
    # M3 memory: after every step the agent's state (its message list) is saved here,
    # keyed by thread_id. The next request on the same thread starts from that state.
    # InMemorySaver lives in RAM, so restarting the server forgets everything.
    checkpointer=InMemorySaver(),
)

router = APIRouter()


def thread_config(req: ChatRequest) -> dict:
    return {"configurable": {"thread_id": req.thread_id}}


# M4: if a run is paused waiting for approval on this thread, this request answers it
# (Command(resume=...) continues the paused run). Otherwise it's a new message as in M3.
async def agent_input(req: ChatRequest):
    state = await agent.aget_state(thread_config(req))
    if state.interrupts:
        # Approve button -> the tool runs. A typed message -> reject with that message as the
        # reason, so the tool doesn't run and the model retries with the change.
        if req.decision == "approve":
            decision = {"type": "approve"}
        else:
            decision = {"type": "reject", "message": req.message}
        return Command(resume={"decisions": [decision]})
    return {"messages": [{"role": "user", "content": req.message}]}


@router.post("/chat")
async def chat(req: ChatRequest):
    # Only the NEW message is sent: the checkpointer adds the earlier ones back in.
    result = await agent.ainvoke(
        await agent_input(req), thread_config(req), version="v2"
    )
    messages = result.value["messages"]
    return {
        "answer": messages[-1].content,
        # M4: the trip brief when the run paused for approval, else None.
        "approval_required": result.interrupts[0].value["action_requests"][0]["args"] if result.interrupts else None,
        # The whole conversation on this thread so far, including earlier requests:
        # human -> ai (tool_calls) -> tool (result) -> ... -> ai (final answer)
        "steps": [m.model_dump(include={"type", "content", "tool_calls", "name"}) for m in messages],
    }


# What the user sees while a tool runs. Raw tool args/results stay server-side
# (in LangSmith traces), and the client only gets a friendly status line.
TOOL_STATUS = {
    "get_coordinates": "Looking up {place}…",
    "get_weather_forecast": "Checking the weather forecast…",
    "convert_currency": "Converting {from_currency} to {to_currency}…",
    "create_itinerary": "Preparing a brief for your {days}-day trip to {destination}…",
}


@router.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    # Same agent and same memory as /chat, but events are sent while it works.
    # stream_mode "messages" = LLM tokens as they're generated
    # stream_mode "updates"  = each node's output once it finishes (used to spot tool calls)
    try:
        async for chunk in agent.astream(
            await agent_input(req),
            thread_config(req),
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                if metadata["langgraph_node"] == "model" and token.text:
                    yield ServerSentEvent(event="token", data=token.text)

            elif chunk["type"] == "updates" and "model" in chunk["data"]:
                for call in chunk["data"]["model"]["messages"][-1].tool_calls:
                    status = TOOL_STATUS.get(call["name"], "Working on it…").format(**call["args"])
                    yield ServerSentEvent(event="status", data=status)

            elif chunk["type"] == "updates" and chunk["data"].get("travel_only.before_agent"):
                # The guardrail blocked the message: its fixed reply isn't streamed as tokens.
                yield ServerSentEvent(event="token", data=chunk["data"]["travel_only.before_agent"]["messages"][-1].text)

        # M4: if the run paused for approval, send the trip brief for the approval card.
        state = await agent.aget_state(thread_config(req))
        if state.interrupts:
            yield ServerSentEvent(event="approval_required", data=state.interrupts[0].value["action_requests"][0]["args"])
    except Exception:
        # The stream has already started (status 200 was sent), so an HTTP error code is
        # no longer possible. Errors have to be sent as an event too.
        yield ServerSentEvent(event="error", data="Something went wrong, please try again")
        raise

    yield ServerSentEvent(event="done", data="")
