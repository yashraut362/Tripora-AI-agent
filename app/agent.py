import os
from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from app.schemas import ChatRequest
from app.tools import convert_currency, get_coordinates, get_weather_forecast

# An agent = a model + tools + a loop:
#   1. send the messages (and the tool descriptions) to the model
#   2. if the model asks for tool calls -> run them, append the results, go to 1
#   3. if it answers with plain text -> stop
agent = create_agent(
    model=f"openai:{os.getenv('MODEL_NAME', 'gpt-5.4-mini')}",
    tools=[get_coordinates, get_weather_forecast, convert_currency],
    system_prompt=(
        "You are Tripora, a friendly travel assistant. Use your tools for live facts "
        "(weather, exchange rates) instead of guessing, and keep answers short."
    ),
    # M3 memory: after every step the agent's state (its message list) is saved here,
    # keyed by thread_id. The next request on the same thread starts from that state.
    # InMemorySaver lives in RAM, so restarting the server forgets everything.
    checkpointer=InMemorySaver(),
)

router = APIRouter()


def thread_config(req: ChatRequest) -> dict:
    return {"configurable": {"thread_id": req.thread_id}}


@router.post("/chat")
async def chat(req: ChatRequest):
    # Only the NEW message is sent: the checkpointer adds the earlier ones back in.
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]}, thread_config(req)
    )
    messages = result["messages"]
    return {
        "answer": messages[-1].content,
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
}


@router.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    # Same agent and same memory as /chat, but events are sent while it works.
    # stream_mode "messages" = LLM tokens as they're generated
    # stream_mode "updates"  = each node's output once it finishes (used to spot tool calls)
    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": req.message}]},
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
    except Exception:
        # The stream has already started (status 200 was sent), so an HTTP error code is
        # no longer possible. Errors have to be sent as an event too.
        yield ServerSentEvent(event="error", data="Something went wrong, please try again")
        raise

    yield ServerSentEvent(event="done", data="")
