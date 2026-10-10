import os

from fastapi import APIRouter
from langchain.agents import create_agent

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
)

router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest):
    result = await agent.ainvoke({"messages": [{"role": "user", "content": req.message}]})
    messages = result["messages"]
    return {
        "answer": messages[-1].content,
        # Every step of the loop, so you can see what the agent did:
        # human -> ai (tool_calls) -> tool (result) -> ... -> ai (final answer)
        "steps": [m.model_dump(include={"type", "content", "tool_calls", "name"}) for m in messages],
    }
