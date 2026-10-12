import os

from langchain.agents.middleware import AgentState, before_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel

# Guardrail: an input check that runs BEFORE the agent. A small, cheap model classifies the
# message; off-topic requests get a fixed reply and the agent (tools, big model) never runs.


class TopicCheck(BaseModel):
    clearly_off_topic: bool


classifier = init_chat_model(os.getenv("GUARDRAIL_MODEL", "gpt-5.4-nano")).with_structured_output(TopicCheck)

# Block only what is CLEARLY off-topic: wrongly blocking a real traveler is worse than
# answering one stray question.
CLASSIFIER_PROMPT = (
    "You check messages sent to Tripora, a travel assistant (trips, itineraries, weather, currency, "
    "packing, visas, local food and culture). Set clearly_off_topic=true ONLY if the LAST user message "
    "asks for something with no link to travel, like writing code, homework, math, sports results or "
    "general writing tasks. Greetings, thanks, questions about Tripora, currency conversion and short "
    "follow-ups ('and there?', 'yes', 'make it 2 days') are NOT off-topic. If unsure, answer false."
)

OFF_TOPIC_REPLY = (
    "I'm Tripora, your travel assistant, so I can only help with trips: planning itineraries, "
    "weather at your destination, currency and travel tips. Where would you like to go?"
)


@before_agent(can_jump_to=["end"])
async def travel_only(state: AgentState, runtime: Runtime) -> dict | None:
    recent = "\n".join(f"{m.type}: {m.text}" for m in state["messages"][-3:])
    check = await classifier.ainvoke([("system", CLASSIFIER_PROMPT), ("human", recent)])
    if not check.clearly_off_topic:
        return None  # let the agent run
    # Add the fixed reply to the conversation and skip straight to the end.
    return {"messages": [AIMessage(OFF_TOPIC_REPLY)], "jump_to": "end"}
