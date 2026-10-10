import os

from fastapi import APIRouter
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import Itinerary, TripRequest

# 1. The model: a LangChain wrapper around the OpenAI API.
model = init_chat_model(os.getenv("MODEL_NAME", "gpt-5.4-mini"))

# 2. The prompt: a template that becomes [SystemMessage, HumanMessage] once filled in.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Tripora, an expert travel planner. Group nearby sights on the same day "
               "and match activities and costs to the budget."),
    ("human", "Plan a {days}-day trip to {destination} for {travelers} traveler(s). "
              "Start date: {start_date}. Budget: {budget}. Interests: {interests}."),
])

# 3. The chain: prompt output is piped into the model, which must answer as an Itinerary.
chain = prompt | model.with_structured_output(Itinerary)

router = APIRouter()


@router.post("/itinerary")
async def create_itinerary(req: TripRequest) -> Itinerary:
    # req.model_dump() -> {"destination": ..., "days": ...} fills the {placeholders}.
    return await chain.ainvoke(req.model_dump())
