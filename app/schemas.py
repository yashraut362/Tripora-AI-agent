from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


# Input: what the client POSTs. FastAPI validates it and returns 422 if it's wrong.
class TripRequest(BaseModel):
    destination: str
    days: int = Field(ge=1, le=14)
    start_date: date | None = None
    budget: Literal["budget", "mid", "luxury"] = "mid"
    interests: list[str] = []
    travelers: int = 1


# Output: what the LLM must return. LangChain turns these classes into a JSON schema
# and sends it to OpenAI, so each `description` is an instruction to the model.
# (OpenAI needs every field required, so "optional" is `X | None` with no default.)
class Activity(BaseModel):
    time_of_day: Literal["morning", "afternoon", "evening"]
    name: str
    description: str = Field(description="One or two sentences on what to do there")
    est_cost_usd: float | None = Field(description="Cost per person in USD, null if free")


class DayPlan(BaseModel):
    day: int
    title: str = Field(description="Short theme for the day")
    activities: list[Activity]


class Itinerary(BaseModel):
    destination: str
    summary: str
    days: list[DayPlan] = Field(description="Exactly one entry per trip day")
    tips: list[str]


class ChatRequest(BaseModel):
    message: str
