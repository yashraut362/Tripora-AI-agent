import httpx
from langchain.tools import tool

from app.itinerary import chain
from app.schemas import TripRequest

# A tool is just a Python function. @tool reads its name, type hints and docstring
# and turns them into a description the LLM sees, so the docstring is the
# instruction that tells the model WHEN and HOW to use it.


@tool
def get_coordinates(place: str) -> dict:
    """Look up a city or place and return its latitude, longitude, country and timezone."""
    res = httpx.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 1})
    results = res.json().get("results")
    if not results:
        return {"error": f"No place found called {place!r}"}
    p = results[0]
    return {k: p.get(k) for k in ("name", "country", "latitude", "longitude", "timezone")}


@tool
def get_weather_forecast(latitude: float, longitude: float, days: int = 7) -> dict:
    """Daily weather forecast for the next 1-16 days at a location: max/min temperature (°C)
    and chance of rain (%). Needs coordinates, so call get_coordinates first."""
    res = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
        },
    )
    return res.json()["daily"]


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between currencies at today's exchange rate. Use ISO codes like USD, EUR, INR."""
    res = httpx.get(
        "https://api.frankfurter.dev/v1/latest",
        params={"base": from_currency.upper(), "symbols": to_currency.upper()},
    )
    rate = res.json()["rates"][to_currency.upper()]
    return {"rate": rate, "converted": round(amount * rate, 2), "currency": to_currency.upper()}


# M4: building a full itinerary is a big, slow LLM call, so it only runs after the traveler
# approves the brief: HumanInTheLoopMiddleware (app/agent.py) pauses the agent before this tool.
@tool(args_schema=TripRequest)
async def create_itinerary(**trip) -> dict:
    """Build a full day-by-day itinerary. Call it once you know the destination and number of days."""
    itinerary = await chain.ainvoke(TripRequest(**trip).model_dump(mode="json"))
    return itinerary.model_dump()
