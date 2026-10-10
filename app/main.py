from dotenv import load_dotenv

# Copy .env into os.environ, where the OpenAI SDK and LangSmith look for their keys.
# Runs before the other imports, because app.itinerary creates the model on import.
load_dotenv()

from fastapi import FastAPI  # noqa: E402

from app import agent, itinerary  # noqa: E402

app = FastAPI(title="Tripora Agent")
app.include_router(itinerary.router)
app.include_router(agent.router)


@app.get("/health")
def health():
    return {"status": "ok"}
