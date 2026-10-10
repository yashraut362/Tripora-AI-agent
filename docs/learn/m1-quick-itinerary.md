# M1: Quick itinerary

`POST /itinerary` takes trip details and returns a day-by-day itinerary from **one LLM call**.
There's no agent and no tools yet: just the three LangChain building blocks that everything
later is made of: a **model**, a **prompt**, and a **chain**.

The whole feature is three small files. Read them in this order:

1. [`app/schemas.py`](../../app/schemas.py): the data shapes (input and output)
2. [`app/itinerary.py`](../../app/itinerary.py): model → prompt → chain → endpoint
3. [`app/main.py`](../../app/main.py): loads `.env`, creates the FastAPI app

## What happens on a request

```
POST /itinerary {"destination": "Goa", "days": 2, "interests": ["beaches", "food"]}
  │
  ├─ FastAPI parses the JSON into a TripRequest (wrong data → 422, your code never runs)
  │
  └─ chain.ainvoke(req.model_dump())
       │
       ├─ prompt   fills the {placeholders} → two messages:
       │             SystemMessage: "You are Tripora, an expert travel planner. …"
       │             HumanMessage:  "Plan a 2-day trip to Goa for 1 traveler(s). … Interests: ['beaches', 'food']."
       │
       ├─ model    sends those messages to OpenAI along with Itinerary's JSON schema
       │             and gets back JSON that matches the schema
       │
       └─ parser   turns that JSON into an Itinerary object (part of with_structured_output)
  │
  └─ FastAPI turns the Itinerary back into JSON → 200
```

## The concepts

**Model: `init_chat_model("gpt-5.4-mini")`.** A LangChain wrapper around an LLM API. It
works out the provider from the name ("gpt-…" → OpenAI) and reads `OPENAI_API_KEY` from
the environment. All chat models share one interface (`invoke`, `ainvoke`, `stream`,
`batch`), so switching to Claude or Gemini later is a one-word change.

**Messages.** An LLM doesn't take "a string". It takes a list of messages with roles:
system (rules and persona), human (the user), and ai (the model's earlier replies). M3's
chat memory is just keeping this list between requests.

**Prompt: `ChatPromptTemplate`.** A recipe that turns a dict of variables into that list
of messages. It just fills in placeholders; anything you pass ends up in the text as-is.
(Notice `['beaches', 'food']` and `Start date: None` above: the LLM copes fine with that.)

**Chain: `prompt | model`.** Prompts, models, and later tools, agents and graphs are all
**Runnables**: objects with `invoke` / `ainvoke` / `stream`. The `|` pipe connects them so
one's output is the next one's input, and the result is itself a Runnable. That's why one
`chain.ainvoke(...)` runs the whole pipeline. This style is called **LCEL**.

**Structured output: `model.with_structured_output(Itinerary)`.** It makes the model answer
with JSON matching your Pydantic class instead of free text. LangChain converts the class
to a JSON schema, OpenAI's "structured outputs" feature guarantees the reply matches it,
and LangChain parses it back into an `Itinerary`. Two things to notice:
- `Field(description=...)` text is sent to the model, so **descriptions are part of the prompt**.
- OpenAI wants every field required, so "optional" is written `float | None` with no default.
  The model then has to say `null` explicitly.

**FastAPI: one class, several jobs.** `TripRequest` validates input. `Itinerary` is the
LLM's output schema, the endpoint's return type (FastAPI uses the `-> Itinerary`
annotation to serialize the response), and the schema you see in `/docs`.

**Why `load_dotenv()` comes first in `main.py`.** The OpenAI SDK and LangSmith read their
keys from environment variables (`os.environ`). `load_dotenv()` copies `.env` into the
environment. It has to run before `app.itinerary` is imported, because that module creates
the model at import time. (That's also why the server won't start without a key.)

## Seeing it in LangSmith (Observability)

Put these in `.env` and LangChain traces every call automatically, with no code needed:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=tripora-agent
```

Open smith.langchain.com → project `tripora-agent` → click a run. You'll see the chain as
a tree: `ChatPromptTemplate` (the exact messages it built) → `ChatOpenAI` (the request,
including the JSON schema, plus the raw reply, tokens, cost, and latency). Get used to
reading this now. From M2 one request becomes many LLM and tool calls, and the trace is how
you'll understand what the agent did.

## Try it

```bash
cp .env.example .env     # add OPENAI_API_KEY (and LANGSMITH_API_KEY)
uv run uvicorn app.main:app --reload --port 8001
```

Open http://localhost:8001/docs → `POST /itinerary` → *Try it out*. Or:

```bash
curl -s -X POST localhost:8001/itinerary -H 'content-type: application/json' -d '{"destination": "Jaipur", "days": 2, "budget": "budget", "interests": ["forts", "street food"]}'
```

## Play with it

1. Remove `.with_structured_output(Itinerary)` and return `(await chain.ainvoke(...)).content`. What comes back now? (Change the return type to `str`.)
2. Delete the `description` on `DayPlan.title` and compare the titles you get.
3. Add `pace: Literal["relaxed", "packed"] = "relaxed"` to `TripRequest` and use `{pace}` in the prompt.
4. Try `model.with_structured_output(Itinerary, include_raw=True)` and look at what it returns (hint: token usage).
5. Switch `MODEL_NAME` in `.env` and compare speed and cost in LangSmith.
