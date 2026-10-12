# Extra: Guardrails (stay on topic)

Code: [`app/guardrails.py`](../../app/guardrails.py), added to both agents (`/chat` and `/deep-plan`).

## What
- An **input guardrail**: before the agent runs, a small model (`gpt-5.4-nano`) checks the message.
- Clearly off-topic (code, homework, sports, poems, "ignore your instructions…") → a fixed reply:
  *"I'm Tripora, your travel assistant…"*. The agent, its tools and the big model never run.
- Everything else (travel, greetings, thanks, follow-ups like "and in Kyoto?") → the agent runs as usual.

## How
- Middleware hook `@before_agent(can_jump_to=["end"])` runs once, before the agent loop.
  - Return `None` → continue.
  - Return `{"messages": [AIMessage(reply)], "jump_to": "end"}` → add the reply and skip to the end.
- The classifier sees the last 3 messages, so short follow-ups are judged in context.
- `with_structured_output(TopicCheck)` gives a typed `clearly_off_topic: bool`.
- `/chat/stream`: the fixed reply doesn't come from the model, so it's sent as a `token` event from the
  `travel_only.before_agent` update.

## Why code and not just the prompt
- "Only talk about travel" in the system prompt is easy to talk around ("ignore your instructions…").
- A separate check the user can't talk to is harder to bypass, cheaper (no tools, no big model), and gives the same reply every time.
- In production, use both: a prompt line **and** a guardrail (defense in depth).

## Lesson: ask the right question
- First version: "Is this about travel?" → **false positives**. It blocked "Hi!", "thanks!", even "200 USD in EUR?" (nano and mini both).
- Fixed version: "Is this **clearly** off-topic? If unsure, false." → 11/11 on our test cases, with both models.
- Rule: wrongly blocking a real user is worse than answering one stray question. Tune for **few false positives**.

## Evals (M5) protect it
- 2 guardrail examples in the dataset (a coding request, a prompt injection). The full eval is 8/8.
- Any change to the guardrail prompt or model → re-run the evals to catch new false positives.

## Production notes
- Extra latency: one small LLM call per message. You can run it in parallel with the agent and cancel the agent if it's blocked.
- Other guardrail types: PII redaction (`PIIMiddleware`), output checks (`@after_agent`), call limits
  (`ModelCallLimitMiddleware`), and approval for risky tools (M4).
