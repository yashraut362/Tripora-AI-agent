# M5: Is the agent any good? (LangSmith Evaluation)

Code: [`evals/agent_eval.py`](../../evals/agent_eval.py), one script with no new endpoints.

```bash
uv run python -m evals.agent_eval                              # current model
MODEL_NAME=gpt-5.4-nano uv run python -m evals.agent_eval      # same tests, another model
```

## Why

- An LLM app has no unit tests for "is the answer good?". Every prompt tweak, model swap or new tool
  can quietly break something that used to work.
- Evals are **tests for LLM behaviour**: a fixed set of questions, run the real agent, score each answer automatically.
- Production teams run them before shipping a prompt or model change (often in CI), and compare against the last run.

## The 4 pieces

| Piece | In our script | What it is |
|---|---|---|
| **Dataset** | `tripora-agent-questions`, 6 examples | Inputs (`message`) + reference outputs (expected `tools`, expected `behavior`). Stored in LangSmith, created once |
| **Target** | `run_agent()` | Runs the **real** agent on one input (fresh `thread_id` each time) and returns what it did: tool calls, answer, paused or not |
| **Evaluators** | `right_tools`, `follows_expected_behavior` | Score one run against its example |
| **Experiment** | `client.aevaluate(...)` | Every example → target → every evaluator. Saved in LangSmith as one experiment |

## Two kinds of evaluator

| | `right_tools` (code) | `follows_expected_behavior` (LLM-as-judge) |
|---|---|---|
| Checks | Called exactly the expected tools? | Did the answer do what `behavior` describes? |
| How | `set(called) == set(expected)` | A second model grades it, with `with_structured_output(Grade)` returning `passed` + `reasoning` |
| Cost / speed | Free, instant, deterministic | One LLM call, can be wrong itself |
| Use for | Anything code can check | Quality, tone, "did it actually answer?" |

Evaluator functions just declare the arguments they need: `inputs`, `outputs`, `reference_outputs`.
They return a `bool`, or `{"score": ..., "comment": ...}` (the comment shows up in LangSmith).

## What the evals found (real runs)

| Run | Result | Found | Fix |
|---|---|---|---|
| 1 | 5/6 | "Plan a trip to Bali" → the agent **guessed 5 days** and called `create_itinerary` | Prompt: *"If they haven't said how many days, ask them first instead of guessing."* |
| 2 | 5/6 | "Weather in Goa" → it looked up **Genoa, Italy**. `right_tools` passed (right tools!), only the **judge** caught it | `get_coordinates` docstring: *"Always include the country (e.g. "Goa, India")"*. The geocoder returns Genoa for a bare "Goa" |
| 3 | 6/6 | Everything passes | |
| nano | 6/6 | The cheaper model passes too | A candidate for cost savings, but check latency, cost and more examples first |

Lessons:
- **Evals find real bugs.** Neither bug showed up in our manual curl tests.
- **The Goa bug was flaky.** The model sometimes wrote "Goa, India" and sometimes "Goa". One manual test
  can't tell you that; a dataset run repeatedly can (`num_repetitions=3` in `aevaluate`).
- **Code evaluators aren't enough.** "Right tools, wrong place" passed the code check.
- The fixes were in a **prompt** and a **docstring**, not code. In agents, prompts and tool descriptions are the code.

## LangSmith UI

- **Datasets & Experiments → tripora-agent-questions**: every experiment run against this dataset.
- Select 2+ experiments and choose **Compare**: rows per example, scores side by side, red/green diffs,
  and latency and token cost per run. That's how you answer "is the new prompt/model better?"
- Click any row to see the **full trace** of that agent run: every model call and tool call.

## Changes outside `evals/`

- `app/agent.py`: one prompt sentence (ask for days).
- `app/tools.py`: one docstring line (include the country).

Both are fixes the evals found.

## Production notes

- The dataset grows over time: every bug from real users becomes a new example, so it never comes back.
- In CI: run on every prompt or model PR, and fail if scores drop below the last baseline.
- LLM-as-judge needs its own checks. Read its `comment`s, and use a strong, fixed judge model (`JUDGE_MODEL`)
  so the judge doesn't change between experiments.
- 6 examples is a demo. Real datasets are tens to hundreds of examples, covering edge cases.

## Play with it

1. Add an example: `"Convert 50 GBP to JPY and tell me if it'll be sunny in Tokyo tomorrow"`. To add
   examples to an existing dataset, use `client.create_examples(dataset_name=DATASET, examples=[...])`
   once (or the LangSmith UI).
2. Remove the "ask for days" sentence from the prompt, re-run, and compare with the last experiment: Bali turns red.
3. Add `num_repetitions=3` to `aevaluate` and check whether any example passes only sometimes.
4. Write a third evaluator `asks_question(outputs) -> bool` that checks whether the answer ends with `?`. Where is it useful?
5. In the LangSmith UI, open the Genoa failure from run 2 and find the `get_coordinates` call in its trace.
