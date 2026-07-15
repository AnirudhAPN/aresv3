"""Prompt template for the Methodology Agent.

Forces the model to select from a curated whitelist first, and to clearly
separate any additional ideas into `unverified_suggestions`.
"""
import json

METHODOLOGY_SCHEMA_HINT = """{
  "datasets": [{"name": "...", "url": "...", "description": "...", "verified": true}],
  "algorithms": [{"name": "...", "url": "...", "description": "...", "verified": true}],
  "evaluation_metrics": ["..."],
  "implementation_steps": ["..."],
  "unverified_suggestions": ["..."]
}"""


def build_methodology_prompt(
    topic: str,
    research_questions: list[str],
    known_datasets: list[dict],
    known_algorithms: list[dict],
) -> str:
    # Only show the LLM the fields it needs to choose from — extra bookkeeping
    # fields (e.g. kaggle_url, resolved separately in code after generation)
    # would just waste tokens and risk it echoing them into its own output.
    prompt_fields = ("name", "url", "description")
    datasets_block = json.dumps([{k: d.get(k) for k in prompt_fields} for d in known_datasets], indent=2)
    algorithms_block = json.dumps([{k: a.get(k) for k in prompt_fields} for a in known_algorithms], indent=2)
    return f"""You are the Methodology Agent in a multi-agent research assistant.

Research topic: "{topic}"
Research questions: {json.dumps(research_questions)}

You have access to a CURATED WHITELIST of real, verified datasets and
algorithms below. You must FIRST attempt to select and justify choices from
this whitelist — set verified=true and copy the exact name/url from the
whitelist for anything you choose from it.

Whitelisted datasets:
{datasets_block}

Whitelisted algorithms:
{algorithms_block}

RULES:
- Only mark verified=true if the name/url came directly from the whitelist
  above. Do not mark verified=true for anything you invented.
- If nothing on the whitelist reasonably fits the topic, you MAY propose
  additional ideas, but they must go in `unverified_suggestions` as plain
  strings, clearly separate from `datasets`/`algorithms`, and must NOT be
  marked verified.
- Recommend 2-5 datasets and 2-5 algorithms (whitelist-first).
- Recommend 3-6 evaluation_metrics appropriate to the topic.
- Recommend 4-8 concrete implementation_steps in a logical order. Each step must be a plain
  sentence with NO leading number, "Step N", or bullet — the array order itself IS the order;
  the UI numbers them automatically, so a number embedded in the text would render doubled.

Respond with ONLY valid JSON matching this exact shape, no prose:
{METHODOLOGY_SCHEMA_HINT}
"""
