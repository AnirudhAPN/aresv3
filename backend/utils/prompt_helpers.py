"""Shared prompt-augmentation helpers used by every agent for the two
distinct retry policies (structural parse-retry and ACS quality-retry)."""


def build_retry_prompt(original_prompt: str, previous_output: str, weak_dimension: str) -> str:
    return f"""{original_prompt}

--- RETRY CONTEXT (quality) ---
Your previous response scored low on: {weak_dimension}.
Previous response:
{previous_output}

Produce an improved response addressing the "{weak_dimension}" weakness
specifically. Respond with ONLY valid JSON in the same schema, no prose.
"""


def build_structural_retry_prompt(original_prompt: str, invalid_output: str, validation_error: str) -> str:
    return f"""{original_prompt}

--- YOUR PREVIOUS RESPONSE FAILED TO PARSE ---
Previous (invalid) response:
{invalid_output}

Validation error:
{validation_error}

Respond again with ONLY valid JSON matching the required schema exactly.
No prose before or after. No markdown code fences.
"""
