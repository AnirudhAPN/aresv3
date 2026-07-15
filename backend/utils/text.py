"""Small, generic text-cleanup helpers shared across agents."""
from __future__ import annotations

import re

# Matches a leading ordinal marker an LLM might embed despite the numbering
# already being applied by the UI/export layer — "1.", "2)", "Step 3:",
# "3 -", etc. — so a single list never ends up double-numbered. Requires at
# least one space after the marker so genuine content starting with a
# number (e.g. "5-fold cross-validation", "24-hour monitoring window") is
# never mistaken for a step prefix and stripped.
_LEADING_ORDINAL_RE = re.compile(r"^\s*(?:step\s*)?\d+\s*[\.\):-]\s+", re.IGNORECASE)


def strip_leading_ordinal(text: str) -> str:
    """Remove a leading step/ordinal number the model may have added,
    since the UI and exports apply their own sequential numbering — if the
    model also numbers its strings, steps render doubled ("1. 1. ...") or,
    worse, out of sync if the model's own numbering isn't sequential."""
    return _LEADING_ORDINAL_RE.sub("", text, count=1).strip()
