import re
from typing import List, Optional


_ALLOWED_CHARS_RE = re.compile(r"^[0-9,\-\s]+$")
_EXPLICIT_LIST_RE = re.compile(r"^\s*\d+\s*(?:[,-]\s*\d+\s*)*$")


def parse_layer_using_strategy(strategy: Optional[str]) -> Optional[List[int]]:
    """Parse `layer_using_strategy` as an explicit list of vision `hidden_states` indices.

    This is intentionally *not* a range parser:
      - "18-23" is interpreted as [18, 23], not 18..23.

    Supported examples:
      - "20" -> [20]
      - "3-18-23" -> [3, 18, 23]
      - "3,18,23" -> [3, 18, 23]
      - "3 - 18 - 23" -> [3, 18, 23]

    Only strings consisting purely of integers separated by '-' or ',' are parsed.
    Anything else (e.g. "former", "latter", "all", "plain") returns None so callers
    can fall back to legacy preset behavior.

    Note:
      - These numbers are *hidden_states indices* returned by the vision model.
        Typically, hidden_states[0] is the embeddings output, and hidden_states[1:]
        are transformer block outputs.
    """
    if strategy is None:
        return None

    text = str(strategy).strip()
    if not text:
        return None

    if not _ALLOWED_CHARS_RE.fullmatch(text):
        return None

    if not _EXPLICIT_LIST_RE.fullmatch(text):
        raise ValueError(
            f"Invalid explicit layer_using_strategy format: {strategy!r}. "
            "Use digits separated by '-' or ',', e.g. '20' or '3-18-23'."
        )

    parts = re.split(r"[,-]", text)
    return [int(part.strip()) for part in parts]
