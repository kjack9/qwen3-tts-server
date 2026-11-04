"""Plan text chunks using a solo speaking-rate estimate, never queue wall time."""
from __future__ import annotations

import re


_WORDS = re.compile(r"\S+")
_SENTENCE_END = re.compile(r'''[.!?。！？]["'”’»\)\]\}]*$''')


def duration_chunks(text, target_seconds=30, max_seconds=60, words_per_second=3.13):
    """Finish the next sentence after target; force after a whole word at max.

    Boundaries refer to estimated spoken duration. Exact word timestamps do not
    exist before synthesis. Keep every character of the trimmed source text,
    and never clip generated audio as a substitute for splitting source text.
    """
    if not 0 < target_seconds <= max_seconds or words_per_second <= 0:
        raise ValueError("Expected 0 < target <= maximum and a positive speech rate")
    text = text.strip()
    matches = list(_WORDS.finditer(text))
    if not matches:
        return []
    if len(matches) / words_per_second <= target_seconds:
        return [text]
    chunks = []
    start = 0
    words = 0
    for match in matches:
        words += 1
        duration = words / words_per_second
        if duration >= max_seconds or (
            duration >= target_seconds and _SENTENCE_END.search(match.group())
        ):
            chunks.append(text[start:match.end()].strip())
            start = match.end()
            words = 0
    if text[start:].strip():
        chunks.append(text[start:].strip())
    return chunks
