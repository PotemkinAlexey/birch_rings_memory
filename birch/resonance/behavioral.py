"""Behavioral Decay metric — classifies session closure from message patterns."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ClosureSignal(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# Positive closure patterns — user got what they needed
_POSITIVE = re.compile(
    r"\b("
    r"работает|взлетело|заработало|всё\s*ок|всё\s*окей|спасибо|понял|разобрался"
    r"|got\s*it|works|perfect|done|thanks|thank\s*you|solved|fixed|great"
    r"|окей|ок|👍|✅|🎉"
    r")\b",
    re.IGNORECASE,
)

# Negative closure patterns — user stuck or frustrated
_NEGATIVE = re.compile(
    r"\b("
    r"не\s*работает|опять|снова|всё\s*равно|не\s*понимаю|что\s*за\s*ерунда"
    r"|почему\s*не|не\s*получается|ничего\s*не"
    r"|doesn'?t\s*work|still\s*not|again|error|failed|broken|wtf|not\s*working"
    r")\b",
    re.IGNORECASE,
)

# Unfinished session — abrupt stop mid-thought
_UNFINISHED = re.compile(r"\.\.\.|^(подожди|wait|стоп|stop)", re.IGNORECASE)


@dataclass
class BehavioralScore:
    signal: ClosureSignal
    score: float          # -1.0 … +1.0
    matched_pattern: str


_FOLLOWUP = re.compile(
    r"\?\s*$"
    r"|а\s+(как|что|где|почему|зачем)"
    r"|(but|and)\s+(how|what|why|where)",
    re.IGNORECASE,
)


def classify_final_message(message: str) -> BehavioralScore:
    """Score the final message of a session for closure signal."""
    text = message.strip()

    # Check negative first — "не работает" must not match positive "работает"
    if _NEGATIVE.search(text):
        return BehavioralScore(ClosureSignal.NEGATIVE, -0.8, "negative_pattern")

    if _UNFINISHED.search(text):
        return BehavioralScore(ClosureSignal.NEGATIVE, -0.5, "unfinished_pattern")

    if _POSITIVE.search(text) and not _FOLLOWUP.search(text):
        return BehavioralScore(ClosureSignal.POSITIVE, +1.0, "positive_pattern")

    # Neutral — no clear signal
    return BehavioralScore(ClosureSignal.NEUTRAL, 0.0, "none")


def score_session_decay(messages: list[str]) -> float:
    """
    Score full session behavioral decay.

    Weights: final message strongest, look at last 3 for trend.
    Returns float in [-1.0, +1.0].
    """
    if not messages:
        return 0.0

    tail = messages[-3:]
    scores = [classify_final_message(m).score for m in tail]

    # If all signals agree — amplify the consensus
    if len(set(s > 0 for s in scores if s != 0)) == 1:
        return scores[-1] * 1.2 if scores[-1] != 0 else 0.0

    # Mixed signals — final message gets 60% weight, previous two split 40%
    if len(scores) == 1:
        return scores[0]
    if len(scores) == 2:
        return scores[-1] * 0.6 + scores[-2] * 0.4
    return scores[-1] * 0.6 + scores[-2] * 0.25 + scores[-3] * 0.15
