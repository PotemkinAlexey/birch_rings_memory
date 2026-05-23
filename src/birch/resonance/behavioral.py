"""Behavioral Decay metric — classifies session closure from message patterns."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ClosureSignal(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# Positive closure patterns — user got what they needed.
# Word alternation needs \b on both sides; emojis are non-word characters,
# so they match outside the \b group.
_POSITIVE = re.compile(
    r"(?:\b("
    # Russian — resolution
    r"работает|сработало|получилось|помогло|взлетело|заработало"
    # Russian — affirmations
    r"|всё\s*ок|всё\s*окей|всё\s*хорошо|всё\s*нормально|всё\s*на\s*месте"
    r"|всё\s*ясно|всё\s*получилось"
    # Russian — acknowledgements
    r"|спасибо|понял|поняла|понятно|разобрался|разобралась"
    # Russian — enthusiasm
    r"|отлично|супер|круто|класс|огонь|красота|ура|идеально|красавчик|чудно"
    r"|окей|ок"
    # English
    r"|got\s*it|found\s*it|figured\s*it\s*out|works|perfect|done|thanks|thank\s*you"
    r"|solved|fixed|great|nice|awesome|brilliant"
    r")\b)"
    # Emojis (no word boundary — they are non-word chars).
    r"|👍|✅|🎉|🔥|🚀|💯",
    re.IGNORECASE,
)

# Negative closure patterns — user stuck or frustrated.
_NEGATIVE = re.compile(
    r"\b("
    # Russian — explicit non-work
    r"не\s*работает|не\s*пашет|не\s*сработало"
    r"|не\s*получается|не\s*получилось|не\s*помогло|не\s*понимаю"
    # Russian — frustration
    r"|опять|снова|всё\s*равно|ничего\s*не"
    r"|сломалось|вылетает|крашится|падает"
    r"|что\s*за\s*ерунда|что\s*за\s*хрень|фигня|хрень"
    r"|почему\s*не"
    # English
    r"|doesn'?t\s*work|still\s*not|again|error|failed|broken|wtf|not\s*working|stuck|crashes"
    r")\b",
    re.IGNORECASE,
)

# Unfinished session — abrupt stop mid-thought.
_UNFINISHED = re.compile(
    r"\.\.\.|^(подожди|погоди|секунду|минут(?:у|ку)|стой|wait|hold\s*on|стоп|stop)",
    re.IGNORECASE,
)


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
