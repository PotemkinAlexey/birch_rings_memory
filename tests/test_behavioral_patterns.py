"""Closure-signal pattern coverage — Russian and English, emojis, edge cases."""
import pytest

from birch.resonance.behavioral import (
    ClosureSignal,
    classify_final_message,
    score_session_decay,
)


@pytest.mark.parametrize("text", [
    "отлично!",
    "супер, спасибо",
    "класс",
    "сработало",
    "получилось наконец",
    "помогло, спасибо",
    "всё на месте",
    "всё хорошо",
    "всё нормально",
    "ура",
    "красота",
    "идеально",
    "красавчик",
    "понятно, разобрался",
    "понял, спасибо",
    "thanks!",
    "perfect",
    "done",
])
def test_positive_patterns(text):
    assert classify_final_message(text).signal is ClosureSignal.POSITIVE


@pytest.mark.parametrize("text", [
    "не помогло",
    "опять не пашет",
    "не сработало",
    "не получилось ничего",
    "сломалось",
    "вылетает каждый раз",
    "что за хрень",
    "фигня какая-то",
    "doesn't work",
    "still not working",
])
def test_negative_patterns(text):
    assert classify_final_message(text).signal is ClosureSignal.NEGATIVE


@pytest.mark.parametrize("text", [
    "погоди",
    "секунду",
    "минутку",
    "стой",
    "wait",
    "stop",
    "хм...",
])
def test_unfinished_patterns_score_negative(text):
    assert classify_final_message(text).signal is ClosureSignal.NEGATIVE


@pytest.mark.parametrize("text", ["🎉", "🔥", "👍 ", "spasibo ✅"])
def test_positive_emojis_standalone(text):
    assert classify_final_message(text).signal is ClosureSignal.POSITIVE


def test_negative_first_beats_positive_substring():
    """'не работает' contains 'работает' — negative check must run first."""
    assert (classify_final_message("не работает").signal
            is ClosureSignal.NEGATIVE)


def test_followup_suppresses_positive():
    """'понял' is positive, but 'понял, а как ...' is a follow-up question."""
    score = classify_final_message("понял, а как теперь дальше?")
    assert score.signal is not ClosureSignal.POSITIVE


def test_score_session_decay_russian_resonant():
    """A Russian wrap-up that ends positively scores resonant."""
    messages = [
        "вот, записал всё что мы построили сегодня",
        "формула теперь обучается на резонансе",
        "отлично, всё на месте",
    ]
    assert score_session_decay(messages) > 0.5


def test_score_session_decay_russian_toxic():
    """A Russian session that ends frustrated scores toxic."""
    messages = [
        "пробую починить",
        "опять не работает",
        "не помогло",
    ]
    assert score_session_decay(messages) < -0.5
