"""Synthetic session fixtures for resonance detector testing."""

SESSIONS = [
    {
        "name": "success_en",
        "description": "User gets working code, confirms it works",
        "messages": [
            "how do I read a csv file in python",
            "what about handling missing values",
            "ok and how to filter rows by column value",
            "perfect, works great thanks",
        ],
        "expected_label": "resonant",
    },
    {
        "name": "success_ru",
        "description": "Пользователь решил задачу, подтвердил",
        "messages": [
            "как настроить nginx для проксирования на python приложение",
            "а как добавить ssl",
            "взлетело, спасибо",
        ],
        "expected_label": "resonant",
    },
    {
        "name": "stuck_loop",
        "description": "User keeps rephrasing the same question",
        "messages": [
            "why is my docker container not starting",
            "the docker container doesn't start",
            "my container still doesn't start what do I do",
            "container not starting again",
        ],
        "expected_label": "toxic",
    },
    {
        "name": "stuck_ru",
        "description": "Пользователь в петле, не получил ответ",
        "messages": [
            "почему не работает запрос к базе",
            "всё равно не работает",
            "не понимаю почему не работает",
        ],
        "expected_label": "toxic",
    },
    {
        "name": "neutral_short",
        "description": "Single question, no clear signal",
        "messages": [
            "what is the difference between list and tuple in python",
        ],
        "expected_label": "neutral",
    },
    {
        "name": "abrupt_end",
        "description": "Session cut off mid-thought",
        "messages": [
            "как сделать миграцию базы данных",
            "понял, а как откатить если что-то пошло не так",
            "подожди...",
        ],
        "expected_label": "toxic",
    },
    # Hard cases — no obvious positive/negative keywords, embeddings should help
    {
        "name": "hard_topic_shift",
        "description": "User moves from vague to concrete — productive narrowing, no keywords",
        "messages": [
            "у меня проблема с производительностью",
            "запросы медленные",
            "окей нашел — индекс отсутствовал на foreign key, добавил",
        ],
        "expected_label": "resonant",
    },
    {
        "name": "hard_circular",
        "description": "User rephrases same vague question without progress",
        "messages": [
            "как улучшить производительность системы",
            "что можно сделать для производительности",
            "какие есть способы повысить производительность",
        ],
        "expected_label": "toxic",
    },
]
