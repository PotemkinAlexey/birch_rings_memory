"""Paired session fixtures for echo validation testing."""

ECHO_PAIRS = [
    {
        "name": "false_resolution",
        "description": "Session 1 looked resonant but problem came back next day",
        "session_1": {
            "id": "s1_nginx",
            "messages": ["как настроить nginx", "попробую, спасибо"],
            "expected_r_before_echo": "resonant",
        },
        "session_2": {
            "messages": ["nginx всё равно не работает после перезапуска"],
            "expected_echo": True,
            "expected_r_after_echo": "toxic",  # session 1 retroactively downgraded
        },
    },
    {
        "name": "genuine_resolution",
        "description": "Session 1 resolved, session 2 is a new unrelated topic",
        "session_1": {
            "id": "s1_docker",
            "messages": ["как запустить docker контейнер", "работает, спасибо!"],
            "expected_r_before_echo": "resonant",
        },
        "session_2": {
            "messages": ["как написать unit тест на pytest"],
            "expected_echo": False,
            "expected_r_after_echo": "resonant",  # session 1 unchanged
        },
    },
    {
        "name": "stuck_then_returns",
        "description": "Session 1 was toxic, user returns with same problem",
        "session_1": {
            "id": "s1_sql",
            "messages": [
                "почему не работает sql запрос",
                "всё равно не работает",
                "не понимаю почему не работает",
            ],
            "expected_r_before_echo": "toxic",
        },
        "session_2": {
            "messages": ["опять проблема с sql запросом, тот же error"],
            "expected_echo": True,
            "expected_r_after_echo": "toxic",  # stays toxic, penalty stacks
        },
    },
]
