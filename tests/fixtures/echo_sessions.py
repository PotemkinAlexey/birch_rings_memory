"""Paired session fixtures for echo validation testing."""

ECHO_PAIRS = [
    {
        "name": "false_resolution",
        "description": "Session 1 looked resonant but problem came back next day",
        "session_1": {
            "id": "s1_nginx",
            "messages": ["how to configure nginx", "I'll try it, thanks"],
            "expected_r_before_echo": "resonant",
        },
        "session_2": {
            "messages": ["nginx still not working after restart"],
            "expected_echo": True,
            "expected_r_after_echo": "toxic",
        },
    },
    {
        "name": "genuine_resolution",
        "description": "Session 1 resolved, session 2 is a new unrelated topic",
        "session_1": {
            "id": "s1_docker",
            "messages": ["how to start a docker container", "works, thanks!"],
            "expected_r_before_echo": "resonant",
        },
        "session_2": {
            "messages": ["how to write a unit test with pytest"],
            "expected_echo": False,
            "expected_r_after_echo": "resonant",
        },
    },
    {
        "name": "stuck_then_returns",
        "description": "Session 1 was toxic, user returns with same problem",
        "session_1": {
            "id": "s1_sql",
            "messages": [
                "why is the sql query not working",
                "still not working",
                "I don't understand why it's not working",
            ],
            "expected_r_before_echo": "toxic",
        },
        "session_2": {
            "messages": ["having the sql query problem again, same error"],
            "expected_echo": True,
            "expected_r_after_echo": "toxic",
        },
    },
    {
        "name": "multi_topic_echo",
        "description": "Session 1 covered nginx AND postgres; user returns on postgres sub-topic",
        "session_1": {
            "id": "s1_multi",
            "messages": [
                "how to configure nginx as reverse proxy",
                "got it, now a question about postgres — why are queries slow",
                "nginx is working, thanks",
            ],
            "expected_r_before_echo": "resonant",
        },
        "session_2": {
            "messages": ["postgres queries are still slow, didn't help"],
            "expected_echo": True,
            "expected_r_after_echo": "toxic",
        },
    },
]
