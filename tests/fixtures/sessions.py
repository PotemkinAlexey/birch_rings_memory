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
        "name": "success_short",
        "description": "User solved problem quickly, confirmed",
        "messages": [
            "how to configure nginx as reverse proxy for python app",
            "how to add ssl",
            "works, thanks",
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
        "name": "stuck_en",
        "description": "User in a loop, didn't get an answer",
        "messages": [
            "why is the database query not working",
            "still not working",
            "I don't understand why it's not working",
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
            "how to run a database migration",
            "got it, and how to roll back if something goes wrong",
            "wait...",
        ],
        "expected_label": "toxic",
    },
    # Hard cases — no obvious positive/negative keywords, embeddings should help
    {
        "name": "hard_topic_shift",
        "description": "User moves from vague to concrete — productive narrowing, no keywords",
        "messages": [
            "I have a performance problem",
            "queries are slow",
            "ok found it — missing index on foreign key, added it",
        ],
        "expected_label": "resonant",
    },
    {
        "name": "hard_circular",
        "description": "User rephrases same vague question without progress",
        "messages": [
            "how to improve system performance",
            "what can be done about performance",
            "what are the ways to increase performance",
        ],
        "expected_label": "toxic",
    },
]
