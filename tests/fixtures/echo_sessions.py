"""Paired session fixtures for echo validation testing.

NOTE ON THE POST-ECHO EXPECTATIONS (theory change, May 2026):
The raw ``EchoStore.detect_echo`` penalty is now proportional to confidence
that the revisit signals failure — ``penalty = base * clamp(1 - prior_r, 0, 1)``
— and the old forced toxic floor (``min(-0.2, …)``) is gone. So a revisit to a
*strongly resonant* past session no longer slams it to "toxic". Measured with
nomic-embed-text these resonant priors close at r≈0.71; the confidence-scaled
penalty (≈-0.23) merely dents them to r≈0.48 — still "resonant". Only the
*toxic-prior* case (r≈-0.48) still lands toxic, at full penalty.

That is deliberate: the raw primitive is NOT supposed to tell "false closure"
apart from "continued use" — both leave a resonant prior resonant here. The
streaming path (session_open → session_close) is what catches the real
"false_resolution" case, by gating the penalty on the *current* session's
outcome at close — against actual evidence, not a guess at open. These pairs
run ONLY under real embeddings (``BIRCH_EMBED_PROVIDER=ollama pytest
tests/test_echo.py``); the deferred/gated behaviour has its own mock-runnable
coverage in test_memory_store.py.
"""

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
            # Was "toxic" under the flat -0.8 + toxic-floor scheme. The
            # confidence-scaled penalty (≈-0.23 on a resonant prior) no longer
            # nukes it; measured r≈0.48 stays "resonant". The outcome gate at
            # close is what catches the genuinely-failed case.
            "expected_r_after_echo": "resonant",
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
            # Resonant prior → confidence-scaled penalty, no toxic floor;
            # dented to r≈0.48, still "resonant".
            "expected_r_after_echo": "resonant",
        },
    },
]
