"""Two failure-mode contracts that survived the previous review rounds:

  1. collapse_singularity mutates self._hole in-memory BEFORE the
     storage writes. If a storage write raises mid-transaction, the
     SQLite txn rolls back cleanly — but the in-memory _hole stays
     mutated, and data_version doesn't bump on rollback so _sync()
     can't detect the desync. Fix: on any exception inside the
     collapse txn, force a full _reload() to re-anchor every cache
     to disk truth before propagating.

  2. _post in the Ollama embedding client had zero retry budget for
     transport-level failures. A brief network blip or an Ollama
     restart turned into an immediate embedding_provider_unavailable
     at the MCP boundary. Fix: 2-attempt loop with 200ms backoff for
     URLError / socket.timeout / ConnectionError / OSError. HTTPError
     is NEVER retried — 4xx and 5xx are deterministic on the same
     input, retrying just doubles latency for nothing.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from birch.memory_store import MemoryStore
from birch.resonance.embeddings import EmbeddingError, _post

# --- I1: collapse rollback re-syncs in-memory state -------------------


def test_collapse_storage_failure_resyncs_hole(tmp_path):
    """If save_meta_facts raises mid-collapse, the in-memory _hole
    must be re-anchored to disk truth — otherwise the in-memory view
    shows phantom-collapsed MetaFacts that no longer exist on disk."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Seed three near-identical facts and force them into singularity.
    f1 = mem.add_fact("svc", "uses", "Postgres")
    f2 = mem.add_fact("svc", "uses", "Postgres v2")
    f3 = mem.add_fact("svc", "uses", "Postgres v3")
    for f in (f1, f2, f3):
        f.gravity_score = 0.05
        mem._storage.save_fact(f)
    mem._absorb_dead()

    singularity_before = dict(mem._hole._singularity)
    assert len(singularity_before) >= 3

    # Patch save_meta_facts to raise after the compactor has already
    # mutated _hole.
    original = mem._storage.save_meta_facts

    def failing_save(metas):
        raise RuntimeError("simulated disk-full mid-collapse")

    mem._storage.save_meta_facts = failing_save  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="disk-full"):
            mem.collapse_singularity(threshold=0.0, min_group_size=2)
    finally:
        mem._storage.save_meta_facts = original  # type: ignore[assignment]

    # After the rollback, in-memory _hole must match disk: the source
    # facts are still there, no phantom MetaFact registered.
    assert mem._hole._singularity.keys() == singularity_before.keys(), (
        "in-memory singularity desynced after rolled-back collapse — "
        "_reload() did not run"
    )
    # No new MetaFact landed in storage.
    assert mem._storage.load_meta_facts() == []
    mem.close()


def test_collapse_success_path_still_persists(tmp_path):
    """Sanity: the happy path still works — the rollback guard
    must not break the normal commit flow."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Three facts that the compactor will bundle (vectors close
    # enough at the mock-embedding level).
    f1 = mem.add_fact("api", "lives in", "Frankfurt zone A")
    f2 = mem.add_fact("api", "lives in", "Frankfurt zone B")
    f3 = mem.add_fact("api", "lives in", "Frankfurt zone C")
    for f in (f1, f2, f3):
        f.gravity_score = 0.05
        mem._storage.save_fact(f)
    mem._absorb_dead()

    report = mem.collapse_singularity(threshold=0.0, min_group_size=2)
    assert report.groups >= 1
    assert report.absorbed_facts >= 2
    # On-disk MetaFact survived.
    assert len(mem._storage.load_meta_facts()) >= 1
    mem.close()


# --- I2: _post retry budget for transport failures --------------------


def test_post_retries_transport_failure_then_succeeds():
    """URLError on first attempt, 200 on retry — caller sees clean
    JSON, no EmbeddingError surfaces."""
    import urllib.error

    call_count = {"n": 0}

    class _OkResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"embedding": [0.1, 0.2, 0.3]}'

    def fake_urlopen(req, timeout):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise urllib.error.URLError("connection refused")
        return _OkResp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        # Sleep is a side-effect — patch it out so the test runs fast.
        with patch("birch.resonance.embeddings.time.sleep"):
            result = _post("http://localhost:11434/api/embed", {"input": "x"})

    assert call_count["n"] == 2, "should have retried once"
    assert result == {"embedding": [0.1, 0.2, 0.3]}


def test_post_gives_up_after_all_attempts_exhausted():
    """Persistent transport failure must still surface as
    EmbeddingError after the retry budget is spent — never as a raw
    urllib exception."""
    import urllib.error

    def always_fail(req, timeout):
        raise urllib.error.URLError("nothing listening")

    with patch("urllib.request.urlopen", side_effect=always_fail):
        with patch("birch.resonance.embeddings.time.sleep"):
            with pytest.raises(EmbeddingError, match="cannot reach Ollama"):
                _post("http://localhost:11434/api/embed", {"input": "x"})


def test_post_does_not_retry_http_5xx():
    """HTTPError 500 means the model failed deterministically on
    this input — retrying just doubles latency for nothing. The
    EmbeddingError should fire on the first attempt."""
    import urllib.error

    call_count = {"n": 0}

    def fake_urlopen(req, timeout):
        call_count["n"] += 1
        raise urllib.error.HTTPError(
            url=req.full_url, code=500, msg="model crashed",
            hdrs=None, fp=None,
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with patch("birch.resonance.embeddings.time.sleep"):
            with pytest.raises(EmbeddingError, match="HTTP 500"):
                _post("http://localhost:11434/api/embed", {"input": "x"})

    assert call_count["n"] == 1, "5xx must NOT trigger a retry"


def test_post_does_not_retry_http_404():
    """404 is the signal the caller uses to fall back to the legacy
    endpoint — it must propagate untouched on the FIRST attempt, not
    after burning the retry budget."""
    import urllib.error

    call_count = {"n": 0}

    def fake_urlopen(req, timeout):
        call_count["n"] += 1
        raise urllib.error.HTTPError(
            url=req.full_url, code=404, msg="not found",
            hdrs=None, fp=None,
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with patch("birch.resonance.embeddings.time.sleep"):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _post("http://localhost:11434/api/embed", {"input": "x"})
            assert exc_info.value.code == 404

    assert call_count["n"] == 1, "404 must propagate untouched, no retry"
