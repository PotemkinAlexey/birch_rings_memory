"""MetaFact — round-trip serialisation, feedback-loop surface, Hawking math."""
import json
import math

from birch.meta_fact import MetaFact


def test_defaults_match_spec():
    m = MetaFact()
    assert m.weight == 1
    assert m.source_texts == []
    assert m.source_fact_ids == []
    assert m.vector == []
    assert m.summary == ""
    assert m.gravity_score == 0.30
    assert m.layer == -1
    assert m.access_count == 0
    assert m.resonance_sum == 0.0
    assert m.resonance_count == 0
    assert len(m.meta_id) == 36, "meta_id should be a full uuid4"
    assert m.created_at > 0


def test_duck_typed_alias_for_fact_id():
    """MetaFact.fact_id mirrors meta_id so duck-typed callers stay happy."""
    m = MetaFact()
    assert m.fact_id == m.meta_id


def test_touch_increments_access_count_and_updates_timestamp():
    m = MetaFact(last_accessed=0.0)
    m.touch()
    assert m.access_count == 1
    assert m.last_accessed > 0


def test_apply_resonance_accumulates_correctly():
    m = MetaFact()
    m.apply_resonance(0.5)
    m.apply_resonance(-0.1)
    assert m.resonance_count == 2
    assert abs(m.resonance_sum - 0.4) < 1e-9
    assert abs(m.avg_resonance - 0.2) < 1e-9


def test_avg_resonance_handles_zero_count():
    assert MetaFact().avg_resonance == 0.0


def test_deprecated_and_expired_always_false():
    m = MetaFact()
    assert m.is_deprecated is False
    assert m.is_expired is False


def test_gravity_on_emission_is_logarithmic_in_weight():
    """A MetaFact with 50 dead facts must not leap into the surface layer."""
    m1 = MetaFact(weight=1)
    m10 = MetaFact(weight=10)
    m100 = MetaFact(weight=100)
    m1000 = MetaFact(weight=1000)

    g1 = m1.gravity_on_emission()
    g10 = m10.gravity_on_emission()
    g100 = m100.gravity_on_emission()
    g1000 = m1000.gravity_on_emission()

    assert g1 == 0.30
    assert abs(g10 - 0.40) < 1e-6
    assert abs(g100 - 0.50) < 1e-6
    assert abs(g1000 - 0.60) < 1e-6
    assert g1000 <= 0.70, "bonus must be capped"


def test_gravity_on_emission_caps_at_0_70():
    m = MetaFact(weight=10**9)
    assert m.gravity_on_emission() == 0.70


def test_to_dict_json_encodes_lists():
    m = MetaFact(
        meta_id="m-1",
        vector=[0.1, 0.2, 0.3],
        weight=3,
        source_texts=["mailer runs on Go"],
        source_fact_ids=["abc", "def"],
        summary="mailer stack",
        gravity_score=0.42,
        created_at=1234.5,
        layer=-1,
        access_count=5,
        last_accessed=4321.0,
        resonance_sum=0.9,
        resonance_count=3,
    )
    d = m.to_dict()
    assert d["meta_id"] == "m-1"
    assert d["weight"] == 3
    assert d["summary"] == "mailer stack"
    assert d["gravity_score"] == 0.42
    assert d["created_at"] == 1234.5
    assert d["layer"] == -1
    assert d["access_count"] == 5
    assert d["last_accessed"] == 4321.0
    assert d["resonance_sum"] == 0.9
    assert d["resonance_count"] == 3
    # All list-valued fields ride in TEXT columns.
    assert isinstance(d["vector"], str)
    assert isinstance(d["source_texts"], str)
    assert isinstance(d["source_fact_ids"], str)
    assert json.loads(d["vector"]) == [0.1, 0.2, 0.3]
    assert json.loads(d["source_texts"]) == ["mailer runs on Go"]
    assert json.loads(d["source_fact_ids"]) == ["abc", "def"]


def test_from_dict_round_trips():
    original = MetaFact(
        vector=[0.5, -0.5],
        weight=4,
        source_texts=["a b c", "d e f"],
        source_fact_ids=["id-a", "id-b"],
        summary="abc/def",
        gravity_score=0.55,
        layer=1,
        access_count=7,
        resonance_sum=1.4,
        resonance_count=4,
    )
    restored = MetaFact.from_dict(original.to_dict())
    assert restored.meta_id == original.meta_id
    assert restored.vector == original.vector
    assert restored.weight == original.weight
    assert restored.source_texts == original.source_texts
    assert restored.source_fact_ids == original.source_fact_ids
    assert restored.summary == original.summary
    assert restored.gravity_score == original.gravity_score
    assert restored.created_at == original.created_at
    assert restored.layer == original.layer
    assert restored.access_count == original.access_count
    assert restored.last_accessed == original.last_accessed
    assert restored.resonance_sum == original.resonance_sum
    assert restored.resonance_count == original.resonance_count


def test_from_dict_accepts_raw_python_lists():
    """Tests and hand-written rows can skip the JSON encode step."""
    m = MetaFact.from_dict({
        "meta_id": "m-2",
        "vector": [1.0, 0.0],
        "weight": 2,
        "source_texts": ["x y z"],
        "source_fact_ids": ["x-id"],
        "summary": "",
        "gravity_score": 0.30,
        "created_at": 1.0,
        "layer": -1,
    })
    assert m.vector == [1.0, 0.0]
    assert m.source_texts == ["x y z"]
    assert m.source_fact_ids == ["x-id"]


def test_from_dict_tolerates_missing_optional_fields():
    m = MetaFact.from_dict({"meta_id": "m-3"})
    assert m.meta_id == "m-3"
    assert m.vector == []
    assert m.source_texts == []
    assert m.source_fact_ids == []
    assert m.weight == 1
    assert m.summary == ""
    assert m.gravity_score == 0.30
    assert m.layer == -1
    assert m.access_count == 0
    assert m.resonance_count == 0


def test_from_dict_recovers_from_corrupted_json():
    """A malformed list blob must not crash — degrade to empty list."""
    m = MetaFact.from_dict({
        "meta_id": "m-4",
        "vector": "not valid json",
        "source_texts": "{also broken",
        "source_fact_ids": "[unterminated",
    })
    assert m.vector == []
    assert m.source_texts == []
    assert m.source_fact_ids == []
