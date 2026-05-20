"""Singularity Compactor — collapses clusters of dead facts into MetaFacts.

Facts that have fallen into the BlackHole at gravity < 0.10 are still
hot mass: every absorbed body keeps a 768-dim vector in the index and a
SingularityRecord in memory. Without compaction the singularity grows
linearly with every session close.

The compactor performs *gravitational collapse*: bodies that lie close
to each other in semantic space (cosine ≥ ``threshold``) are merged into
a single MetaFact whose vector is the weight-1 center of mass of the
group, normalised to unit length. The originals are deleted from both
the singularity and the index — their texts and ids live on inside the
MetaFact's ``source_texts`` / ``source_fact_ids`` for lineage.

Grouping is done with a Union-Find on a sparse adjacency graph built
from one numpy matmul against the singularity's own index. That gives
linear-in-N memory and O(N^2) compute in the worst case, but with a
high threshold the matmul is the bottleneck — not the union-find walk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .meta_fact import MetaFact

if TYPE_CHECKING:
    from .black_hole import BlackHole


# Default collapse threshold. 0.92 is tight enough to keep semantically
# distinct facts apart and loose enough to actually compact — at 0.95
# (the Hawking threshold) almost nothing groups, and below ~0.85 the
# bundle starts gluing unrelated topics together.
_DEFAULT_THRESHOLD = 0.92


@dataclass
class CollapseReport:
    """Outcome of one collapse pass — useful for stats and tests."""
    groups: int                          # number of new MetaFacts created
    absorbed_facts: int                  # FactPassports removed from singularity
    fact_mass_before: int
    fact_mass_after: int
    meta_mass_before: int
    meta_mass_after: int


class _UnionFind:
    """Path-compressing Union-Find. Strings as element ids."""

    def __init__(self, items):
        self._parent = {item: item for item in items}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def _center_of_mass(vectors: list[list[float]], weights: list[float]) -> list[float]:
    """Weighted mean of vectors, L2-normalised so cosine math stays clean.

    Public utility used by the compactor; lives here rather than as a
    standalone module because it is the only caller — promoting it to
    a separate file would be premature.
    """
    M = np.asarray(vectors, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    centre = (M * w).sum(axis=0) / w.sum()
    n = float(np.linalg.norm(centre))
    if n == 0.0:
        return centre.tolist()
    return (centre / n).tolist()


def collapse_singularity(
    hole: "BlackHole",
    threshold: float = _DEFAULT_THRESHOLD,
    min_group_size: int = 2,
) -> tuple[list[MetaFact], CollapseReport]:
    """Run one collapse pass on the FactPassports inside the black hole.

    Returns the newly created MetaFacts and a CollapseReport. The hole's
    fact singularity shrinks; its meta singularity grows. Persistence is
    the caller's responsibility — pass each new MetaFact through the
    storage layer before the next process restart.

    MetaFacts already in the singularity are left alone. A future pass
    may collapse MetaFacts with each other (recursive collapse), but
    that needs its own threshold tuning.
    """
    fact_mass_before = hole.fact_mass
    meta_mass_before = hole.meta_mass

    fact_ids = list(hole._singularity.keys())
    if len(fact_ids) < min_group_size:
        return [], CollapseReport(
            groups=0,
            absorbed_facts=0,
            fact_mass_before=fact_mass_before,
            fact_mass_after=fact_mass_before,
            meta_mass_before=meta_mass_before,
            meta_mass_after=meta_mass_before,
        )

    # Snapshot vectors aligned with fact_ids so similarity can be computed
    # in one matmul. Skip facts with empty vectors — they can't be grouped.
    vec_rows: list[list[float]] = []
    valid_ids: list[str] = []
    for fid in fact_ids:
        rec = hole._singularity.get(fid)
        if rec is None or not rec.fact.vector:
            continue
        vec_rows.append(rec.fact.vector)
        valid_ids.append(fid)

    if len(valid_ids) < min_group_size:
        return [], CollapseReport(
            groups=0,
            absorbed_facts=0,
            fact_mass_before=fact_mass_before,
            fact_mass_after=fact_mass_before,
            meta_mass_before=meta_mass_before,
            meta_mass_after=meta_mass_before,
        )

    M = np.asarray(vec_rows, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0   # avoid div-by-zero; zero-norm rows ignored anyway
    Mn = M / norms

    # Pairwise cosine — symmetric, so upper triangle is enough.
    sims = Mn @ Mn.T

    uf = _UnionFind(valid_ids)
    n = len(valid_ids)
    for i in range(n):
        # Use boolean masking on row i to skip the Python loop where possible.
        row = sims[i, i + 1:]
        hits = np.where(row >= threshold)[0]
        for offset in hits:
            j = i + 1 + int(offset)
            uf.union(valid_ids[i], valid_ids[j])

    new_metas: list[MetaFact] = []
    absorbed = 0
    for _root, members in uf.groups().items():
        if len(members) < min_group_size:
            continue

        # Build the new MetaFact from this group.
        records = [hole._singularity[mid] for mid in members]
        vectors = [r.fact.vector for r in records]
        weights = [1.0] * len(records)
        centre = _center_of_mass(vectors, weights)
        source_texts = [
            f"{r.fact.subject} {r.fact.predicate} {r.fact.object}"
            for r in records
        ]
        source_fact_ids = [r.fact.fact_id for r in records]

        meta = MetaFact(
            vector=centre,
            weight=len(records),
            source_texts=source_texts,
            source_fact_ids=source_fact_ids,
        )

        # Drop the originals from the singularity and its vector index.
        for mid in members:
            hole._singularity.pop(mid, None)
            hole._index.remove(mid)
            absorbed += 1

        hole.absorb_meta(meta)
        new_metas.append(meta)

    return new_metas, CollapseReport(
        groups=len(new_metas),
        absorbed_facts=absorbed,
        fact_mass_before=fact_mass_before,
        fact_mass_after=hole.fact_mass,
        meta_mass_before=meta_mass_before,
        meta_mass_after=hole.meta_mass,
    )
