"""End-to-end experiment — full BirchKM memory lifecycle."""
import sys
sys.path.insert(0, "/Users/alexpotemkin/IdeaProjects/birch_rings_memory")

from birch.memory_store import MemoryStore


def section(title: str) -> None:
    print(f"\n── {title} {'─' * (54 - len(title))}")


def run():
    print(f"\n{'='*62}")
    print("BirchKM — Full Memory Lifecycle Experiment")
    print(f"{'='*62}")

    mem = MemoryStore()

    # ── 1. Add facts ─────────────────────────────────────────────
    section("Adding facts")
    f_go     = mem.add_fact("mailer service", "runs on",    "Go")
    f_python = mem.add_fact("legacy script",  "written in", "Python")
    f_db     = mem.add_fact("database",       "uses",       "PostgreSQL")
    mem.link(f_go.fact_id, f_db.fact_id)
    mem.deprecate(f_python.fact_id, f_go.fact_id)

    print(f"  registered: {f_go}")
    print(f"  registered: {f_python} [deprecated]")
    print(f"  registered: {f_db}")
    print(f"  stats: {mem.stats}")

    # ── 2. Resonant session ───────────────────────────────────────
    section("Session A — resonant (user solved problem)")
    mem.session_start("session_A")
    mem.session_message("how to configure the mailer service on Go")
    mem.session_message("how to connect it to PostgreSQL")
    mem.session_message("everything works, thanks!")
    mem._session_fact_ids = [f_go.fact_id, f_db.fact_id]
    summary_a = mem.session_close()
    print(f"  R={summary_a['r']:+.3f} label={summary_a['label']!r}")
    print(f"  migrations={summary_a['migrations']}")
    print(f"  absorbed={summary_a['absorbed']}")
    print(f"  stats after: {mem.stats}")

    # ── 3. Toxic session ─────────────────────────────────────────
    section("Session B — toxic (deprecated fact, user stuck)")
    mem.session_start("session_B")
    mem.session_message("why is the old python script not working")
    mem.session_message("still not working")
    mem.session_message("I don't understand why it's not working")
    summary_b = mem.session_close()
    print(f"  R={summary_b['r']:+.3f} label={summary_b['label']!r}")
    print(f"  absorbed={summary_b['absorbed']}")
    print(f"  stats after: {mem.stats}")

    # ── 4. Query ─────────────────────────────────────────────────
    section("Query — 'how does the mailer service work'")
    results = mem.query("how does the mailer service work", top_k=3)
    for r in results:
        print(f"  [{r.source}] sim={r.similarity:.4f}  {r.fact}")

    # ── 5. Echo detection ────────────────────────────────────────
    section("Echo check — user returns with same problem")
    echo = mem.check_echo("old python script not working again")
    print(f"  echo={echo['echo']}  matched={echo['matched_session']}"
          f"  sim={echo['similarity']:.4f}")

    # ── 6. Hawking emission test ──────────────────────────────────
    section("Black hole mass + Hawking emission")
    print(f"  black hole mass={mem.stats['black_hole_mass']}")

    f_dead = mem.add_fact("expired token", "expired at", "2024-01-01")
    f_dead.gravity_score = 0.05
    mem._engine.register(f_dead)
    absorbed = mem._absorb_dead()
    print(f"  absorbed weak fact: {absorbed}")
    print(f"  black hole mass after: {mem.stats['black_hole_mass']}")

    hawking = mem.query("expired token expired at 2024-01-01", top_k=1, hawking=True)
    if hawking and hawking[0].source == "hawking":
        print(f"  ✓ Hawking emission: sim={hawking[0].similarity:.4f}  {hawking[0].fact}")
    else:
        print(f"  ✗ No Hawking emission (similarity below threshold 0.95)")
        if hawking:
            print(f"    best match: sim={hawking[0].similarity:.4f}  source={hawking[0].source}")

    print(f"\n  final stats: {mem.stats}")
    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    run()
