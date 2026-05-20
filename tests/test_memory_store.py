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
    f_go     = mem.add_fact("модуль рассылок", "работает на", "Go")
    f_python = mem.add_fact("старый скрипт",   "написан на",  "Python")
    f_db     = mem.add_fact("база данных",     "использует",  "PostgreSQL")
    mem.link(f_go.fact_id, f_db.fact_id)     # рассылки зависят от БД
    mem.deprecate(f_python.fact_id, f_go.fact_id)

    print(f"  registered: {f_go}")
    print(f"  registered: {f_python} [deprecated]")
    print(f"  registered: {f_db}")
    print(f"  stats: {mem.stats}")

    # ── 2. Resonant session ───────────────────────────────────────
    section("Session A — resonant (user solved problem)")
    mem.session_start("session_A")
    mem.session_message("как настроить модуль рассылок на Go")
    mem.session_message("как подключить к PostgreSQL")
    mem.session_message("всё заработало, спасибо!")
    mem._session_fact_ids = [f_go.fact_id, f_db.fact_id]
    summary_a = mem.session_close()
    print(f"  R={summary_a['r']:+.3f} label={summary_a['label']!r}")
    print(f"  migrations={summary_a['migrations']}")
    print(f"  absorbed={summary_a['absorbed']}")
    print(f"  stats after: {mem.stats}")

    # ── 3. Toxic session ─────────────────────────────────────────
    section("Session B — toxic (deprecated fact, user stuck)")
    mem.session_start("session_B")
    mem.session_message("почему старый python скрипт не работает")
    mem.session_message("всё равно не работает")
    mem.session_message("не понимаю почему не работает")
    summary_b = mem.session_close()
    print(f"  R={summary_b['r']:+.3f} label={summary_b['label']!r}")
    print(f"  absorbed={summary_b['absorbed']}")
    print(f"  stats after: {mem.stats}")

    # ── 4. Query ─────────────────────────────────────────────────
    section("Query — 'как работает модуль рассылок'")
    results = mem.query("как работает модуль рассылок", top_k=3)
    for r in results:
        print(f"  [{r.source}] sim={r.similarity:.4f}  {r.fact}")

    # ── 5. Echo detection ────────────────────────────────────────
    section("Echo check — user returns with same problem")
    echo = mem.check_echo("старый python скрипт опять не работает")
    print(f"  echo={echo['echo']}  matched={echo['matched_session']}"
          f"  sim={echo['similarity']:.4f}")

    # ── 6. Hawking emission test ──────────────────────────────────
    section("Black hole mass + Hawking emission")
    print(f"  black hole mass={mem.stats['black_hole_mass']}")

    # Force a fact into black hole for demo — add weak fact and tick
    f_dead = mem.add_fact("временный токен", "истёк", "2024-01-01")
    f_dead.gravity_score = 0.05     # already dying
    mem._engine.register(f_dead)
    absorbed = mem._absorb_dead()
    print(f"  absorbed weak fact: {absorbed}")
    print(f"  black hole mass after: {mem.stats['black_hole_mass']}")

    hawking = mem.query("временный токен истёк", top_k=1, hawking=True)
    if hawking and hawking[0].source == "hawking":
        print(f"  ✓ Hawking emission: sim={hawking[0].similarity:.4f}  {hawking[0].fact}")
    else:
        print(f"  ✗ No Hawking emission (similarity below threshold {0.95})")
        if hawking:
            print(f"    best match: sim={hawking[0].similarity:.4f}  source={hawking[0].source}")

    print(f"\n  final stats: {mem.stats}")
    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    run()
