# AGENTS.md — Connecting BirchKM to AI Agents

BirchKM exposes itself as an MCP (Model Context Protocol) server. Any agent
that supports MCP — Claude Desktop, Claude Code, or a custom agent built on
the Anthropic SDK — can use it as a persistent, kinetic memory backend.

---

## How it works

The agent calls three tools during its lifecycle:

```
query_memory(text)          ← before answering: retrieve relevant facts
record_fact(s, p, o)        ← during session: store something worth remembering
record_session(messages)    ← after session: score it, update gravity
```

BirchKM does the rest automatically — resonance scoring, gravity migration,
echo detection, Hawking emission. The agent doesn't need to manage any of that.

---

## Setup

### 1. Install

```bash
git clone https://github.com/PotemkinAlexey/birch_rings_memory.git
cd birch_rings_memory
python -m pip install -e .
```

Ollama must be running with `nomic-embed-text`:

```bash
ollama pull nomic-embed-text
```

### 2. Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "birch-km": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "birch.server"],
      "env": {
        "BIRCH_DB": "/Users/you/.birch/memory.db"
      }
    }
  }
}
```

Replace `/absolute/path/to/venv/bin/python` with the Python interpreter from
your virtual environment (e.g. `/Users/alexpotemkin/birch_rings_memory/bin/python`).

Restart Claude Desktop. You should see `birch-km` listed in the MCP servers panel.

### 3. Configure Claude Code

Add to `~/.claude/mcp_servers.json`:

```json
{
  "birch-km": {
    "command": "/absolute/path/to/venv/bin/python",
    "args": ["-m", "birch.server"],
    "env": {
      "BIRCH_DB": "/Users/you/.birch/memory.db"
    }
  }
}
```

---

## Available tools

### `query_memory(text, top_k=5)`

Search memory for facts relevant to the given text. Call this before
composing an answer to retrieve context from past sessions.

**Returns** — list of facts, each with:

| Field | Type | Description |
|---|---|---|
| `fact_id` | str | Unique identifier |
| `subject` | str | Subject of the triple |
| `predicate` | str | Relationship |
| `object` | str | Object of the triple |
| `similarity` | float | Cosine similarity to the query |
| `layer` | int | 0=surface, 1=kinetic, 2=core |
| `gravity_score` | float | Current gravity (0–1) |
| `source` | str | `"surface"`, `"kinetic"`, `"core"`, or `"hawking"` |

**Example:**
```
query_memory("how does the mailer service connect to the database")
→ [
    {subject: "mailer service", predicate: "runs on", object: "Go", similarity: 0.91, layer: 0},
    {subject: "database", predicate: "uses", object: "PostgreSQL", similarity: 0.87, layer: 1}
  ]
```

---

### `record_fact(subject, predicate, object)`

Store a new fact as a subject–predicate–object triple. The fact is immediately
embedded and registered in the gravity engine.

Use triples that capture relationships, not prose:

| Good | Avoid |
|---|---|
| `("mailer", "runs on", "Go")` | `("mailer runs on Go", "is", "true")` |
| `("user", "prefers", "dark mode")` | `("the user said they like dark mode")` |
| `("auth service", "uses", "JWT")` | `("JWT auth")` |

**Returns:**
```json
{"fact_id": "a3f7c1b2", "layer": 1, "gravity_score": 0.5}
```

---

### `record_session(messages, agent_id="default")`

Score a completed session and update memory gravity. Pass all user messages
in order. The system will:

- Compute R score (resonant / neutral / toxic) from behavioral + semantic + repetition signals
- Propagate R to all facts touched during the session
- Detect echo if this session returns to an unresolved past problem
- Trigger gravity migration and absorb dead facts into the black hole

**Returns:**
```json
{
  "session_id": "default-1716200000-a3f7",
  "label": "resonant",
  "r_score": 0.71,
  "migrations": 2,
  "absorbed": 0,
  "stats": {"surface": 1, "kinetic": 3, "core": 0, "black_hole_mass": 0}
}
```

Call this **at the end of every session**, not after every message.

---

### `memory_stats()`

Return current memory state — layer distribution and black hole status.

```json
{
  "surface": 2,
  "kinetic": 5,
  "core": 1,
  "black_hole_mass": 3,
  "hawking_emissions": 1,
  "total_live": 8
}
```

Useful for monitoring and debugging.

---

## Recommended agent workflow

```
┌─ Session starts ─────────────────────────────────────────┐
│  1. query_memory(user's first message)                   │
│     → inject top facts into system prompt as context     │
├─ During session ─────────────────────────────────────────┤
│  2. record_fact(...) for each new piece of information   │
│     worth remembering across sessions                    │
├─ Session ends ───────────────────────────────────────────┤
│  3. record_session(all_user_messages)                    │
│     → BirchKM scores and updates gravity automatically   │
└──────────────────────────────────────────────────────────┘
```

---

## Custom backend

BirchKM's persistence layer is pluggable. `MemoryStore` accepts any object
that satisfies the `StorageBackend` protocol — no inheritance required:

```python
from birch.storage import StorageBackend   # Protocol for type-checking
from birch.memory_store import MemoryStore

class MyRedisBackend:
    def save_fact(self, fact): ...
    def delete_fact(self, fact_id): ...
    def load_facts(self): ...
    def save_edge(self, from_id, to_id): ...
    def load_edges(self): ...
    def save_echo_session(self, session_id, centroids, r_score, recorded_at): ...
    def load_echo_sessions(self): ...
    def close(self): ...

mem = MemoryStore(storage=MyRedisBackend(...))
```

Then use `mem` directly in your agent code, or point the MCP server at it
by replacing `_store` in `server.py`.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `BIRCH_DB` | `~/.birch/memory.db` | Path to the SQLite database file |

---

## Multi-agent memory (experimental)

Multiple agents can share one `MemoryStore` by pointing to the same database:

```json
{
  "mcpServers": {
    "birch-km": {
      "command": "...",
      "env": { "BIRCH_DB": "/shared/team.db" }
    }
  }
}
```

Facts that help one agent float up for all agents. Echo detection works
cross-agent — if agent A failed to resolve a problem, agent B will see the
warning when the same topic resurfaces.

`SQLiteBackend` uses `check_same_thread=False` and commit-per-write, so it
is safe for multiple processes reading and writing to the same file, though
write throughput under heavy concurrent load will be limited by SQLite's
write lock. For high-concurrency scenarios, implement a `PostgresBackend` or
`RedisBackend` using the `StorageBackend` protocol.
