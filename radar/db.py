"""SQLite persistence: watchlists, runs, findings, signals and the LLM audit log."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    config TEXT NOT NULL,              -- JSON: vehicles, markets, components, suppliers, topics, role
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,              -- running | ok | error
    stats TEXT,                        -- JSON
    brief TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    key TEXT NOT NULL,                 -- stable de-dup key (campaign number or URL)
    source TEXT NOT NULL,              -- nhtsa | web
    type TEXT NOT NULL,                -- recall | regulation | supplier | other
    title TEXT NOT NULL,
    url TEXT,
    summary TEXT,
    published TEXT,
    market TEXT,
    severity TEXT,
    score REAL,
    affected TEXT,
    action TEXT,
    status TEXT NOT NULL DEFAULT 'new', -- new | acknowledged | actioned | dismissed
    first_run_id INTEGER,
    first_seen TEXT NOT NULL,
    raw TEXT,
    UNIQUE (watchlist_id, key)
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    watchlist_id INTEGER NOT NULL,
    vehicle TEXT NOT NULL,
    component TEXT NOT NULL,
    recent INTEGER, baseline REAL, ratio REAL,
    recalled INTEGER,                  -- 1 if an open recall already covers this component
    analysis TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    at TEXT NOT NULL,
    step TEXT, model TEXT, seconds REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER, ok INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,                -- run | question
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_findings_wl ON findings(watchlist_id, first_seen);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def q(self, sql, args=()):
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # ---- watchlists
    def save_watchlist(self, name: str, config: dict, wl_id: int | None = None) -> int:
        with self.tx() as c:
            if wl_id:
                c.execute("UPDATE watchlists SET name=?, config=?, updated_at=? WHERE id=?",
                          (name, json.dumps(config), now(), wl_id))
                return wl_id
            cur = c.execute("INSERT INTO watchlists (name, config, created_at, updated_at) VALUES (?,?,?,?)",
                            (name, json.dumps(config), now(), now()))
            return cur.lastrowid

    def watchlists(self) -> list[dict]:
        rows = self.q("SELECT * FROM watchlists ORDER BY name")
        for r in rows:
            r["config"] = json.loads(r["config"])
        return rows

    def watchlist(self, wl_id: int) -> dict | None:
        rows = [w for w in self.watchlists() if w["id"] == wl_id]
        return rows[0] if rows else None

    def delete_watchlist(self, wl_id: int):
        with self.tx() as c:
            c.execute("DELETE FROM watchlists WHERE id=?", (wl_id,))

    # ---- runs
    def start_run(self, wl_id: int) -> int:
        with self.tx() as c:
            return c.execute("INSERT INTO runs (watchlist_id, started_at, status) VALUES (?,?,'running')",
                             (wl_id, now())).lastrowid

    def finish_run(self, run_id: int, status: str, stats: dict | None = None,
                   brief: str | None = None, error: str | None = None):
        with self.tx() as c:
            c.execute("UPDATE runs SET finished_at=?, status=?, stats=?, brief=?, error=? WHERE id=?",
                      (now(), status, json.dumps(stats or {}), brief, error, run_id))

    def mark_interrupted_runs(self) -> int:
        """Runs still 'running' when a new process starts were cut off by a restart."""
        with self.tx() as c:
            return c.execute("UPDATE runs SET status='error', finished_at=?, error=? WHERE status='running'",
                             (now(), "Interrupted: the app restarted during this run")).rowcount

    def runs(self, wl_id: int, limit: int = 50) -> list[dict]:
        rows = self.q("SELECT * FROM runs WHERE watchlist_id=? ORDER BY id DESC LIMIT ?", (wl_id, limit))
        for r in rows:
            r["stats"] = json.loads(r["stats"] or "{}")
        return rows

    # ---- findings
    def known_keys(self, wl_id: int) -> set[str]:
        return {r["key"] for r in self.q("SELECT key FROM findings WHERE watchlist_id=?", (wl_id,))}

    def add_findings(self, wl_id: int, run_id: int, items: list[dict]) -> int:
        cols = ["key", "source", "type", "title", "url", "summary", "published", "market",
                "severity", "score", "affected", "action"]
        added = 0
        with self.tx() as c:
            for it in items:
                cur = c.execute(
                    f"INSERT OR IGNORE INTO findings (watchlist_id, first_run_id, first_seen, raw, status,"
                    f" {', '.join(cols)}) VALUES (?,?,?,?,?,{','.join('?' * len(cols))})",
                    [wl_id, run_id, now(), json.dumps(it.get("raw", {})), it.get("status") or "new"]
                    + [it.get(k) for k in cols])
                added += cur.rowcount
        return added

    def findings(self, wl_id: int, run_id: int | None = None) -> list[dict]:
        sql = "SELECT * FROM findings WHERE watchlist_id=?"
        args: list = [wl_id]
        if run_id:
            sql += " AND first_run_id=?"
            args.append(run_id)
        sql += (" ORDER BY CASE severity WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END,"
                " score DESC, first_seen DESC")
        return self.q(sql, args)

    def set_status(self, finding_id: int, status: str):
        with self.tx() as c:
            c.execute("UPDATE findings SET status=? WHERE id=?", (status, finding_id))

    def search_findings(self, wl_id: int, text: str, limit: int = 15) -> list[dict]:
        like = f"%{text}%"
        return self.q("SELECT id, type, severity, title, url, summary, affected, status, first_seen "
                      "FROM findings WHERE watchlist_id=? AND (title LIKE ? OR summary LIKE ? OR affected LIKE ?) "
                      "ORDER BY first_seen DESC LIMIT ?", (wl_id, like, like, like, limit))

    # ---- signals
    def add_signals(self, run_id: int, wl_id: int, signals: list[dict]):
        with self.tx() as c:
            for s in signals:
                c.execute("INSERT INTO signals (run_id, watchlist_id, vehicle, component, recent, baseline,"
                          " ratio, recalled, analysis) VALUES (?,?,?,?,?,?,?,?,?)",
                          (run_id, wl_id, s["vehicle"], s["component"], s["recent"], s["baseline"],
                           s["ratio"], int(s["recalled"]), s.get("analysis")))

    def latest_signals(self, wl_id: int) -> list[dict]:
        return self.q("SELECT * FROM signals WHERE run_id = (SELECT MAX(run_id) FROM signals WHERE watchlist_id=?)"
                      " ORDER BY recalled, ratio DESC", (wl_id,))

    # ---- LLM audit log
    def log_llm(self, run_id: int | None, rec: dict):
        with self.tx() as c:
            c.execute("INSERT INTO llm_calls (run_id, at, step, model, seconds, prompt_tokens, completion_tokens,"
                      " ok, error) VALUES (?,?,?,?,?,?,?,?,?)",
                      (run_id, now(), rec["step"], rec["model"], rec["seconds"], rec.get("prompt_tokens"),
                       rec.get("completion_tokens"), int(rec["ok"]), rec.get("error")))

    def add_usage(self, kind: str):
        with self.tx() as c:
            c.execute("INSERT INTO usage_events (kind, at) VALUES (?, ?)", (kind, now()))

    def llm_calls(self, run_id: int) -> list[dict]:
        return self.q("SELECT step, model, seconds, prompt_tokens, completion_tokens, ok, error "
                      "FROM llm_calls WHERE run_id=? ORDER BY id", (run_id,))
