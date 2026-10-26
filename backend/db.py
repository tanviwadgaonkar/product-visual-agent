"""SQLite store for the agent's side-effects: demo bookings and cart items."""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("PVA_DATA_DIR", Path(__file__).resolve().parent.parent / "data")) / "app.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS leads (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT NOT NULL,
                   email TEXT NOT NULL,
                   product_id TEXT,
                   created_at TEXT NOT NULL)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS cart (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   product_id TEXT NOT NULL,
                   created_at TEXT NOT NULL)"""
        )


def add_lead(name: str, email: str, product_id: str | None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO leads (name, email, product_id, created_at) VALUES (?, ?, ?, ?)",
            (name.strip(), email.strip(), product_id, _now()),
        )
        return cur.lastrowid


def add_cart_item(product_id: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO cart (product_id, created_at) VALUES (?, ?)",
            (product_id, _now()),
        )
        return cur.lastrowid


def list_leads() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM leads ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def stats() -> dict:
    with _conn() as c:
        leads = c.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        cart = c.execute("SELECT COUNT(*) FROM cart").fetchone()[0]
        return {"leads": leads, "cart_items": cart}
