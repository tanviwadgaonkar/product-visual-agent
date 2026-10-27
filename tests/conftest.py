"""Offline fixtures: an in-memory stand-in for the Chroma collection, a
deterministic hashing embedder for Ollama embeddings, and a scripted LLM.
Every test gets its own data directory."""
import hashlib
import math
import re

import pytest
from fastapi.testclient import TestClient

from backend import agent, db, main, rag


class FakeCollection:
    """The subset of the Chroma collection API that rag.py uses."""

    def __init__(self):
        self.rows = {}  # id -> (document, metadata, embedding)

    def add(self, ids, documents, metadatas, embeddings):
        for i, d, m, e in zip(ids, documents, metadatas, embeddings):
            if i in self.rows:
                raise ValueError(f"duplicate id {i}")
            self.rows[i] = (d, m, e)

    def get(self, where, include=None):
        key, value = next(iter(where.items()))
        return {"ids": [i for i, (_, m, _) in self.rows.items() if m.get(key) == value]}

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results):
        q = query_embeddings[0]
        ranked = sorted(self.rows.items(), key=lambda kv: -_cos(q, kv[1][2]))[:n_results]
        return {
            "ids": [[i for i, _ in ranked]],
            "documents": [[r[0] for _, r in ranked]],
            "metadatas": [[r[1] for _, r in ranked]],
            "distances": [[1 - _cos(q, r[2]) for _, r in ranked]],
        }


def _cos(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def hash_embed(texts, dim=128):
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1
        out.append(v)
    return out


class ScriptedLLM:
    """Replaces ollama.chat; tests set ``reply`` to the JSON the model returns."""

    def __init__(self):
        self.reply = '{"answer": "ok", "product_id": null, "action": null}'
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append(messages)
        return {"message": {"content": self.reply}}


@pytest.fixture
def llm(monkeypatch):
    scripted = ScriptedLLM()
    monkeypatch.setattr(agent.ollama, "chat", scripted)
    return scripted


@pytest.fixture
def client(tmp_path, monkeypatch, llm):
    monkeypatch.setattr(rag, "PRODUCTS_JSON", tmp_path / "products.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(rag, "embed", hash_embed)
    rag.set_collection(FakeCollection())
    with TestClient(main.app) as c:  # runs startup: init db + seed sample data
        yield c
    rag.set_collection(None)
