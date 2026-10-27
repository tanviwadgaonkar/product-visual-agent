"""Source-grounded product Q&A over the catalogue.

Retrieval picks the products; the LLM may only talk about those. Grounding
is enforced in code, not trusted to the prompt: every product id the model
cites is checked against the retrieved set, unknown ids are dropped, and an
answer that cites nothing it was given is reported as ungrounded.
"""
from __future__ import annotations

import json
import os
import re

import ollama

from backend.catalog import Catalog, Product

CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
CITE_RE = re.compile(r"\[([A-Z0-9]{10})\]")

SYSTEM = """You are a shopping assistant for a product catalogue.
Answer ONLY from the PRODUCTS listed in the user message. Each product starts with its ID in square brackets.
- Write full sentences that describe the products (name, colour, material, what makes them different). An answer that is only a list of IDs is wrong.
- After each product you mention, cite its ID in square brackets.
- The PRODUCTS are search results and may not match the question. If none of them is the kind of item the user asked for, say "We don't carry that." and cite nothing.
- Never invent products, prices, sizes or stock levels; prices and stock are not in the data.
- Keep the answer under 120 words.
Reply with JSON only, for example:
{"answer": "We have two table lamps: the UMI Table Lamp with a fabric shade and wooden base [B000000001], and a polished nickel orb lamp with a black shade [B000000002]."}"""


def _describe(p: Product) -> str:
    parts = [f"[{p.id}] {p.title}", f"category: {p.category}"]
    for name in ("brand", "color", "material", "style"):
        if getattr(p, name):
            parts.append(f"{name}: {getattr(p, name)}")
    if p.features:
        parts.append("features: " + "; ".join(f[:200] for f in p.features[:5]))
    return " | ".join(parts)


def _parse(text: str) -> str:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data["answer"].strip()
    except json.JSONDecodeError:
        pass
    return text.strip()


def answer(question: str, products: list[Product], catalog: Catalog) -> dict:
    if not products:
        return {"answer": "I couldn't find any products matching that question.", "citations": [],
                "grounded": False, "dropped_citations": []}

    context = "\n".join(_describe(p) for p in products)
    res = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"PRODUCTS:\n{context}\n\nQUESTION: {question}"}],
        format="json",
        options={"temperature": 0.1},
    )
    text = _parse(res["message"]["content"]) or "Sorry, I couldn't produce an answer."

    allowed = {p.id for p in products}
    cited = list(dict.fromkeys(CITE_RE.findall(text)))
    dropped = [c for c in cited if c not in allowed]
    for c in dropped:  # remove citations of products the model was not given
        text = text.replace(f"[{c}]", "")
    citations = [catalog.products[c].public() for c in cited if c in allowed]
    return {"answer": re.sub(r"\s{2,}", " ", text).strip(), "citations": citations,
            "grounded": bool(citations), "dropped_citations": dropped}
