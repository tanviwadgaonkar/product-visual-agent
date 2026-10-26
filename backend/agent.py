"""The conversational agent: grounded answer + optional action, as strict JSON from Ollama."""
import json
import os
import re

import ollama

CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Kite")
BRAND = os.getenv("BRAND_NAME", "Kite Running Co.")

SYSTEM = f"""You are {ASSISTANT_NAME}, the on-screen product assistant for {BRAND}. Your words are spoken aloud by an animated character, so keep every answer to one or two short, natural sentences.

Rules
- Answer ONLY from the CONTEXT passages. If the context does not contain the answer, say you don't have that information and offer to book a demo with the store team.
- Never invent products, prices, colours or policies.
- When your answer is about one specific product, put its ID in "product_id".
- Actions you can take: "book_demo" (needs name and email) and "add_to_cart" (needs product_id).
  - If the user wants a demo but has not given BOTH a name and an email, ask for the missing details in "answer" and leave "action" null.
  - Once you have name and email, set action to book_demo and confirm in "answer".
  - If the user asks to add a product to the cart, set action to add_to_cart with the product_id.
- Reply with JSON only, in exactly this shape:
{{"answer": "<spoken reply>", "product_id": "<ID or null>", "action": null | {{"type": "book_demo", "name": "...", "email": "...", "product_id": "<ID or null>"}} | {{"type": "add_to_cart", "product_id": "<ID>"}}}}
"""


def _format_context(passages: list[dict]) -> str:
    if not passages:
        return "(no passages found)"
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"[{i}] source={p['source']} product_id={p['product_id'] or '-'}\n{p['text']}")
    return "\n\n".join(parts)


def _safe_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"answer": text or "Sorry, I didn't catch that.", "product_id": None, "action": None}


def run(question: str, history: list[dict], passages: list[dict]) -> dict:
    """history: [{"role": "user"|"assistant", "content": str}, ...] (most recent last)."""
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in history[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"CONTEXT:\n{_format_context(passages)}\n\nUSER SAID: {question}\n\nRespond with the JSON object only.",
        }
    )
    res = ollama.chat(model=CHAT_MODEL, messages=messages, format="json", options={"temperature": 0.2})
    data = _safe_json(res["message"]["content"])

    # normalise
    answer = str(data.get("answer") or "").strip() or "Sorry, I didn't catch that."
    pid = data.get("product_id")
    pid = str(pid).strip() if pid and str(pid).lower() != "null" else None
    action = data.get("action") if isinstance(data.get("action"), dict) else None
    if action and action.get("type") not in ("book_demo", "add_to_cart"):
        action = None
    return {"answer": answer, "product_id": pid, "action": action}
