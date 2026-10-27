"""Ask the running pipeline a fixed set of questions and print a Markdown report.

Used by .github/workflows/qa-smoke.yml with a real CLIP encoder and Ollama.
Exits non-zero if the pipeline errors; groundedness is reported, not asserted,
because a 1B-parameter model is expected to miss citations sometimes.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

QUESTIONS = [
    "Which table lamps do you have?",
    "I need a red handbag. What are my options?",
    "Do you have a black leather boot for women?",
    "Compare the dining tables you have.",
    "Is there a gold ring with a gemstone?",
    "Do you sell laptops?",  # not in the catalogue: should say so
]

grounded = dropped = 0
print("| Question | Retrieval | Grounded | Cited | Dropped | Seconds |\n|---|---|---|---|---|---|")
details = []
with TestClient(app) as c:
    for q in QUESTIONS:
        t = time.perf_counter()
        r = c.post("/api/v2/ask", json={"question": q})
        r.raise_for_status()
        b = r.json()
        grounded += b["grounded"]
        dropped += len(b["dropped_citations"])
        print(f"| {q} | {b['retrieval_mode']} | {'yes' if b['grounded'] else 'no'} | {len(b['citations'])} | "
              f"{len(b['dropped_citations'])} | {time.perf_counter() - t:.1f} |")
        details.append(f"**{q}**\n\n> {b['answer']}\n")
print(f"\n{grounded}/{len(QUESTIONS)} answers grounded; {dropped} citations dropped.\n")
print("\n".join(details))
