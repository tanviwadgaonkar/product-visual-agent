"""Ask the running pipeline a fixed set of questions and print a Markdown report.

Used by .github/workflows/qa-smoke.yml with a real CLIP encoder and Ollama.
Exits non-zero if the pipeline errors. Answer quality is reported, not
asserted, because a small local model is expected to miss sometimes:
  - grounded: at least one citation, and every citation was a retrieved product
  - prose:    the answer has at least 5 words besides the citations
  - the out-of-catalogue question should be answered without any citation
"""
import re
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
]
OUT_OF_CATALOGUE = "Do you sell laptops?"  # no laptops in the data

grounded = prose = dropped = 0
refused = False
details = []
print("| Question | Retrieval | Grounded | Cited | Dropped | Prose | Seconds |")
print("|---|---|---|---|---|---|---|")
with TestClient(app) as c:
    for q in QUESTIONS + [OUT_OF_CATALOGUE]:
        t = time.perf_counter()
        r = c.post("/api/v2/ask", json={"question": q})
        r.raise_for_status()
        b = r.json()
        has_prose = len(re.sub(r"\[[A-Z0-9]{10}\]", " ", b["answer"]).split()) >= 5
        if q == OUT_OF_CATALOGUE:
            refused = not b["citations"]
        else:
            grounded += b["grounded"]
            prose += has_prose
        dropped += len(b["dropped_citations"])
        print(f"| {q} | {b['retrieval_mode']} | {'yes' if b['grounded'] else 'no'} | {len(b['citations'])} | "
              f"{len(b['dropped_citations'])} | {'yes' if has_prose else 'no'} | {time.perf_counter() - t:.1f} |")
        details.append(f"**{q}**\n\n> {b['answer']}\n")

print(f"\nIn-catalogue questions: {grounded}/{len(QUESTIONS)} grounded, {prose}/{len(QUESTIONS)} written as prose. "
      f"Out-of-catalogue question answered without citations: {'yes' if refused else 'no'}. "
      f"Citations dropped: {dropped}.\n")
print("\n".join(details))
