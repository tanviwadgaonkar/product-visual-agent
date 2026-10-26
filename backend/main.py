"""Product Visual Agent — FastAPI app.

Run:  uvicorn backend.main:app --reload --port 8000
Open: http://localhost:8000
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import agent, db, rag

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DATA = ROOT / "data"

app = FastAPI(title="Product Visual Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.on_event("startup")
def startup() -> None:
    db.init_db()
    # Seed the index with the bundled sample data on first run.
    if rag.count() == 0:
        for f in ("catalogue.csv", "store_faq.md"):
            p = DATA / f
            if p.exists():
                rag.ingest_path(p)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    return rag.ingest_bytes(file.filename, raw)


@app.get("/api/products")
def products() -> dict:
    return {"products": list(rag.load_products().values())}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    passages = rag.retrieve(q, k=4)
    result = agent.run(q, req.history, passages)

    products = rag.load_products()
    pid = result["product_id"] if result["product_id"] in products else None
    if pid is None and passages and passages[0]["product_id"] in products:
        # the model didn't name a product, but the best match is one — show it
        pid = passages[0]["product_id"]

    action_result = None
    action = result["action"]
    if action:
        if action["type"] == "book_demo" and action.get("name") and action.get("email"):
            lead_id = db.add_lead(action["name"], action["email"], action.get("product_id") or pid)
            action_result = {"type": "book_demo", "id": lead_id, "name": action["name"], "email": action["email"], "product_id": action.get("product_id") or pid}
        elif action["type"] == "add_to_cart":
            target = action.get("product_id") if action.get("product_id") in products else pid
            if target:
                item_id = db.add_cart_item(target)
                action_result = {"type": "add_to_cart", "id": item_id, "product_id": target, "product_name": products[target]["name"]}

    return {
        "answer": result["answer"],
        "product": products.get(pid),
        "sources": [{"source": p["source"], "product_id": p["product_id"], "text": p["text"][:400], "score": p["score"]} for p in passages],
        "action": action_result,
        "stats": db.stats(),
    }


@app.get("/api/leads")
def leads() -> dict:
    return {"leads": db.list_leads()}


@app.get("/api/stats")
def stats() -> dict:
    return {**db.stats(), "indexed_chunks": rag.count()}
