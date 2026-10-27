"""Product Visual Agent — FastAPI app.

Run:  uvicorn backend.main:app --reload --port 8000
Open: http://localhost:8000
"""
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import ollama
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend import agent, api_v2, db, rag
from backend.catalog import ABO_DIR

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SEED_DATA = ROOT / "data"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Seed the index with the bundled sample data on first run.
    if rag.count() == 0:
        for f in ("catalogue.csv", "store_faq.md"):
            p = SEED_DATA / f
            if p.exists():
                rag.ingest_path(p)
    yield


app = FastAPI(title="Product Visual Agent", version="0.3.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
app.mount("/images", StaticFiles(directory=ABO_DIR / "images"), name="images")
app.include_router(api_v2.router)
app.add_exception_handler(api_v2.APIError, api_v2.api_error_handler)
app.add_exception_handler(RequestValidationError, api_v2.validation_handler)


@app.exception_handler(ConnectionError)
@app.exception_handler(httpx.HTTPError)
@app.exception_handler(ollama.ResponseError)
async def model_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """Ollama down or missing a model: an upstream failure (503), not a server bug."""
    return JSONResponse(status_code=503, content={"detail": f"Local model unavailable: {exc}. Is Ollama running?"})


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(max_length=1000)
    history: list[Turn] = Field(default_factory=list)

    @field_validator("history")
    @classmethod
    def recent_turns_only(cls, turns: list[Turn]) -> list[Turn]:
        # The browser keeps the whole conversation; only recent turns matter
        # (the agent uses the last 6), so trim instead of rejecting.
        return turns[-20:]


def _public(product: dict | None) -> dict | None:
    return {k: v for k, v in product.items() if not k.startswith("_")} if product else None


def _user_text(question: str, history: list[Turn]) -> str:
    return "\n".join([t.content for t in history if t.role == "user"] + [question])


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File larger than 10 MB")
    try:
        return rag.ingest_bytes(file.filename or "upload", raw)
    except rag.IngestError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/products")
def products() -> dict:
    return {"products": [_public(p) for p in rag.load_products().values()]}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    passages = rag.retrieve(q, k=4)
    result = agent.run(q, [t.model_dump() for t in req.history], passages)

    products = rag.load_products()
    pid = result["product_id"] if result["product_id"] in products else None
    if pid is None and passages and passages[0]["product_id"] in products:
        # the model didn't name a product, but the best match is one — show it
        pid = passages[0]["product_id"]

    action_result = None
    action = result["action"]
    if action:
        if action["type"] == "book_demo":
            name = str(action.get("name") or "").strip()
            email = str(action.get("email") or "").strip()
            # The model's output is shaped by retrieved passages, which come from
            # uploaded files. Only book with an email the *user* actually typed,
            # so injected document text can't create leads.
            if name and EMAIL_RE.fullmatch(email) and email.lower() in _user_text(q, req.history).lower():
                target = action.get("product_id") if action.get("product_id") in products else pid
                lead_id = db.add_lead(name, email, target)
                action_result = {"type": "book_demo", "id": lead_id, "name": name, "email": email, "product_id": target}
        elif action["type"] == "add_to_cart":
            target = action.get("product_id") if action.get("product_id") in products else pid
            if target:
                item_id = db.add_cart_item(target)
                action_result = {"type": "add_to_cart", "id": item_id, "product_id": target, "product_name": products[target]["name"]}

    return {
        "answer": result["answer"],
        "product": _public(products.get(pid)),
        "sources": [{"source": p["source"], "product_id": p["product_id"], "text": p["text"][:400], "score": p["score"]} for p in passages],
        "action": action_result,
        "stats": db.stats(),
    }


@app.get("/api/leads")
def leads(x_admin_token: str | None = Header(default=None)) -> dict:
    """Names and emails: only with the ADMIN_TOKEN set on the server."""
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(403, "Set ADMIN_TOKEN on the server and send it as X-Admin-Token")
    return {"leads": db.list_leads()}


@app.get("/api/stats")
def stats() -> dict:
    return {**db.stats(), "indexed_chunks": rag.count()}
