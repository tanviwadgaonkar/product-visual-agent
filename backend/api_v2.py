"""Typed catalogue API (/api/v2): search, product details, visual similarity, comparison.

Errors use one shape: {"error": {"code": ..., "message": ..., "details": ...}}.
"""
from __future__ import annotations

import logging
import time

import httpx
import ollama
from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend import qa, vectors
from backend.catalog import default_catalog
from backend.search import Mode, SearchUnavailable, Sort, compare, search

log = logging.getLogger("pva.api")
router = APIRouter(prefix="/api/v2", tags=["catalogue"])

MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, details: object = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def _error(status: int, code: str, message: str, details: object = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return _error(exc.status, exc.code, exc.message, exc.details)


async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    if request.url.path.startswith("/api/v2"):
        return _error(422, "invalid_request", "Request parameters failed validation", details)
    return JSONResponse(status_code=422, content={"detail": details})  # v1 keeps FastAPI's shape


# ---------- response models ----------

class ProductOut(BaseModel):
    id: str
    title: str
    brand: str | None
    category: str
    color: str | None
    material: str | None
    style: str | None
    features: list[str]
    image_url: str
    source: str


class ScoredProduct(ProductOut):
    score: float


class FacetValue(BaseModel):
    value: str
    count: int


class SearchOut(BaseModel):
    items: list[ScoredProduct]
    total: int
    page: int
    page_size: int
    mode: str
    facets: dict[str, list[FacetValue]]


class CompareRow(BaseModel):
    field: str
    values: list[str | None]
    same: bool


class CompareOut(BaseModel):
    products: list[ProductOut]
    attributes: list[CompareRow]
    shared_features: list[str]


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    category: str | None = Field(None, max_length=50)


class AskOut(BaseModel):
    answer: str
    citations: list[ProductOut]
    grounded: bool
    dropped_citations: list[str]
    retrieved: list[ScoredProduct]
    retrieval_mode: str


class Capabilities(BaseModel):
    products: int
    keyword_search: bool
    similar_products: bool
    semantic_search: bool
    image_search: bool
    embedding_model: str | None


# ---------- helpers ----------

def _product_or_404(product_id: str):
    p = default_catalog().get(product_id)
    if p is None:
        raise APIError(404, "product_not_found", f"No product with id {product_id!r}")
    return p


# ---------- routes ----------

@router.get("/capabilities", response_model=Capabilities)
def capabilities() -> Capabilities:
    index, enc = vectors.default_index(), vectors.encoder()
    return Capabilities(products=len(default_catalog()), keyword_search=True, similar_products=index is not None,
                        semantic_search=index is not None and enc is not None,
                        image_search=index is not None and enc is not None,
                        embedding_model=index.model if index else None)


@router.get("/products", response_model=SearchOut)
def list_products(
    q: str = Query("", max_length=200),
    category: str | None = Query(None, max_length=50),
    brand: str | None = Query(None, max_length=100),
    color: str | None = Query(None, max_length=50),
    mode: Mode = "auto",
    sort: Sort = "relevance",
    page: int = Query(1, ge=1, le=1000),
    page_size: int = Query(24, ge=1, le=100),
) -> SearchOut:
    started = time.perf_counter()
    try:
        result = search(default_catalog(), q, category=category, brand=brand, color=color, mode=mode, sort=sort,
                        page=page, page_size=page_size, index=vectors.default_index(), encoder=vectors.encoder())
    except SearchUnavailable as exc:
        raise APIError(503, "semantic_search_unavailable", str(exc)) from exc
    log.info("search q=%r mode=%s total=%d ms=%.1f", q, result.mode, result.total, (time.perf_counter() - started) * 1000)
    return SearchOut(**result.__dict__)


@router.get("/products/{product_id}", response_model=ProductOut)
def product_detail(product_id: str) -> dict:
    return _product_or_404(product_id).public()


@router.get("/products/{product_id}/similar", response_model=list[ScoredProduct])
def similar_products(product_id: str, limit: int = Query(8, ge=1, le=50)) -> list[dict]:
    _product_or_404(product_id)
    index = vectors.default_index()
    if index is None or product_id not in index:
        raise APIError(503, "embeddings_unavailable", "Image embeddings have not been generated (data/abo/embeddings.npz)")
    catalog = default_catalog()
    return [{**catalog.products[pid].public(), "score": round(s, 4)}
            for pid, s in index.similar_to(product_id)[:limit] if pid in catalog.products]


@router.post("/search/image", response_model=list[ScoredProduct])
async def search_by_image(file: UploadFile = File(...), limit: int = Query(8, ge=1, le=50)) -> list[dict]:
    if file.content_type not in IMAGE_TYPES:
        raise APIError(415, "unsupported_media_type", "Upload a JPEG, PNG or WebP image", {"got": file.content_type})
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise APIError(400, "empty_file", "The uploaded file is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise APIError(413, "file_too_large", "Images must be 5 MB or smaller")
    index, enc = vectors.default_index(), vectors.encoder()
    if index is None or enc is None:
        raise APIError(503, "image_search_unavailable", "Image search needs the CLIP encoder (PVA_CLIP=1) and embeddings")
    try:
        # CLIP inference is CPU-bound (seconds); on the event loop it would stall every other request.
        vec = await run_in_threadpool(enc.encode_image, raw)
    except Exception as exc:  # PIL raises several unrelated types for corrupt files
        raise APIError(422, "invalid_image", "Could not decode the image") from exc
    catalog = default_catalog()
    return [{**catalog.products[pid].public(), "score": round(s, 4)}
            for pid, s in index.by_image_vector(vec)[:limit] if pid in catalog.products]


@router.get("/compare", response_model=CompareOut)
def compare_products(ids: str = Query(..., description="2 to 4 comma-separated product ids")) -> dict:
    wanted = list(dict.fromkeys(i.strip() for i in ids.split(",") if i.strip()))
    if not 2 <= len(wanted) <= 4:
        raise APIError(422, "invalid_request", "Compare between 2 and 4 distinct products", {"ids": wanted})
    for pid in wanted:
        _product_or_404(pid)
    return compare(default_catalog(), wanted)


@router.post("/ask", response_model=AskOut)
def ask(body: AskIn) -> dict:
    """Retrieve the 5 best-matching products, then answer only from them, with checked citations."""
    question = body.question.strip()
    if not question:
        raise APIError(422, "invalid_request", "Question is empty")
    catalog = default_catalog()
    result = search(catalog, question, category=body.category, page_size=5,
                    index=vectors.default_index(), encoder=vectors.encoder())
    try:
        out = qa.answer(question, [catalog.products[i["id"]] for i in result.items], catalog)
    except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
        raise APIError(503, "llm_unavailable", f"The language model is unavailable: {exc}") from exc
    log.info("ask q=%r retrieved=%d grounded=%s dropped=%d", question, len(result.items), out["grounded"],
             len(out["dropped_citations"]))
    return {**out, "retrieved": result.items, "retrieval_mode": result.mode}
