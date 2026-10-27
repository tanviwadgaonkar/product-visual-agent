"""Catalogue search and the /api/v2 endpoints, on the real 360-product ABO subset.

Vectors are stand-ins (hashed title tokens) so the tests run in seconds without
CLIP; the real-model path is exercised by the embeddings workflow."""
import io

import numpy as np
import pytest

from backend import vectors
from backend.catalog import _clean_keywords, default_catalog, tokenize
from backend.search import SearchUnavailable, search
from tests.conftest import hash_embed

DIM = 64


def _vec(text: str) -> np.ndarray:
    v = np.asarray(hash_embed([text], dim=DIM)[0], dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


class FakeEncoder(vectors.Encoder):
    def encode_text(self, text):
        return _vec(text)

    def encode_image(self, raw):
        if raw.startswith(b"not an image"):
            raise OSError("cannot identify image file")
        return _vec(raw.decode("utf-8", errors="ignore"))


@pytest.fixture
def catalog():
    return default_catalog()


@pytest.fixture
def fake_index(catalog):
    ids = sorted(catalog.products)
    mat = np.stack([_vec(catalog.products[i].title) for i in ids])
    return vectors.VectorIndex(ids=ids, image=mat, text=mat, model="fake-hash")


@pytest.fixture
def with_vectors(monkeypatch, fake_index):
    monkeypatch.setattr(vectors, "default_index", lambda: fake_index)
    vectors.set_encoder(FakeEncoder())
    yield fake_index
    vectors.set_encoder(None)


@pytest.fixture
def without_vectors(monkeypatch):
    monkeypatch.setattr(vectors, "default_index", lambda: None)
    vectors.set_encoder(None)


# ---------- catalogue ----------

def test_catalogue_loads_all_products(catalog):
    assert len(catalog) == 360
    assert {p.category for p in catalog.products.values()} >= {"Shoes", "Lamps", "Rugs"}


def test_keywords_are_split_and_deduplicated():
    assert _clean_keywords(["Shoes", "shoes^dress shoes", "shoes^dress shoes"]) == ["shoes", "dress shoes"]


def test_tokenize_drops_stopwords():
    assert tokenize("A lamp for the Desk") == ["lamp", "desk"]


def test_bm25_ranks_matching_titles_first(catalog):
    top = [catalog.products[pid] for pid, _ in catalog.bm25("oxford")[:3]]
    assert top and all("oxford" in (p.title + " ".join(p.keywords)).lower() for p in top)


def test_bm25_unknown_terms_match_nothing(catalog):
    assert catalog.bm25("zzqxv") == []
    assert catalog.bm25("the and of") == []


def test_filters_paging_and_facets(catalog):
    res = search(catalog, category="lamps", page_size=10)
    assert res.total == 30 and len(res.items) == 10
    assert all(i["category"] == "Lamps" for i in res.items)
    assert res.facets["category"] == [{"value": "Lamps", "count": 30}]
    page3 = search(catalog, category="Lamps", page=3, page_size=10)
    assert len(page3.items) == 10
    assert not {i["id"] for i in res.items} & {i["id"] for i in page3.items}


def test_title_sort(catalog):
    titles = [i["title"].lower() for i in search(catalog, category="Rugs", sort="title_desc", page_size=100).items]
    assert titles == sorted(titles, reverse=True)


def test_semantic_mode_without_encoder_is_explicit_error(catalog):
    with pytest.raises(SearchUnavailable):
        search(catalog, "red chair", mode="semantic")


def test_auto_mode_picks_hybrid_when_vectors_exist(catalog, fake_index):
    res = search(catalog, "black oxford", index=fake_index, encoder=FakeEncoder())
    assert res.mode == "hybrid"
    assert "oxford" in res.items[0]["title"].lower()


def test_rrf_rewards_agreement():
    fused = vectors.rrf([("a", 9), ("b", 5)], [("b", 0.9), ("c", 0.8)])
    assert fused[0][0] == "b"


# ---------- API ----------

def test_capabilities_reflect_missing_vectors(client, without_vectors):
    body = client.get("/api/v2/capabilities").json()
    assert body == {"products": 360, "keyword_search": True, "similar_products": False,
                    "semantic_search": False, "image_search": False, "embedding_model": None}


def test_search_endpoint_keyword_fallback(client, without_vectors):
    r = client.get("/api/v2/products", params={"q": "leather boot", "page_size": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "keyword" and body["total"] > 0 and len(body["items"]) <= 5
    assert body["items"][0]["image_url"].startswith("/images/")


def test_semantic_unavailable_is_503_with_error_shape(client, without_vectors):
    r = client.get("/api/v2/products", params={"q": "lamp", "mode": "semantic"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "semantic_search_unavailable"


def test_validation_errors_use_error_shape(client):
    r = client.get("/api/v2/products", params={"page_size": 0, "sort": "price"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert {tuple(d["loc"]) for d in err["details"]} == {("query", "page_size"), ("query", "sort")}


def test_product_detail_and_404(client, catalog):
    pid = next(iter(catalog.products))
    assert client.get(f"/api/v2/products/{pid}").json()["id"] == pid
    r = client.get("/api/v2/products/NOPE")
    assert r.status_code == 404 and r.json()["error"]["code"] == "product_not_found"


def test_product_images_are_served(client, catalog):
    p = next(iter(catalog.products.values()))
    r = client.get(f"/images/{p.image}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_similar_products(client, with_vectors, catalog):
    pid = next(iter(catalog.products))
    items = client.get(f"/api/v2/products/{pid}/similar", params={"limit": 5}).json()
    assert len(items) == 5 and pid not in {i["id"] for i in items}
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True)


def test_similar_without_embeddings_is_503(client, without_vectors, catalog):
    pid = next(iter(catalog.products))
    assert client.get(f"/api/v2/products/{pid}/similar").json()["error"]["code"] == "embeddings_unavailable"


def test_image_search(client, with_vectors):
    files = {"file": ("q.jpg", io.BytesIO(b"oxford shoe black"), "image/jpeg")}
    items = client.post("/api/v2/search/image", files=files, params={"limit": 3}).json()
    assert len(items) == 3


@pytest.mark.parametrize("name,data,ctype,status,code", [
    ("q.gif", b"GIF89a", "image/gif", 415, "unsupported_media_type"),
    ("q.jpg", b"", "image/jpeg", 400, "empty_file"),
    ("q.jpg", b"not an image", "image/jpeg", 422, "invalid_image"),
])
def test_image_search_rejects_bad_uploads(client, with_vectors, name, data, ctype, status, code):
    r = client.post("/api/v2/search/image", files={"file": (name, io.BytesIO(data), ctype)})
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_compare(client, catalog):
    lamps = [p.id for p in catalog.products.values() if p.category == "Lamps"][:2]
    body = client.get("/api/v2/compare", params={"ids": ",".join(lamps)}).json()
    assert [p["id"] for p in body["products"]] == lamps
    category = next(r for r in body["attributes"] if r["field"] == "category")
    assert category == {"field": "category", "values": ["Lamps", "Lamps"], "same": True}


@pytest.mark.parametrize("ids", ["ONE", "A,A", "A,B,C,D,E"])
def test_compare_needs_two_to_four_distinct(client, ids):
    assert client.get("/api/v2/compare", params={"ids": ids}).status_code == 422


def test_compare_unknown_product_is_404(client, catalog):
    pid = next(iter(catalog.products))
    assert client.get("/api/v2/compare", params={"ids": f"{pid},NOPE"}).status_code == 404


# ---------- embedding evaluation maths ----------

def test_embedding_eval_metrics_on_synthetic_clusters():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "embed_catalog.py"
    spec = importlib.util.spec_from_file_location("embed_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Two perfectly separated categories of 6 items each: precision must be 1.0.
    ids = [f"p{i}" for i in range(12)]
    cats = {pid: ("Lamps" if i < 6 else "Rugs") for i, pid in enumerate(ids)}
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    image = np.stack([a if cats[i] == "Lamps" else b for i in ids])
    encode = lambda texts: np.stack([a if "lamps" in t else b for t in texts])  # noqa: E731
    report = mod.evaluate(ids, image, cats, encode)
    assert report["image_to_image_precision_at_5"] == 1.0
    assert report["text_to_image_precision_at_10_by_category"] == {"Lamps": 0.6, "Rugs": 0.6}  # only 6 of 10 can match
    assert report["chance_level"] == 0.5


# ---------- committed CLIP vectors (from the embeddings workflow) ----------

def test_committed_embeddings_match_catalogue(catalog):
    index = vectors.load_index()
    if index is None:
        pytest.skip("data/abo/embeddings.npz not generated")
    assert sorted(index.ids) == sorted(catalog.products)
    assert index.image.shape == index.text.shape == (len(catalog), 512)
    norms = np.linalg.norm(index.image, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)  # stored as float16, so not exact


def test_real_vectors_similar_products_share_category(catalog):
    index = vectors.load_index()
    if index is None:
        pytest.skip("data/abo/embeddings.npz not generated")
    lamp = next(p.id for p in catalog.products.values() if p.category == "Lamps")
    top = [catalog.products[pid].category for pid, _ in index.similar_to(lamp)[:5]]
    assert top.count("Lamps") >= 3


# ---------- grounded Q&A ----------

def test_ask_cites_only_retrieved_products(client, llm, without_vectors, catalog):
    retrieved = client.get("/api/v2/products", params={"q": "table lamp", "page_size": 5}).json()["items"]
    good, fake = retrieved[0]["id"], "B0FAKE0000"
    llm.reply = f'{{"answer": "Try [{good}] or [{fake}]."}}'
    body = client.post("/api/v2/ask", json={"question": "table lamp"}).json()
    assert body["grounded"] is True
    assert [c["id"] for c in body["citations"]] == [good]
    assert body["dropped_citations"] == [fake] and fake not in body["answer"]
    assert [r["id"] for r in body["retrieved"]] == [r["id"] for r in retrieved]
    # the model only ever saw the retrieved products
    prompt = llm.calls[-1][-1]["content"]
    assert all(f"[{r['id']}]" in prompt for r in retrieved)


def test_ask_without_citations_is_ungrounded(client, llm, without_vectors):
    llm.reply = '{"answer": "Lamps are great."}'
    body = client.post("/api/v2/ask", json={"question": "lamp"}).json()
    assert body["grounded"] is False and body["citations"] == []


def test_ask_with_no_matches_skips_the_llm(client, llm, without_vectors):
    calls = len(llm.calls)
    body = client.post("/api/v2/ask", json={"question": "zzqxv"}).json()
    assert body["grounded"] is False and len(llm.calls) == calls


def test_ask_llm_down_is_503(client, monkeypatch, without_vectors):
    from backend import qa

    def down(**kwargs):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(qa.ollama, "chat", down)
    r = client.post("/api/v2/ask", json={"question": "lamp"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unavailable"


def test_ask_validates_input(client):
    assert client.post("/api/v2/ask", json={"question": ""}).json()["error"]["code"] == "invalid_request"
    assert client.post("/api/v2/ask", json={"question": "x" * 501}).status_code == 422


def test_facets_merge_case_variants(catalog):
    colors = [f["value"].lower() for f in catalog.facets()["color"]]
    assert len(colors) == len(set(colors))
    pink = next(f for f in catalog.facets()["color"] if f["value"].lower() == "pink")
    assert pink["count"] == sum(1 for p in catalog.products.values() if (p.color or "").lower() == "pink")
