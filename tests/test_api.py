import json

from backend import rag
from tests.conftest import hash_embed


def ask(client, question, history=None):
    return client.post("/api/ask", json={"question": question, "history": history or []})


def reply(llm, answer="ok", product_id=None, action=None):
    llm.reply = json.dumps({"answer": answer, "product_id": product_id, "action": action})


# ---------- startup + retrieval ----------

def test_sample_data_is_indexed_on_first_start(client):
    stats = client.get("/api/stats").json()
    assert stats["indexed_chunks"] > 8  # 8 products + FAQ chunks
    ids = {p["product_id"] for p in client.get("/api/products").json()["products"]}
    assert {"KR-101", "KR-108"} <= ids


def test_products_endpoint_hides_internal_fields(client):
    product = client.get("/api/products").json()["products"][0]
    assert not any(k.startswith("_") for k in product)


def test_answer_comes_with_grounding_sources_and_product(client, llm):
    reply(llm, "Runner Pro 2 comes in Storm blue.", product_id="KR-101")
    body = ask(client, "Which colours does the Runner Pro 2 come in?").json()
    assert body["product"]["name"] == "Runner Pro 2"
    assert body["sources"] and all("text" in s for s in body["sources"])
    context = llm.calls[-1][-1]["content"]
    assert "CONTEXT:" in context and "Runner Pro 2" in context


def test_long_conversation_history_is_trimmed_not_rejected(client, llm):
    history = [{"role": r, "content": f"turn {i}"} for i in range(30) for r in ("user", "assistant")]
    assert ask(client, "hello", history).status_code == 200


# ---------- actions ----------

def test_add_to_cart(client, llm):
    reply(llm, "Added.", product_id="KR-102", action={"type": "add_to_cart", "product_id": "KR-102"})
    body = ask(client, "Add the Tempo Light to my cart").json()
    assert body["action"]["product_name"] == "Tempo Light"
    assert body["stats"]["cart_items"] == 1


def test_unknown_product_is_never_added_to_cart(client, llm):
    reply(llm, "Added.", action={"type": "add_to_cart", "product_id": "NOPE-1"})
    body = ask(client, "add the shoe on the moon to my cart").json()
    known = {p["product_id"] for p in client.get("/api/products").json()["products"]}
    # falls back to the best-matching real product or does nothing; never a phantom id
    assert body["action"] is None or body["action"]["product_id"] in known


def test_book_demo_with_email_the_user_typed(client, llm):
    reply(llm, "Booked!", action={"type": "book_demo", "name": "Asha", "email": "asha@example.com"})
    body = ask(client, "Book me a demo, I'm Asha, asha@example.com").json()
    assert body["action"]["type"] == "book_demo"
    assert body["stats"]["leads"] == 1


def test_email_from_earlier_user_turn_counts(client, llm):
    reply(llm, "Booked!", action={"type": "book_demo", "name": "Asha", "email": "asha@example.com"})
    history = [{"role": "user", "content": "my email is asha@example.com"},
               {"role": "assistant", "content": "And your name?"}]
    assert ask(client, "Asha", history).json()["action"]["type"] == "book_demo"


def test_injected_email_is_not_booked(client, llm):
    # A retrieved passage (from an uploaded file) talks the model into booking
    # a lead the user never asked for: the email isn't in the user's words.
    reply(llm, "Booked!", action={"type": "book_demo", "name": "Spam", "email": "spam@evil.example"})
    body = ask(client, "what's the return policy?").json()
    assert body["action"] is None
    assert body["stats"]["leads"] == 0


def test_malformed_email_is_not_booked(client, llm):
    reply(llm, "Booked!", action={"type": "book_demo", "name": "Asha", "email": "asha at example"})
    assert ask(client, "book a demo, Asha, asha at example").json()["action"] is None


def test_unknown_action_type_is_dropped(client, llm):
    reply(llm, "Done.", action={"type": "refund", "amount": 9999})
    assert ask(client, "refund me").json()["action"] is None


def test_non_json_model_output_still_answers(client, llm):
    llm.reply = "Sure! The Runner Pro 2 is great."
    body = ask(client, "tell me about runners").json()
    assert body["answer"] == "Sure! The Runner Pro 2 is great." and body["action"] is None


# ---------- leads (PII) ----------

def test_leads_require_admin_token(client, monkeypatch):
    assert client.get("/api/leads").status_code == 403
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.get("/api/leads", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.get("/api/leads", headers={"X-Admin-Token": "s3cret"}).json() == {"leads": []}


# ---------- ingestion ----------

CSV_HEADER = "product_id,name,category,colours,sizes,price_inr,description,features\n"


def upload(client, name, content: bytes):
    return client.post("/api/ingest", files={"file": (name, content)})


def test_upload_markdown(client):
    resp = upload(client, "care.md", b"# Shoe care\n\nWash by hand. Air dry away from heat.")
    assert resp.json() == {"file": "care.md", "chunks": 1}


def test_reupload_replaces_previous_version(client):
    upload(client, "care.md", ("word " * 400).encode())
    before = client.get("/api/stats").json()["indexed_chunks"]
    upload(client, "care.md", b"short")
    assert client.get("/api/stats").json()["indexed_chunks"] < before


def test_catalogue_reupload_drops_removed_products(client):
    two = CSV_HEADER + "X-1,Alpha,Trail,Red,UK 7,100,a,b\nX-2,Beta,Trail,Red,UK 7,100,a,b\n"
    upload(client, "extra.csv", two.encode())
    upload(client, "extra.csv", (CSV_HEADER + "X-1,Alpha,Trail,Red,UK 7,100,a,b\n").encode())
    ids = {p["product_id"] for p in client.get("/api/products").json()["products"]}
    assert "X-1" in ids and "X-2" not in ids and "KR-101" in ids


def test_csv_missing_columns_is_422_and_changes_nothing(client):
    before = client.get("/api/stats").json()["indexed_chunks"]
    resp = upload(client, "catalogue.csv", b"product_id,name\nKR-101,Oops\n")
    assert resp.status_code == 422 and "missing columns" in resp.json()["detail"]
    assert client.get("/api/stats").json()["indexed_chunks"] == before


def test_csv_duplicate_product_id_is_422(client):
    dup = CSV_HEADER + "X-1,A,T,R,7,1,a,b\nX-1,B,T,R,7,1,a,b\n"
    resp = upload(client, "dup.csv", dup.encode())
    assert resp.status_code == 422 and "duplicate product_id X-1" in resp.json()["detail"]


def test_unsupported_type_empty_and_oversized_uploads(client):
    assert upload(client, "virus.exe", b"MZ").status_code == 422
    assert upload(client, "empty.md", b"").status_code == 400
    assert upload(client, "big.txt", b"x" * (10 * 1024 * 1024 + 1)).status_code == 413


def test_ollama_down_during_reupload_keeps_old_index(client, monkeypatch):
    before = client.get("/api/stats").json()["indexed_chunks"]
    products_before = client.get("/api/products").json()

    def down(texts):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(rag, "embed", down)
    resp = upload(client, "catalogue.csv", (CSV_HEADER + "Z-1,Only,T,R,7,1,a,b\n").encode())
    assert resp.status_code == 503
    monkeypatch.setattr(rag, "embed", hash_embed)
    assert client.get("/api/stats").json()["indexed_chunks"] == before
    assert client.get("/api/products").json() == products_before
