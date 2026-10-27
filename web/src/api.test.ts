import { describe, expect, it } from "vitest";

import { errorFrom, toQuery } from "./api";
import { splitCitations } from "./citations";

describe("toQuery", () => {
  it("drops empty values and encodes the rest", () => {
    expect(toQuery({ q: "red bag", category: "", brand: undefined, page: 2 })).toBe("?q=red+bag&page=2");
  });
  it("returns an empty string when nothing is set", () => {
    expect(toQuery({ q: "" })).toBe("");
  });
});

describe("errorFrom", () => {
  it("reads the backend's structured error", async () => {
    const res = new Response(JSON.stringify({ error: { code: "product_not_found", message: "No product" } }), {
      status: 404,
    });
    const err = await errorFrom(res);
    expect([err.status, err.code, err.message]).toEqual([404, "product_not_found", "No product"]);
  });
  it("falls back for non-JSON bodies", async () => {
    const err = await errorFrom(new Response("<html>", { status: 502 }));
    expect(err.code).toBe("http_error");
  });
});

describe("splitCitations", () => {
  it("separates text from known citations and leaves unknown ids as text", () => {
    const parts = splitCitations("Try [B000AAAAAA] or [B000ZZZZZZ].", new Set(["B000AAAAAA"]));
    expect(parts).toEqual([
      { text: "Try " },
      { cite: "B000AAAAAA" },
      { text: " or [B000ZZZZZZ]." },
    ]);
  });
});
