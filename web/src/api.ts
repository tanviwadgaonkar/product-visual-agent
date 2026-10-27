// Typed client for the FastAPI /api/v2 endpoints (shapes mirror backend/api_v2.py).

export interface Product {
  id: string;
  title: string;
  brand: string | null;
  category: string;
  color: string | null;
  material: string | null;
  style: string | null;
  features: string[];
  image_url: string;
  source: string;
}

export interface ScoredProduct extends Product {
  score: number;
}

export interface FacetValue {
  value: string;
  count: number;
}

export type Mode = "auto" | "keyword" | "semantic" | "hybrid";
export type Sort = "relevance" | "title_asc" | "title_desc";

export interface SearchParams {
  q?: string;
  category?: string;
  brand?: string;
  color?: string;
  mode?: Mode;
  sort?: Sort;
  page?: number;
  page_size?: number;
}

export interface SearchResponse {
  items: ScoredProduct[];
  total: number;
  page: number;
  page_size: number;
  mode: string;
  facets: Record<"category" | "brand" | "color", FacetValue[]>;
}

export interface CompareResponse {
  products: Product[];
  attributes: { field: string; values: (string | null)[]; same: boolean }[];
  shared_features: string[];
}

export interface AskResponse {
  answer: string;
  citations: Product[];
  grounded: boolean;
  dropped_citations: string[];
  retrieved: ScoredProduct[];
  retrieval_mode: string;
}

export interface Capabilities {
  products: number;
  keyword_search: boolean;
  similar_products: boolean;
  semantic_search: boolean;
  image_search: boolean;
  embedding_model: string | null;
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

/** Query string with empty values left out, so URLs stay short and cacheable. */
export function toQuery(params: Record<string, string | number | undefined | null>): string {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
  }
  const s = qs.toString();
  return s ? `?${s}` : "";
}

/** Turn any failed response into an ApiError, using the backend's {"error": {...}} shape when present. */
export async function errorFrom(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    if (body?.error?.message) return new ApiError(res.status, body.error.code, body.error.message);
    if (typeof body?.detail === "string") return new ApiError(res.status, "error", body.detail);
  } catch {
    // not JSON
  }
  return new ApiError(res.status, "http_error", `Request failed (${res.status})`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) throw await errorFrom(res);
  return res.json() as Promise<T>;
}

export const api = {
  capabilities: () => request<Capabilities>("/api/v2/capabilities"),
  search: (p: SearchParams) => request<SearchResponse>(`/api/v2/products${toQuery({ ...p })}`),
  product: (id: string) => request<Product>(`/api/v2/products/${encodeURIComponent(id)}`),
  similar: (id: string, limit = 8) =>
    request<ScoredProduct[]>(`/api/v2/products/${encodeURIComponent(id)}/similar${toQuery({ limit })}`),
  compare: (ids: string[]) => request<CompareResponse>(`/api/v2/compare${toQuery({ ids: ids.join(",") })}`),
  ask: (question: string, category?: string) =>
    request<AskResponse>("/api/v2/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, category: category || null }),
    }),
  searchByImage: (file: File, limit = 12) => {
    const form = new FormData();
    form.append("file", file);
    return request<ScoredProduct[]>(`/api/v2/search/image${toQuery({ limit })}`, { method: "POST", body: form });
  },
};
