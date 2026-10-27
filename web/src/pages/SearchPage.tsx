import { useRef, useState, type FormEvent } from "react";

import { api, type Capabilities, type ScoredProduct, type SearchParams, type Sort } from "../api";
import { Badge, Empty, ErrorBox, ProductCard, Spinner } from "../components/ui";
import { useAsync } from "../useAsync";

const PAGE_SIZE = 24;
const FACET_LABELS = { category: "Category", brand: "Brand", color: "Colour" } as const;

interface Props {
  caps: Capabilities | null;
  compare: string[];
  onToggleCompare: (id: string) => void;
}

export function SearchPage({ caps, compare, onToggleCompare }: Props) {
  const [draft, setDraft] = useState("");
  const [params, setParams] = useState<SearchParams>({ sort: "relevance", page: 1 });
  const [imageResults, setImageResults] = useState<{ name: string; items: ScoredProduct[] } | null>(null);
  const [imageState, setImageState] = useState<{ busy: boolean; error: unknown }>({ busy: false, error: null });
  const fileInput = useRef<HTMLInputElement>(null);

  const [state, retry] = useAsync(() => api.search({ ...params, page_size: PAGE_SIZE }), [JSON.stringify(params)]);

  const update = (patch: Partial<SearchParams>) => {
    setImageResults(null);
    setParams((p) => ({ ...p, page: 1, ...patch }));
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    update({ q: draft.trim() });
  };

  const onImage = async (file: File | undefined) => {
    if (!file) return;
    setImageState({ busy: true, error: null });
    try {
      setImageResults({ name: file.name, items: await api.searchByImage(file) });
      setImageState({ busy: false, error: null });
    } catch (error) {
      setImageState({ busy: false, error });
    }
  };

  return (
    <div className="space-y-6">
      <section className="rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-600 px-6 py-10 text-white sm:px-10">
        <h1 className="text-2xl font-semibold sm:text-3xl">Find products by words or by photo</h1>
        <p className="mt-2 max-w-2xl text-indigo-100">
          {caps ? `${caps.products} real products from the Amazon Berkeley Objects dataset. ` : ""}
          Keyword (BM25) and CLIP vector search, fused with reciprocal rank fusion.
        </p>
        <form onSubmit={submit} className="mt-6 flex flex-col gap-3 sm:flex-row" role="search">
          <label htmlFor="q" className="sr-only">Search products</label>
          <input
            id="q"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="e.g. red leather handbag, wooden dining table"
            maxLength={200}
            className="flex-1 rounded-lg border-0 px-4 py-3 text-slate-900 shadow focus:ring-2 focus:ring-white"
          />
          <button className="rounded-lg bg-white px-5 py-3 font-medium text-indigo-700 shadow hover:bg-indigo-50">
            Search
          </button>
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            disabled={!caps?.image_search || imageState.busy}
            title={caps?.image_search ? "Search with a photo" : "Image search needs the CLIP encoder on the server (PVA_CLIP=1)"}
            className="rounded-lg border border-white/60 px-5 py-3 font-medium text-white hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {imageState.busy ? "Searching…" : "Search by photo"}
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            aria-label="Upload a product photo"
            onChange={(e) => onImage(e.target.files?.[0])}
          />
        </form>
        {caps && !caps.semantic_search && (
          <p className="mt-3 text-sm text-indigo-100">
            Semantic and photo search are off on this server; results use keyword search only.
          </p>
        )}
      </section>

      {imageState.error != null && <ErrorBox error={imageState.error} />}

      {imageResults ? (
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-semibold">Visually similar to “{imageResults.name}”</h2>
            <button onClick={() => setImageResults(null)} className="text-sm text-indigo-700 hover:underline">
              Back to results
            </button>
          </div>
          <Grid items={imageResults.items} compare={compare} onToggleCompare={onToggleCompare} />
        </section>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
          <aside className="space-y-5" aria-label="Filters">
            {state.status === "ok" &&
              (Object.keys(FACET_LABELS) as (keyof typeof FACET_LABELS)[]).map((name) => (
                <div key={name}>
                  <label htmlFor={`f-${name}`} className="mb-1 block text-sm font-medium text-slate-700">
                    {FACET_LABELS[name]}
                  </label>
                  <select
                    id={`f-${name}`}
                    value={params[name] ?? ""}
                    onChange={(e) => update({ [name]: e.target.value || undefined } as Partial<SearchParams>)}
                    className="w-full rounded-md border-slate-300 text-sm"
                  >
                    <option value="">All</option>
                    {params[name] && !state.data.facets[name].some((f) => f.value === params[name]) && (
                      <option value={params[name]}>{params[name]}</option>
                    )}
                    {state.data.facets[name].slice(0, 40).map((f) => (
                      <option key={f.value} value={f.value}>
                        {f.value} ({f.count})
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            {(params.category || params.brand || params.color) && (
              <button
                onClick={() => update({ category: undefined, brand: undefined, color: undefined })}
                className="text-sm text-indigo-700 hover:underline"
              >
                Clear filters
              </button>
            )}
          </aside>

          <section aria-live="polite">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm text-slate-600">
                {state.status === "ok" && (
                  <>
                    {state.data.total} result{state.data.total === 1 ? "" : "s"}
                    {params.q && <> for “{params.q}”</>} <Badge>{state.data.mode}</Badge>
                  </>
                )}
              </p>
              <label className="flex items-center gap-2 text-sm text-slate-600">
                Sort
                <select
                  value={params.sort}
                  onChange={(e) => update({ sort: e.target.value as Sort })}
                  className="rounded-md border-slate-300 text-sm"
                >
                  <option value="relevance">Relevance</option>
                  <option value="title_asc">Name A–Z</option>
                  <option value="title_desc">Name Z–A</option>
                </select>
              </label>
            </div>

            {state.status === "loading" && <Spinner label="Searching" />}
            {state.status === "error" && <ErrorBox error={state.error} onRetry={retry} />}
            {state.status === "ok" && state.data.total === 0 && (
              <Empty title="No products match">Try fewer words or clear the filters.</Empty>
            )}
            {state.status === "ok" && state.data.total > 0 && (
              <>
                <Grid items={state.data.items} compare={compare} onToggleCompare={onToggleCompare} />
                <Pager
                  page={state.data.page}
                  pages={Math.ceil(state.data.total / PAGE_SIZE)}
                  onPage={(page) => setParams((p) => ({ ...p, page }))}
                />
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

function Grid({ items, compare, onToggleCompare }: { items: ScoredProduct[] } & Omit<Props, "caps">) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
      {items.map((p) => (
        <ProductCard key={p.id} product={p} comparing={compare.includes(p.id)} onToggleCompare={onToggleCompare} />
      ))}
    </div>
  );
}

function Pager({ page, pages, onPage }: { page: number; pages: number; onPage: (p: number) => void }) {
  if (pages <= 1) return null;
  return (
    <nav className="mt-6 flex items-center justify-center gap-3" aria-label="Pagination">
      <button
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
        className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm disabled:opacity-40"
      >
        Previous
      </button>
      <span className="text-sm text-slate-600">
        Page {page} of {pages}
      </span>
      <button
        disabled={page >= pages}
        onClick={() => onPage(page + 1)}
        className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm disabled:opacity-40"
      >
        Next
      </button>
    </nav>
  );
}
