import { api, type Capabilities } from "../api";
import { Badge, Empty, ErrorBox, ProductCard, Spinner } from "../components/ui";
import { useAsync } from "../useAsync";

interface Props {
  id: string;
  caps: Capabilities | null;
  compare: string[];
  onToggleCompare: (id: string) => void;
}

const ATTRS = ["brand", "category", "color", "material", "style"] as const;

export function ProductPage({ id, caps, compare, onToggleCompare }: Props) {
  const [product, retry] = useAsync(() => api.product(id), [id]);
  const [similar] = useAsync(
    () => (caps?.similar_products ? api.similar(id, 8) : Promise.resolve(null)),
    [id, caps?.similar_products],
  );

  if (product.status === "loading") return <Spinner />;
  if (product.status === "error") return <ErrorBox error={product.error} onRetry={retry} />;
  const p = product.data;
  const inCompare = compare.includes(p.id);

  return (
    <div className="space-y-10">
      <a href="#/" className="text-sm text-indigo-700 hover:underline">← Back to search</a>
      <div className="grid gap-8 md:grid-cols-2">
        <div className="flex aspect-square items-center justify-center rounded-2xl border border-slate-200 bg-white p-8">
          <img src={p.image_url} alt={p.title} className="max-h-full max-w-full object-contain" />
        </div>
        <div className="space-y-5">
          <h1 className="text-2xl font-semibold">{p.title}</h1>
          <dl className="grid grid-cols-[120px_1fr] gap-y-2 text-sm">
            {ATTRS.filter((a) => p[a]).map((a) => (
              <div key={a} className="contents">
                <dt className="capitalize text-slate-500">{a === "color" ? "Colour" : a}</dt>
                <dd>{p[a]}</dd>
              </div>
            ))}
          </dl>
          {p.features.length > 0 && (
            <div>
              <h2 className="mb-2 font-medium">Features</h2>
              <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
                {p.features.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </div>
          )}
          <button
            onClick={() => onToggleCompare(p.id)}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          >
            {inCompare ? "Remove from comparison" : "Add to comparison"}
          </button>
          <p className="text-xs text-slate-500">
            Source: {p.source}, Amazon Berkeley Objects (CC BY 4.0). Prices and stock are not part of the dataset.
          </p>
        </div>
      </div>

      <section>
        <h2 className="mb-3 text-lg font-semibold">
          Visually similar <Badge>CLIP image embeddings</Badge>
        </h2>
        {!caps?.similar_products && <Empty title="Similar products are unavailable">Embeddings are not loaded on this server.</Empty>}
        {caps?.similar_products && similar.status === "loading" && <Spinner />}
        {similar.status === "error" && <ErrorBox error={similar.error} />}
        {similar.status === "ok" && similar.data && (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {similar.data.map((s) => (
              <ProductCard key={s.id} product={s} comparing={compare.includes(s.id)} onToggleCompare={onToggleCompare} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
