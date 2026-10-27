import { api } from "../api";
import { Empty, ErrorBox, Spinner } from "../components/ui";
import { useAsync } from "../useAsync";

interface Props {
  ids: string[];
  onRemove: (id: string) => void;
}

export function ComparePage({ ids, onRemove }: Props) {
  const ready = ids.length >= 2;
  const [state, retry] = useAsync(() => (ready ? api.compare(ids) : Promise.resolve(null)), [ids.join(",")]);

  if (!ready) {
    return (
      <Empty title="Pick at least two products to compare">
        Tick “Compare” on up to four products in <a href="#/" className="text-indigo-700 underline">search</a>.
      </Empty>
    );
  }
  if (state.status === "loading") return <Spinner />;
  if (state.status === "error") return <ErrorBox error={state.error} onRetry={retry} />;
  if (!state.data) return null;
  const { products, attributes, shared_features } = state.data;

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Compare products</h1>
      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="w-32 p-3 text-left font-medium text-slate-500">Product</th>
              {products.map((p) => (
                <th key={p.id} className="p-3 text-left align-top font-normal">
                  <img src={p.image_url} alt="" className="mb-2 h-28 w-full object-contain" />
                  <a href={`#/product/${p.id}`} className="line-clamp-2 font-medium hover:text-indigo-700">{p.title}</a>
                  <button onClick={() => onRemove(p.id)} className="mt-1 text-xs text-slate-500 hover:text-red-600">
                    Remove
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {attributes.map((row) => (
              <tr key={row.field} className="border-b border-slate-100">
                <th scope="row" className="p-3 text-left font-medium capitalize text-slate-500">
                  {row.field === "color" ? "Colour" : row.field}
                </th>
                {row.values.map((v, i) => (
                  <td key={i} className={`p-3 ${row.same ? "text-slate-500" : "font-medium"}`}>{v ?? "—"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-slate-600">
        Differences are shown in bold.{" "}
        {shared_features.length > 0 ? `Shared features: ${shared_features.join("; ")}.` : "No listed features in common."}
      </p>
    </div>
  );
}
