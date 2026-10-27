import type { ReactNode } from "react";

import type { Product, ScoredProduct } from "../api";

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center justify-center gap-3 py-16 text-slate-500">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />
      <span>{label}…</span>
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-800">
      <p className="font-medium">Something went wrong</p>
      <p className="mt-1 text-sm">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-3 rounded-md bg-red-600 px-3 py-1.5 text-sm text-white hover:bg-red-700">
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white p-10 text-center">
      <p className="font-medium text-slate-700">{title}</p>
      {children && <div className="mt-2 text-sm text-slate-500">{children}</div>}
    </div>
  );
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "green" | "amber" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-700",
    green: "bg-emerald-100 text-emerald-800",
    amber: "bg-amber-100 text-amber-800",
  };
  return <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>;
}

interface CardProps {
  product: Product | ScoredProduct;
  comparing: boolean;
  onToggleCompare: (id: string) => void;
}

export function ProductCard({ product, comparing, onToggleCompare }: CardProps) {
  return (
    <article className="group flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm transition hover:shadow-md">
      <a href={`#/product/${product.id}`} className="block aspect-square bg-white p-4">
        <img
          src={product.image_url}
          alt={product.title}
          loading="lazy"
          className="h-full w-full object-contain transition group-hover:scale-[1.03]"
        />
      </a>
      <div className="flex flex-1 flex-col gap-2 border-t border-slate-100 p-3">
        <div className="flex flex-wrap gap-1">
          <Badge>{product.category}</Badge>
          {product.color && <Badge>{product.color}</Badge>}
        </div>
        <a href={`#/product/${product.id}`} className="line-clamp-2 text-sm font-medium hover:text-indigo-700">
          {product.title}
        </a>
        {product.brand && <p className="text-xs text-slate-500">{product.brand}</p>}
        <label className="mt-auto flex cursor-pointer items-center gap-2 pt-1 text-xs text-slate-600">
          <input
            type="checkbox"
            checked={comparing}
            onChange={() => onToggleCompare(product.id)}
            className="h-4 w-4 rounded border-slate-300 text-indigo-600"
          />
          Compare
        </label>
      </div>
    </article>
  );
}
