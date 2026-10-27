import { useEffect, useState } from "react";

import { api, type Capabilities } from "./api";
import { AskPage } from "./pages/AskPage";
import { ComparePage } from "./pages/ComparePage";
import { ProductPage } from "./pages/ProductPage";
import { SearchPage } from "./pages/SearchPage";

const MAX_COMPARE = 4;
const COMPARE_KEY = "pva.compare";

function useHashRoute(): string {
  const [hash, setHash] = useState(() => window.location.hash || "#/");
  useEffect(() => {
    const onChange = () => {
      setHash(window.location.hash || "#/");
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

function loadCompare(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(COMPARE_KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((x) => typeof x === "string").slice(0, MAX_COMPARE) : [];
  } catch {
    return [];
  }
}

export default function App() {
  const route = useHashRoute();
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [compare, setCompare] = useState<string[]>(loadCompare);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    api.capabilities().then(setCaps, () => setCaps(null));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(COMPARE_KEY, JSON.stringify(compare));
    } catch {
      // storage unavailable (private mode): comparison still works for this visit
    }
  }, [compare]);

  const toggleCompare = (id: string) => {
    setCompare((c) => {
      if (c.includes(id)) return c.filter((x) => x !== id);
      if (c.length >= MAX_COMPARE) {
        setNotice(`You can compare up to ${MAX_COMPARE} products.`);
        return c;
      }
      return [...c, id];
    });
  };

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3000);
    return () => clearTimeout(t);
  }, [notice]);

  const productMatch = route.match(/^#\/product\/([A-Za-z0-9]+)$/);
  let page;
  if (productMatch) {
    page = <ProductPage id={productMatch[1]} caps={caps} compare={compare} onToggleCompare={toggleCompare} />;
  } else if (route === "#/compare") {
    page = <ComparePage ids={compare} onRemove={toggleCompare} />;
  } else if (route === "#/ask") {
    page = <AskPage />;
  } else {
    page = <SearchPage caps={caps} compare={compare} onToggleCompare={toggleCompare} />;
  }

  const link = (href: string, label: string) => (
    <a
      href={href}
      aria-current={route === href ? "page" : undefined}
      className={`rounded-md px-3 py-2 text-sm font-medium ${route === href ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:text-slate-900"}`}
    >
      {label}
    </a>
  );

  return (
    <>
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <nav className="mx-auto flex max-w-6xl flex-wrap items-center gap-2 px-4 py-3">
          <a href="#/" className="mr-auto font-semibold text-indigo-700">Product Visual Agent</a>
          {link("#/", "Search")}
          {link("#/ask", "Ask")}
          {link("#/compare", `Compare (${compare.length})`)}
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">{page}</main>
      {notice && (
        <div role="status" className="fixed bottom-4 left-1/2 -translate-x-1/2 rounded-lg bg-slate-900 px-4 py-2 text-sm text-white shadow-lg">
          {notice}
        </div>
      )}
      <footer className="mx-auto max-w-6xl px-4 py-8 text-xs text-slate-500">
        Product data and images: Amazon Berkeley Objects, CC BY 4.0. Demonstration project, no real orders.
      </footer>
    </>
  );
}
