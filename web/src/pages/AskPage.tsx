import { useState, type FormEvent } from "react";

import { api, type AskResponse } from "../api";
import { splitCitations } from "../citations";
import { Badge, ErrorBox, Spinner } from "../components/ui";

const EXAMPLES = ["Which table lamps do you have?", "I need a red handbag", "Do you sell laptops?"];

export function AskPage() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<
    { status: "idle" } | { status: "loading" } | { status: "error"; error: unknown } | { status: "ok"; data: AskResponse }
  >({ status: "idle" });

  const ask = async (q: string) => {
    if (!q.trim()) return;
    setQuestion(q);
    setState({ status: "loading" });
    try {
      setState({ status: "ok", data: await api.ask(q.trim()) });
    } catch (error) {
      setState({ status: "error", error });
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    ask(question);
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Ask about the catalogue</h1>
        <p className="mt-1 text-sm text-slate-600">
          Answers come only from the five best-matching products. Every cited product is checked against what was
          retrieved; citations the model invents are removed.
        </p>
      </div>
      <form onSubmit={submit} className="flex gap-3">
        <label htmlFor="question" className="sr-only">Question</label>
        <input
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          maxLength={500}
          placeholder="Ask a question about the products"
          className="flex-1 rounded-lg border-slate-300 px-4 py-2.5"
        />
        <button
          disabled={state.status === "loading"}
          className="rounded-lg bg-indigo-600 px-5 py-2.5 font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          Ask
        </button>
      </form>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((q) => (
          <button key={q} onClick={() => ask(q)} className="rounded-full bg-white px-3 py-1 text-sm shadow-sm hover:bg-slate-100">
            {q}
          </button>
        ))}
      </div>

      {state.status === "loading" && <Spinner label="Thinking" />}
      {state.status === "error" && <ErrorBox error={state.error} />}
      {state.status === "ok" && <Answer data={state.data} />}
    </div>
  );
}

function Answer({ data }: { data: AskResponse }) {
  const byId = new Map(data.citations.map((c) => [c.id, c]));
  const parts = splitCitations(data.answer, new Set(byId.keys()));
  return (
    <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-5" aria-live="polite">
      <div className="flex flex-wrap gap-2">
        {data.grounded ? <Badge tone="green">Grounded in catalogue</Badge> : <Badge tone="amber">No product cited</Badge>}
        <Badge>retrieval: {data.retrieval_mode}</Badge>
        {data.dropped_citations.length > 0 && (
          <Badge tone="amber">{data.dropped_citations.length} invented citation(s) removed</Badge>
        )}
      </div>
      <p className="leading-relaxed">
        {parts.map((part, i) =>
          "cite" in part ? (
            <a key={i} href={`#/product/${part.cite}`} className="mx-0.5 rounded bg-indigo-50 px-1 text-xs font-medium text-indigo-700">
              {byId.get(part.cite)?.title.slice(0, 40)}
            </a>
          ) : (
            <span key={i}>{part.text}</span>
          ),
        )}
      </p>
      <details className="text-sm text-slate-600">
        <summary className="cursor-pointer">Retrieved products ({data.retrieved.length})</summary>
        <ul className="mt-2 space-y-1">
          {data.retrieved.map((r) => (
            <li key={r.id}>
              <a href={`#/product/${r.id}`} className="hover:text-indigo-700">{r.title}</a>{" "}
              <span className="text-slate-400">score {r.score}</span>
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
