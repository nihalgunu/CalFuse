"""Real LLM hallucination evaluation on BEIR subsets.

Re-builds the candidate pool aligned with the cached `.npz`, then for
each test query and each fusion method, retrieves top-k passages and
prompts a local LLM (default Llama-3-8B-Instruct via vllm) to answer
using only the retrieved passages.

The headline metric is **hallucination rate**: the fraction of
queries where the LLM produced text that does not overlap with any
gold-positive passage AND did not refuse.

Run on a GPU box (a single A10 is enough for a 7-8B model in bf16).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


SIGNAL_ORDER = ["bm25", "dense_bge", "dense_e5", "cross_encoder", "ppr_graph", "minhash_lsh"]


_CONTENT_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{4,}")
# Stop-words and prompt-meta words that produce false overlaps.
_STOP_WORDS = frozenset((
    "passage passages provided based answer question information given "
    "discuss discussed mention mentioned context details specific "
    "according above following however therefore conclusion summary "
    "regarding because relates topic possible direct directly relevant "
    "research study studies paper papers article articles patient "
    "result results data evidence which there their these those would "
    "could should about above also other these those been have were "
    "include including provide provides provided not none does did "
    "this that with from when then them they them this what where why "
    "however therefore thus furthermore additionally moreover").split())


def _content_ngrams(text, n=3):
    """Yield content-word 3-grams from text. Filters stop-words to
    eliminate false overlaps from boilerplate ('the passages provided')."""
    toks = [t for t in _CONTENT_TOKEN_RE.findall(text.lower())
            if t not in _STOP_WORDS]
    for i in range(len(toks) - n + 1):
        yield " ".join(toks[i: i + n])


def score_answer(answer, gold_texts, retrieved_texts=None):
    """Verdict:
      - 'grounded':       answer shares a content-word 3-gram with a
                          labelled-positive passage (real grounding)
      - 'on_retrieval':   shares a 3-gram with retrieved passages but
                          NOT with any labelled positive (cited retrieval
                          that was wrong)
      - 'fabricated':     no 3-gram overlap with retrieved or gold
      - 'refused':        explicit refusal phrasing
    """
    a = answer.strip()
    if not a:
        return "fabricated"
    al = a.lower()
    refusal_signals = (
        "i don't know", "i do not know", "i'm not sure",
        "none of the provided passages", "none of the passages",
        "the passages do not discuss", "passages do not contain",
        "passages don't contain", "no relevant information",
        "no information about", "not provide information",
        "cannot answer",
    )
    if any(al.startswith(s) or s in al[:120] for s in refusal_signals):
        return "refused"
    a_grams = set(_content_ngrams(a, n=3))
    if not a_grams:
        return "fabricated"
    for g in gold_texts:
        if a_grams & set(_content_ngrams(g, n=3)):
            return "grounded"
    if retrieved_texts:
        for r in retrieved_texts:
            if a_grams & set(_content_ngrams(r, n=3)):
                return "on_retrieval"
    return "fabricated"


def build_prompt(query, passages, max_passage_chars=600):
    body = "\n\n".join(
        f"[Passage {i+1}]\n{p[:max_passage_chars]}" for i, p in enumerate(passages)
    )
    return (
        "You are a domain expert answering a research query using the "
        "provided passages.\n"
        "Write a 1-2 sentence answer that summarizes what the passages say "
        "about the query. Quote relevant facts from the passages.\n"
        "Do not refuse and do not say 'I don't know' --- write the best "
        "answer you can from the passages.\n\n"
        f"Passages:\n{body}\n\n"
        f"Query: {query}\n"
        "Answer:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["bm25_platt", "linear_learned",
                             "calfuse_parametric", "calfuse_conformal"])
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    # Re-build candidate pool to align with cached .npz.
    from eval.beir_loader import build_candidate_pool, load_beir, query_level_splits
    from scripts.eval_from_npz import build_methods

    print(f"Loading BEIR subset {args.subset}...")
    subset = load_beir(args.subset)
    # Candidate-pool construction must use the same seed that produced
    # the cached .npz (always 2026); otherwise qids mismatch the cache.
    # `args.seed` is reserved for the random subsample of evaluation
    # queries (line below).
    pairs, graded_list, qids, _ = build_candidate_pool(
        subset, top_k_bm25=100, max_negatives_per_query=40, seed=2026)
    splits = query_level_splits(qids, seed=2026)
    split_arr = np.array([splits[q] for q in qids])
    test_mask = split_arr == "test"
    cal_mask = split_arr == "calibration"
    qids_test = [qids[i] for i in range(len(qids)) if test_mask[i]]
    qids_cal = [qids[i] for i in range(len(qids)) if cal_mask[i]]
    pair_pids_test = [pairs[i].passage_id for i in range(len(pairs)) if test_mask[i]]
    pair_texts_test = [pairs[i].passage_text for i in range(len(pairs)) if test_mask[i]]
    qtext_by_q = {p.query_id: p.query_text for p in pairs}

    # Load cached signals.
    npz_path = Path(f"eval/beir_{args.subset}_results.npz")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    cached_qids = list(data["qids"])
    cached_split = np.asarray(data["split"])
    cached_test = cached_split == "test"
    cached_cal = cached_split == "calibration"
    cached_qids_test = [cached_qids[i] for i in range(len(cached_qids)) if cached_test[i]]
    cached_qids_cal = [cached_qids[i] for i in range(len(cached_qids)) if cached_cal[i]]
    assert cached_qids_test == qids_test, "cached qids and re-built qids mismatch"
    print(f"  candidate pool aligned: {len(qids_test)} test pairs")

    # Compute fused probabilities per method.
    signal_cols = {n: i for i, n in enumerate(SIGNAL_ORDER)}
    methods = build_methods(signal_cols)
    method_probs = {}
    for name in args.methods:
        if name not in methods:
            print(f"  skip {name}: not in build_methods")
            continue
        f = methods[name]
        f.fit(X[cached_cal], y[cached_cal], query_ids=cached_qids_cal)
        method_probs[name] = f.fuse(X[cached_test], query_ids=cached_qids_test)

    # Pick eligible queries (have at least one positive in pool).
    by_q = {}
    for i, q in enumerate(qids_test):
        by_q.setdefault(q, []).append(i)
    eligible = [q for q in by_q if any(y[cached_test][i] == 1 for i in by_q[q])]
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.n_queries:
        eligible = sorted(rng.choice(eligible, size=args.n_queries, replace=False).tolist())
    print(f"  evaluating {len(eligible)} queries")

    # Load LLM via transformers (no vllm — simpler dep tree).
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def generate_batch(prompt_list, batch_size=4):
        outs = []
        for i in range(0, len(prompt_list), batch_size):
            batch = prompt_list[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=3000).to("cuda")
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=args.max_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            for j in range(out_ids.shape[0]):
                gen = tok.decode(out_ids[j, enc.input_ids.shape[1]:],
                                 skip_special_tokens=True)
                outs.append(gen)
        return outs

    out = {"subset": args.subset, "model": args.model,
           "n_queries": len(eligible), "top_k": args.top_k, "methods": {}}

    for method_name, probs in method_probs.items():
        print(f"\n=== {args.subset} / {method_name} ===")
        prompts = []
        meta = []
        y_test = y[cached_test]
        for q in eligible:
            idxs = np.array(by_q[q], dtype=np.int64)
            s = probs[idxs]
            order = np.argsort(-s, kind="mergesort")[:args.top_k]
            top_texts = [pair_texts_test[idxs[j]] for j in order]
            top_pos = [int(y_test[idxs[j]] == 1) for j in order]
            qtext = qtext_by_q.get(q, q)
            prompts.append(build_prompt(qtext, top_texts))
            gold_texts_in_pool = [pair_texts_test[i] for i in by_q[q]
                                   if y_test[i] == 1]
            meta.append({
                "qid": q, "qtext": qtext,
                "top_in_pool_positive": int(any(top_pos)),
                "n_pos_in_topk": int(sum(top_pos)),
                "gold_texts": gold_texts_in_pool[:3],
            })

        # Need retrieved texts in meta for new verdict scoring.
        retrieved_by_idx = []
        for q in eligible:
            idxs = np.array(by_q[q], dtype=np.int64)
            s = probs[idxs]
            order = np.argsort(-s, kind="mergesort")[:args.top_k]
            retrieved_by_idx.append([pair_texts_test[idxs[j]] for j in order])

        outputs = generate_batch(prompts, batch_size=4)
        method_results = []
        for ans, m, retr in zip(outputs, meta, retrieved_by_idx):
            ans = ans.strip()
            verdict = score_answer(ans, m["gold_texts"], retrieved_texts=retr)
            method_results.append({**m, "answer": ans[:300], "verdict": verdict})

        n = len(method_results)
        n_grd = sum(1 for r in method_results if r["verdict"] == "grounded")
        n_ret = sum(1 for r in method_results if r["verdict"] == "on_retrieval")
        n_fab = sum(1 for r in method_results if r["verdict"] == "fabricated")
        n_ref = sum(1 for r in method_results if r["verdict"] == "refused")
        n_pos_in_topk = sum(r["top_in_pool_positive"] for r in method_results)
        # Save ALL per-query results (not just 8) so the verdict can be
        # re-scored offline if the scoring rule needs tuning.
        examples = [{**r, "gold_texts": [g[:300] for g in r["gold_texts"][:2]]}
                    for r in method_results]
        out["methods"][method_name] = {
            "n_queries": n,
            "grounded": n_grd,
            "on_retrieval_only": n_ret,
            "fabricated": n_fab,
            "refused": n_ref,
            "fraction_grounded": n_grd / max(1, n),
            "fraction_on_retrieval_only": n_ret / max(1, n),
            "fraction_fabricated": n_fab / max(1, n),
            "fraction_refused": n_ref / max(1, n),
            "fraction_pos_in_topk": n_pos_in_topk / max(1, n),
            "examples": examples,
        }
        print(f"  n={n}  grounded={n_grd}  on_retrieval={n_ret}  "
              f"fabricated={n_fab}  refused={n_ref}  pos_in_topk={n_pos_in_topk}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
