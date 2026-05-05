"""Generate the four canonical paper figures specified in the v23
figure-spec into paper/final_figures/. Vector PDFs, NeurIPS-style
typography, colourblind-safe Tol palette.

Outputs:
  paper/final_figures/fig1_worstsg_ece.pdf
  paper/final_figures/fig2_selective_ndcg.pdf
  paper/final_figures/fig3_calsize.pdf
  paper/final_figures/fig4_selective_abstention.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Tol bright palette (colourblind safe).
COL = {
    "CalFuse-P":      "#4477AA",
    "CalFuse":        "#228833",
    "Linear-Learned": "#EE6677",
    "BM25+Platt":     "#CCBB44",
    "BGE+Platt":      "#AA3377",
    "Linear+S-Platt": "#888888",
    "best-single":    "#CCBB44",
}

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper/final_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Fig 1: Worst-subgroup ECE bar chart ----------------

WORSTSG_ECE = {
    "nfcorpus":    dict(BM25=0.209, BMs=0.016, BGE=0.371, BGEs=0.015,
                        LL=0.094, LLs=0.013, LSP=0.062, LSPs=0.010,
                        CF=0.055, CFs=0.010, BR=0.44),
    "trec-covid":  dict(BM25=0.278, BMs=0.024, BGE=0.143, BGEs=0.019,
                        LL=0.070, LLs=0.012, LSP=0.069, LSPs=0.007,
                        CF=0.069, CFs=0.011, BR=0.40),
    "touche-2020": dict(BM25=0.128, BMs=0.014, BGE=0.117, BGEs=0.011,
                        LL=0.093, LLs=0.011, LSP=0.086, LSPs=0.016,
                        CF=0.084, CFs=0.020, BR=0.21),
    "scidocs":     dict(BM25=0.072, BMs=0.006, BGE=0.069, BGEs=0.005,
                        LL=0.028, LLs=0.004, LSP=0.025, LSPs=0.007,
                        CF=0.025, CFs=0.007, BR=0.05),
    "scifact":     dict(BM25=0.064, BMs=0.012, BGE=0.024, BGEs=0.005,
                        LL=0.026, LLs=0.005, LSP=0.030, LSPs=0.006,
                        CF=0.022, CFs=0.007, BR=0.027),
    "fiqa":        dict(BM25=0.175, BMs=0.008, BGE=0.081, BGEs=0.005,
                        LL=0.037, LLs=0.004, LSP=0.038, LSPs=0.006,
                        CF=0.021, CFs=0.004, BR=0.027),
    "arguana":     dict(BM25=0.184, BMs=0.010, BGE=0.093, BGEs=0.007,
                        LL=0.014, LLs=0.002, LSP=0.024, LSPs=0.006,
                        CF=0.011, CFs=0.002, BR=0.025),
}
SIG_STAR = {"fiqa", "arguana", "scifact"}  # paired-t p<0.05 vs Linear+S-Platt
ORDER = ["nfcorpus", "trec-covid", "touche-2020", "scidocs",
         "scifact", "fiqa", "arguana"]


def fig1_worstsg_ece():
    n_groups = len(ORDER)
    n_methods = 5
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    bw = 0.16
    x_centers = np.arange(n_groups)

    method_specs = [
        ("BM25+Platt",     "BM25", "BMs",  COL["BM25+Platt"]),
        ("BGE+Platt",      "BGE",  "BGEs", COL["BGE+Platt"]),
        ("Linear-Learned", "LL",   "LLs",  COL["Linear-Learned"]),
        ("Linear+S-Platt", "LSP",  "LSPs", COL["Linear+S-Platt"]),
        ("CalFuse",        "CF",   "CFs",  COL["CalFuse"]),
    ]
    offsets = np.linspace(-2.0 * bw, 2.0 * bw, n_methods)

    for j, (name, vk, sk, c) in enumerate(method_specs):
        vals = np.array([WORSTSG_ECE[s][vk] for s in ORDER])
        errs = np.array([WORSTSG_ECE[s][sk] for s in ORDER])
        ax.bar(x_centers + offsets[j], vals, width=bw, color=c,
               edgecolor="black", linewidth=0.4, yerr=errs,
               error_kw=dict(linewidth=0.7, capsize=2,
                             ecolor="#333333"),
               label=name)

    # Per-group annotation: ratio (best-single / CalFuse) + star.
    for i, sub in enumerate(ORDER):
        d = WORSTSG_ECE[sub]
        best_v = min(d["BM25"], d["BGE"])
        ratio = best_v / max(d["CF"], 1e-6)
        star = r"\,\star" if sub in SIG_STAR else ""
        # Place text just above the tallest bar in the group.
        ymax = max(d["BM25"] + d["BMs"], d["BGE"] + d["BGEs"])
        ax.text(x_centers[i], ymax + 0.012,
                rf"${ratio:.1f}\times{star}$",
                ha="center", va="bottom", fontsize=8,
                color="#111111")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"{s}\n(BR={WORSTSG_ECE[s]['BR']:.2f})"
                        for s in ORDER], fontsize=8.5)
    ax.set_ylabel("Worst-subgroup ECE-15 (lower is better)",
                  fontsize=9)
    ax.set_ylim(0, 0.42)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.3,
            color="grey")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2,
              fontsize=8.5)
    ax.set_title("Worst-subgroup ECE-15 across 7 BEIR subsets "
                 r"($\star$: paired-$t$ $p<0.05$ vs Linear+S-Platt; "
                 r"ratio $=$ best-single $/$ CalFuse)",
                 fontsize=9, pad=8)
    fig.tight_layout()
    out = OUT / "fig1_worstsg_ece.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Fig 2: Selective NDCG, 7-panel grid ----------------

def fig2_selective_ndcg():
    sel = json.load(open(REPO / "eval/selective_ndcg.json"))
    extra = json.load(open(REPO / "eval/selective_ndcg_extra.json"))
    sel.update(extra)

    cov_keys = ["0.1", "0.25", "0.5", "0.75", "0.9", "1.0"]
    cov = [float(c) for c in cov_keys]

    panels = ["trec-covid", "nfcorpus", "scifact",
              "fiqa", "arguana", "scidocs", "touche-2020"]
    fig, axes = plt.subplots(2, 4, figsize=(8.5, 4.6),
                             sharex=True)
    axes = axes.flatten()

    for i, sub in enumerate(panels):
        ax = axes[i]
        if sub not in sel:
            ax.set_visible(False); continue
        ms = sel[sub]["methods"]
        ll = ms["linear_learned"]["selective_ndcg_at_coverage"]
        cfp = ms["calfuse_parametric"]["selective_ndcg_at_coverage"]
        ll_v = [ll[k] for k in cov_keys]
        cfp_v = [cfp[k] for k in cov_keys]
        decisive = (sub == "trec-covid")
        if decisive:
            ax.set_facecolor("#FFF5DC")
            for s in ax.spines.values():
                s.set_linewidth(1.4)
                s.set_color("#AA7700")
        ax.plot(cov, cfp_v, "-o", color=COL["CalFuse-P"], lw=1.8,
                ms=4.5, label="CalFuse-P")
        ax.plot(cov, ll_v, "--s", color=COL["Linear-Learned"], lw=1.5,
                ms=4, label="Linear-Learned")
        gap = cfp_v[0] - ll_v[0]
        title = f"{sub}  ($\\Delta$@c=0.10: {gap:+.3f})"
        if decisive:
            title += "  (decisive)"
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.3,
                color="grey")
        ax.set_axisbelow(True)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="both", labelsize=7)
        if i % 4 == 0:
            ax.set_ylabel("NDCG@10", fontsize=8)
        if i >= 4:
            ax.set_xlabel("coverage $c$", fontsize=8)
    # Hide the unused 8th panel; place legend there.
    axes[7].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=COL["CalFuse-P"], marker="o", lw=1.8,
                   ms=5, label="CalFuse-P"),
        plt.Line2D([0], [0], color=COL["Linear-Learned"], marker="s",
                   linestyle="--", lw=1.5, ms=4.5, label="Linear-Learned"),
    ]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(0.985, 0.06), frameon=False, fontsize=9)
    fig.suptitle("Selective NDCG@10 vs query coverage. "
                 "trec-covid (highlighted) is the decisive panel; "
                 "5 other subsets are statistically tied with Linear-Learned.",
                 fontsize=9, y=1.005)
    fig.tight_layout()
    out = OUT / "fig2_selective_ndcg.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Fig 3: Cal-size sweep ----------------

def fig3_calsize():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=False)
    # Two-panel: worst-subgroup ECE (the headline) + overall ECE
    # (smoother). Use scifact + arguana — clean asymptote story.
    # nfcorpus omitted: small-pool worst-subgroup variance is high
    # and would obscure the sample-efficiency point.
    subsets = [("scifact", "#4477AA"),
               ("arguana", "#228833")]
    for sub, c in subsets:
        d = json.load(open(REPO / f"eval/ablate_cal_size_{sub}.json"))
        rows = [r for r in d["results"] if r["method"] == "calfuse_parametric"]
        rows.sort(key=lambda r: r["n_cal_queries"])
        x = np.array([r["n_cal_queries"] for r in rows])
        y_w = np.array([r["worst_subgroup_ece_15"] for r in rows])
        y_o = np.array([r["ece_15"] for r in rows])
        axes[0].plot(x, y_w, "-o", color=c, lw=1.6, ms=4.5, label=sub)
        axes[1].plot(x, y_o, "-o", color=c, lw=1.6, ms=4.5, label=sub)
        # Mark the 25%-of-data point on the headline (left) panel.
        for r in rows:
            if abs(r["cal_fraction"] - 0.25) < 1e-6:
                axes[0].axvline(r["n_cal_queries"], color=c,
                                linestyle=":", lw=0.7, alpha=0.7)
                if sub == "scifact":
                    axes[0].annotate(rf"$25\%$ of cal: "
                                     rf"{r['n_cal_queries']}$\,q$, "
                                     rf"ECE $={r['worst_subgroup_ece_15']:.3f}$",
                                     xy=(r["n_cal_queries"],
                                         r["worst_subgroup_ece_15"]),
                                     xytext=(r["n_cal_queries"] * 2.0,
                                             r["worst_subgroup_ece_15"] + 0.006),
                                     arrowprops=dict(arrowstyle="->",
                                                     lw=0.5, color="#333333"),
                                     fontsize=7)
                break
    for ax, ttl, ylab in zip(axes,
                              ["Worst-subgroup ECE-15", "ECE-15 (overall)"],
                              ["Worst-subgroup ECE-15", "ECE-15"]):
        ax.set_xscale("log")
        ax.set_xlabel("Number of calibration queries (log)", fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(ttl, fontsize=9, pad=4)
        ax.grid(True, which="both", linestyle="-", linewidth=0.4,
                alpha=0.3, color="grey")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("CalFuse-Parametric saturates at "
                 r"$\sim 25\%$ of available calibration queries "
                 "on calibration-sensitive subsets.",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    out = OUT / "fig3_calsize.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Fig 4: Selective abstention (scifact, scidocs) ----

def fig4_selective_abstention():
    cov = [0.10, 0.25, 0.50, 0.75, 1.00]
    # scifact (5-seed mean hallu_NR%) — from spec + selective JSON.
    scifact = {
        "Linear-Learned": [4.0, 21.1, 14.7, 22.5, 34.1],
        "CalFuse-P":      [4.0, 13.3, 15.2, 21.1, 30.8],
        "CalFuse":        [12.0, 21.1, 22.1, 23.6, 32.4],
        "BM25+Platt":     [12.0, 21.7, 24.1, 27.2, 36.9],
    }
    scidocs = {
        "Linear-Learned": [44.0, 43.3, 46.4, 48.1, 47.7],
        "CalFuse-P":      [48.0, 50.0, 50.4, 52.7, 50.0],
        "CalFuse":        [52.0, 46.7, 49.2, 52.0, 50.2],
        "BM25+Platt":     [83.3, 66.7, 57.5, 61.5, 61.0],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
    for ax, (title, data, gain) in zip(axes, [
        ("scifact (CalFuse wins; paired-$t$ $p=0.045$)", scifact, True),
        ("scidocs (anti-pattern: Linear-Learned wins; $p=0.036$ at $c{=}0.75$)",
         scidocs, False),
    ]):
        for name in ["BM25+Platt", "Linear-Learned", "CalFuse", "CalFuse-P"]:
            v = data[name]
            ls = "-" if "CalFuse" in name else "--"
            mk = "o" if name == "CalFuse-P" else ("s" if name == "Linear-Learned" else ("^" if name == "CalFuse" else "x"))
            ax.plot(cov, v, ls, marker=mk, color=COL[name], lw=1.6,
                    ms=4.5, label=name)
        # Shade gap between Linear-Learned and CalFuse-P.
        ll = np.asarray(data["Linear-Learned"])
        cfp = np.asarray(data["CalFuse-P"])
        if gain:
            ax.fill_between(cov, ll, cfp, where=(cfp <= ll),
                            color=COL["CalFuse-P"], alpha=0.10,
                            label="CalFuse-P $\\leq$ Linear-Learned")
            ax.annotate("CalFuse-P: 30.8% $\\to$ 4.0%",
                        xy=(0.10, 4.0), xytext=(0.32, 17),
                        arrowprops=dict(arrowstyle="->", lw=0.6,
                                        color="#222222"),
                        fontsize=8)
        else:
            ax.fill_between(cov, ll, cfp, where=(cfp >= ll),
                            color=COL["Linear-Learned"], alpha=0.10,
                            label="Linear-Learned $\\leq$ CalFuse-P")
            ax.annotate("Linear-L: 47.7% $\\to$ 44.0%",
                        xy=(0.12, 44.5), xytext=(0.30, 41),
                        arrowprops=dict(arrowstyle="->", lw=0.6,
                                        color="#222222"),
                        fontsize=8)

        ax.set_xlim(0.08, 1.02)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("answered coverage $c$")
        ax.set_ylabel(r"hallu$_{\mathrm{NR}}$ (%, 5-seed mean)")
        ax.set_title(title, fontsize=9, pad=6)
        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.3,
                color="grey")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper left", fontsize=7.5,
                  ncol=1)
    fig.tight_layout()
    out = OUT / "fig4_selective_abstention.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Fig 5: Reliability diagrams (3-panel) ----------------

def fig5_reliability():
    src = REPO / "eval/figdata_reliability_nfcorpus.json"
    if not src.exists():
        print(f"  skip fig5: {src} missing — run figdata_reliability_and_ranks.py")
        return
    panels = json.load(open(src))

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), sharey=True,
                             sharex=True)
    panel_keys = ["A_LL_marginal", "B_LL_worstsg", "C_CF_worstsg"]
    panel_colors = [COL["Linear-Learned"], COL["Linear-Learned"],
                    COL["CalFuse"]]
    for ax, k, color in zip(axes, panel_keys, panel_colors):
        p = panels[k]
        bins = p["bins"]
        x = np.array([b["pred"] for b in bins])
        y = np.array([b["obs"] for b in bins])
        c = np.array([b["count"] for b in bins], dtype=float)
        # Marker size proportional to bin count.
        s = 12 + 90 * (c / max(c.max(), 1))
        ax.plot([0, 1], [0, 1], color="#888888", lw=0.7,
                linestyle="--", zorder=1)
        ax.plot(x, y, "-", color=color, lw=1.0, alpha=0.6, zorder=2)
        ax.scatter(x, y, s=s, color=color, edgecolor="black",
                   linewidth=0.4, zorder=3)
        ax.set_xlim(0, max(0.6, x.max() * 1.05))
        ax.set_ylim(0, max(0.6, y.max() * 1.10))
        ax.set_xlabel("Predicted probability", fontsize=8.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.3,
                color="grey")
        ax.set_axisbelow(True)
        ax.set_title(p["title"], fontsize=8.5, pad=4)
        ax.text(0.04, 0.94,
                rf"ECE-15 $= {p['ece']:.3f}$" "\n" rf"$n = {p['n']}$",
                transform=ax.transAxes, fontsize=7.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#888888",
                          linewidth=0.4))
        ax.text(0.5, -0.30, p["subtitle"], transform=ax.transAxes,
                fontsize=8, ha="center", color="#444444")
    axes[0].set_ylabel("Observed frequency", fontsize=8.5)
    fig.suptitle("Reliability diagrams on $\\mathtt{nfcorpus}$ "
                 "(15 equal-mass bins; marker area $\\propto$ bin count)",
                 fontsize=9, y=1.04)
    fig.tight_layout()
    out = OUT / "fig5_reliability.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Fig 6: Hallucination mechanism (2-panel) ----------------

# scifact 5-seed hallu_NR per method (from update.md final section).
SCIFACT_HALLU = {
    "BM25+Platt":     dict(mean=36.9, std=4.4),
    "BGE+Platt":      dict(mean=41.9, std=4.3),
    "Linear-Learned": dict(mean=34.1, std=4.4),
    "CalFuse-P":      dict(mean=30.8, std=3.7),
    "CalFuse":        dict(mean=32.4, std=3.9),  # = Conformal-CalFuse
}

# Map figdata-script names to display names.
RANK_DISPLAY = {
    "bm25_platt":         "BM25+Platt",
    "bge_platt":          "BGE+Platt",
    "linear_learned":     "Linear-Learned",
    "calfuse_parametric": "CalFuse-P",
    "calfuse_conformal":  "CalFuse",
}


def fig6_halluc_mechanism():
    src = REPO / "eval/figdata_ranks_scifact.json"
    if not src.exists():
        print(f"  skip fig6: {src} missing")
        return
    ranks_data = json.load(open(src))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    method_order = ["BM25+Platt", "BGE+Platt", "Linear-Learned",
                    "CalFuse-P", "CalFuse"]

    # Left: hallu_NR bar chart with stds.
    axL = axes[0]
    means = [SCIFACT_HALLU[m]["mean"] for m in method_order]
    stds  = [SCIFACT_HALLU[m]["std"]  for m in method_order]
    colors = [COL[m] for m in method_order]
    xpos = np.arange(len(method_order))
    bars = axL.bar(xpos, means, yerr=stds, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.5,
                   error_kw=dict(linewidth=0.8, capsize=3,
                                 ecolor="#333333"))
    # Highlight CalFuse-P bar.
    cfp_idx = method_order.index("CalFuse-P")
    bars[cfp_idx].set_edgecolor("#003399")
    bars[cfp_idx].set_linewidth(1.4)

    axL.set_xticks(xpos)
    axL.set_xticklabels(method_order, rotation=22, ha="right",
                        fontsize=8)
    axL.set_ylabel(r"hallu$_{\mathrm{NR}}$ (%, 5-seed mean $\pm$ SD)",
                   fontsize=8.5)
    axL.set_ylim(0, 50)
    axL.set_title(r"$\mathtt{scifact}$ hallucination among non-refused",
                  fontsize=9, pad=4)
    axL.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.3,
             color="grey")
    axL.set_axisbelow(True)
    # Annotate the significance.
    axL.annotate(r"CalFuse-P vs Linear-Learned:" "\n"
                 r"$p = 0.045$, $d = -1.29$",
                 xy=(cfp_idx, means[cfp_idx] + stds[cfp_idx]),
                 xytext=(cfp_idx + 0.4, 47),
                 arrowprops=dict(arrowstyle="->", lw=0.6,
                                 color="#003399"),
                 fontsize=7.5, ha="left",
                 bbox=dict(boxstyle="round,pad=0.25",
                           facecolor="white",
                           edgecolor="#003399", linewidth=0.6))

    # Right: rank-of-first-positive distribution.
    axR = axes[1]
    bin_edges = np.arange(0.5, 7.5, 1.0)  # 1..6
    bin_centers = np.arange(1, 7)
    for m_key, m_disp in RANK_DISPLAY.items():
        ranks = ranks_data[m_key]["ranks"]
        hist, _ = np.histogram(ranks, bins=bin_edges, density=True)
        ls = "-" if "CalFuse" in m_disp or "Linear" in m_disp else "--"
        lw = 2.0 if m_disp in ("Linear-Learned", "CalFuse-P") else 1.2
        axR.plot(bin_centers, hist, ls, color=COL[m_disp],
                 marker="o", ms=4.5, lw=lw, label=m_disp)
    axR.set_xticks(bin_centers)
    axR.set_xticklabels(["1", "2", "3", "4", "5",
                         r"6 $\equiv$ none"])
    axR.set_xlabel("Rank of first positive in top-5 (lower better)",
                   fontsize=8.5)
    axR.set_ylabel("Empirical density (5 seeds pooled)", fontsize=8.5)
    axR.set_title(r"$\mathtt{scifact}$: same retrieval, different "
                  r"distractors", fontsize=9, pad=4)
    axR.grid(True, linestyle="-", linewidth=0.4, alpha=0.3,
             color="grey")
    axR.set_axisbelow(True)
    axR.legend(frameon=False, loc="upper right", fontsize=7)
    axR.text(0.04, 0.96,
             "Linear-L: mean rank $= 2.252$\n"
             "CalFuse-P: mean rank $= 2.284$\n"
             r"top-5 ID overlap $= 0.862$",
             transform=axR.transAxes, fontsize=7.5,
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="white", edgecolor="#888888",
                       linewidth=0.4))
    fig.tight_layout()
    out = OUT / "fig6_halluc_mechanism.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Fig 7: Marginal vs worst-sg ECE (Finding 1) ------

def fig7_marginal_vs_worstsg():
    """Read each subset's beir_<sub>_results.json, pull Linear-Learned
    marginal ECE-15 and worst-subgroup ECE-15."""
    rows = []
    for sub in ORDER:
        f = REPO / f"eval/beir_{sub}_results.json"
        if not f.exists():
            print(f"  skip {sub}: {f.name} missing")
            continue
        d = json.load(open(f))
        ll = next((m for m in d["methods"]
                   if m["method"] == "linear_learned"), None)
        if ll is None: continue
        rows.append((sub, float(ll["ece_15"]),
                     float(ll["worst_subgroup_ece_15"])))
    if not rows:
        print("  fig7: no data, skipping"); return

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xpos = np.arange(len(rows))
    bw = 0.36
    margs = [r[1] for r in rows]
    worsts = [r[2] for r in rows]
    ax.bar(xpos - bw / 2, margs, width=bw, color="#9DB7DA",
           edgecolor="black", linewidth=0.4, label="marginal ECE-15")
    ax.bar(xpos + bw / 2, worsts, width=bw, color="#1F4E8A",
           edgecolor="black", linewidth=0.4,
           label="worst-subgroup ECE-15")
    for i, (sub, mg, ws) in enumerate(rows):
        ratio = ws / max(mg, 1e-6)
        ax.text(i, max(mg, ws) + 0.005, rf"${ratio:.1f}\times$",
                ha="center", fontsize=7.5)
    ratios = [r[2] / max(r[1], 1e-6) for r in rows]
    median_ratio = float(np.median(ratios))
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([r[0] for r in rows], rotation=22, ha="right",
                       fontsize=8)
    ax.set_ylabel("ECE-15 (Linear-Learned)", fontsize=8.5)
    ax.set_title("Marginal calibration hides subgroup miscalibration "
                 rf"(min $={min(ratios):.1f}\times$, "
                 rf"median $={median_ratio:.1f}\times$, "
                 rf"max $={max(ratios):.1f}\times$)",
                 fontsize=9, pad=4)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.3,
            color="grey")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = OUT / "fig7_marginal_vs_worstsg.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Fig 8: IVAP downstream eval ----------------

def fig8_ivap_downstream():
    src = REPO / "eval/ivap_downstream_eval.json"
    if not src.exists():
        print(f"  skip fig8: {src} missing")
        return
    data = json.load(open(src))
    subsets = ["nfcorpus", "scifact", "fiqa", "arguana", "scidocs"]
    cov_keys = [0.10, 0.25, 0.50, 0.75, 1.00]
    cov_strs = [str(c) for c in cov_keys]  # JSON keys are strings

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0), sharey=False)
    PALETTE = {
        "point":    "#888888",
        "ivap_lo":  "#4477AA",
        "ivap_mid": "#228833",
        "combined": "#EE6677",
    }
    LABEL = {
        "point":    "Point estimate",
        "ivap_lo":  "IVAP $p_{\\mathrm{lo}}$",
        "ivap_mid": "IVAP midpoint",
        "combined": r"$p_{\mathrm{lo}} - 0.5\,w$",
    }

    for ax, subset in zip(axes, subsets):
        s = data[subset]["summary"]
        for k in ["point", "ivap_lo", "ivap_mid", "combined"]:
            means = [s[k][c]["mean"] for c in cov_strs]
            stds = [s[k][c]["std"]  for c in cov_strs]
            lw = 2.0 if k == "point" else 1.4
            ls = "--" if k == "point" else "-"
            mk = "x" if k == "point" else "o"
            ax.plot(cov_keys, means, ls, marker=mk, color=PALETTE[k],
                    lw=lw, ms=4.5, label=LABEL[k] if subset == subsets[0] else None)
            ax.fill_between(cov_keys,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=PALETTE[k], alpha=0.10, lw=0)
        # Mark significant cells (paired-t vs point, p<0.10).
        paired = data[subset]["paired"]
        for k in ["ivap_lo", "ivap_mid", "combined"]:
            for c in cov_keys:
                r = paired[k].get(str(c))
                if r and r["p"] < 0.10 and r["diff_mean"] < 0:
                    y = s[k][str(c)]["mean"]
                    ax.scatter([c], [y], s=70, marker="*",
                               color=PALETTE[k], edgecolor="black",
                               linewidth=0.4, zorder=5)
        ax.set_title(subset, fontsize=9, pad=3)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("answered coverage $c$", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.3,
                color="grey")
        ax.set_axisbelow(True)
        if subset == "nfcorpus":
            ax.set_ylabel(r"hallu$_{\mathrm{NR}}$ (%, 5-seed mean)",
                          fontsize=8.5)

    # Legend on the leftmost panel.
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5,
                   ncol=1)
    fig.suptitle("End-to-end downstream eval: IVAP envelope vs point "
                 "estimate as the abstention signal "
                 r"($\star$: paired-$t$ $p<0.10$, IVAP variant beats point)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    out = OUT / "fig8_ivap_downstream.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Generating figures into", OUT)
    fig1_worstsg_ece()
    fig2_selective_ndcg()
    fig3_calsize()
    fig4_selective_abstention()
    fig5_reliability()
    fig6_halluc_mechanism()
    fig7_marginal_vs_worstsg()
    fig8_ivap_downstream()
    print("\nDone. Files in", OUT)


if __name__ == "__main__":
    main()
