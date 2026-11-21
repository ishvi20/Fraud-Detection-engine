"""
Offline evaluation of the detection engine.

    python evaluate.py            # full run (~2 min), writes ../docs/RESULTS.md and ../docs/pr_curve.png
    python evaluate.py --quick    # smaller streams, for a fast smoke test

What it measures
  1. Ablation   rules only | Isolation Forest only | fused hybrid (the engine) | supervised GBM | GBM + rules
  2. Prevalence the same comparison at ~3% fraud (demo) and ~0.4% fraud (closer to reality)
  3. Novelty    leave-one-attack-out: the supervised model never sees one attack type in training,
                then is asked to catch it. This is where unsupervised signals and rules earn their keep.

All data is synthetic (see simulator.py), and train/test streams use different seeds and different
simulated customers. The numbers show how the pipeline behaves, not real-world accuracy.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from engine import FeatureStore, FraudEngine
from simulator import SCENARIOS, Simulator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

START_TS = 1_700_000_000  # fixed so results are reproducible
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
METHODS = ["Rules only", "Isolation Forest only", "Hybrid (engine)", "Supervised GBM", "GBM + rules"]


def run_stream(engine: FraudEngine, n: int, seed: int, fraud_rate: float):
    """Score a fresh simulated stream. Returns dict of aligned numpy arrays plus latency stats."""
    engine.store = FeatureStore()  # fresh customer state for every stream
    sim = Simulator(seed=seed, fraud_rate=fraud_rate, start_ts=START_TS)
    X, ml, rule, fused, y, scen, lat = [], [], [], [], [], [], []
    t0 = time.perf_counter()
    done = 0
    while done < n:
        batch = []
        for _ in range(min(8, n - done)):
            t = sim.next_txn()
            t["_t_in"] = time.perf_counter()
            batch.append(t)
        results = engine._score_batch(batch)
        end = time.perf_counter()
        for t, r in zip(batch, results):
            X.append(r["_x"])
            ml.append(r["ml_score"])
            rule.append(r["rule_score"])
            fused.append(r["score"])
            y.append(bool(r["_truth"]))
            scen.append(r["_scenario"] or "")
            lat.append((end - t["_t_in"]) * 1000)
        done += len(batch)
    return dict(X=np.array(X), ml=np.array(ml), rule=np.array(rule), fused=np.array(fused), y=np.array(y),
                scen=np.array(scen), lat=np.array(lat), seconds=time.perf_counter() - t0)


def fuse(engine, p_ml, rule):
    return 1 - (1 - engine.ML_WEIGHT * p_ml) * (1 - engine.RULE_WEIGHT * rule)


def fit_gbm(X, y):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X, y)


def method_scores(engine, d, gbm):
    g = gbm.predict_proba(d["X"])[:, 1]
    return {
        "Rules only": d["rule"],
        "Isolation Forest only": d["ml"],
        "Hybrid (engine)": d["fused"],
        "Supervised GBM": g,
        "GBM + rules": fuse(engine, g, d["rule"]),
    }


def recall_at_precision(y, s, target=0.8):
    p, r, _ = precision_recall_curve(y, s)
    ok = r[p >= target]
    return float(ok.max()) if ok.size else 0.0


def threshold_at_fpr(y, s, fpr_target=0.01):
    fpr, _, thr = roc_curve(y, s)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(thr[max(idx, 0)])


def summarize(y, s):
    return dict(
        roc=roc_auc_score(y, s),
        pr=average_precision_score(y, s),
        r80=recall_at_precision(y, s, 0.8),
        r_fpr=float((s[y] >= threshold_at_fpr(y, s, 0.01)).mean()),
    )


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller streams for a fast smoke test")
    args = ap.parse_args()
    n_train, n_test = (8000, 6000) if args.quick else (40000, 30000)

    print("Training Isolation Forest ...")
    engine = FraudEngine()
    engine.train(start_ts=START_TS - 3 * 86400)

    print(f"Building labelled training stream ({n_train:,} txns) for the supervised baseline ...")
    tr = run_stream(engine, n_train, seed=101, fraud_rate=0.04)
    gbm = fit_gbm(tr["X"], tr["y"])

    sections = []
    # ---- 1 + 2: ablation at two prevalences ----
    plot_data = None
    for label, rate in (("Demo prevalence", 0.04), ("Low prevalence", 0.005)):
        print(f"Evaluating on {n_test:,} test txns ({label}) ...")
        te = run_stream(engine, n_test, seed=2024, fraud_rate=rate)
        sc = method_scores(engine, te, gbm)
        rows = []
        for m in METHODS:
            s = summarize(te["y"], sc[m])
            rows.append([f"**{m}**" if m == "Hybrid (engine)" else m, f"{s['roc']:.3f}", f"{s['pr']:.3f}",
                         f"{s['r80']:.2f}", f"{s['r_fpr']:.2f}"])
        n_f = int(te["y"].sum())
        sections.append(f"### {label}: {n_test:,} transactions, {n_f:,} fraudulent ({100 * te['y'].mean():.2f}%)\n\n"
                        + md_table(["Method", "ROC-AUC", "PR-AUC", "Recall @ 80% precision", "Recall @ 1% false-positive rate"], rows))
        if rate == 0.04:
            plot_data = (te["y"], sc)
            demo = te

    # ---- operating point of the shipped engine ----
    y, s = demo["y"], demo["fused"]
    rows = []
    for thr in (0.4, 0.5, 0.6, 0.65, 0.7, 0.8):
        p = s >= thr
        tp = int((p & y).sum())
        prec = tp / p.sum() if p.sum() else 0.0
        rec = tp / y.sum()
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append([f"{thr:.2f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{int(p.sum()):,}"])
    sections.append("### Hybrid engine: threshold sweep (demo prevalence)\n\n"
                    + md_table(["Threshold", "Precision", "Recall", "F1", "Flagged"], rows))

    # ---- per-attack recall at the default threshold ----
    rows = []
    for sc_name in SCENARIOS:
        m = (demo["scen"] == sc_name) & y
        rows.append([sc_name, f"{int(m.sum()):,}", f"{(demo['fused'][m] >= engine.threshold).mean():.2f}"])
    sections.append(f"### Recall by attack type (hybrid, threshold {engine.threshold})\n\n" + md_table(["Attack", "Fraud txns", "Recall"], rows))

    # ---- 3: leave-one-attack-out ----
    print("Leave-one-attack-out ...")
    legit_thr = {}
    for m in METHODS:
        legit_thr[m] = None
    rows = []
    for held in SCENARIOS:
        keep = ~((tr["scen"] == held) & tr["y"])
        g_held = fit_gbm(tr["X"][keep], tr["y"][keep])
        sc = method_scores(engine, demo, g_held)
        mask = (demo["scen"] == held) & demo["y"]
        row = [held, f"{int(mask.sum()):,}"]
        for m in ("Rules only", "Isolation Forest only", "Hybrid (engine)", "Supervised GBM"):
            thr = threshold_at_fpr(demo["y"], sc[m], 0.01)
            row.append(f"{(sc[m][mask] >= thr).mean():.2f}")
        rows.append(row)
    sections.append(
        "### Novel attacks: recall when the attack type was never seen in training (1% false-positive rate)\n\n"
        "The supervised model is retrained with the named attack removed from its training labels, then tested on that attack. "
        "Rules and the Isolation Forest do not use labels, so their columns are unaffected.\n\n"
        + md_table(["Held-out attack", "Fraud txns", "Rules only", "Isolation Forest only", "Hybrid (engine)", "Supervised GBM"], rows))

    lat = demo["lat"]
    sections.append(f"### Speed (single CPU core, micro-batches of 8)\n\nMean latency {lat.mean():.1f} ms, p95 {np.percentile(lat, 95):.1f} ms "
                    f"per transaction; throughput {len(lat) / demo['seconds']:,.0f} txn/s.")

    os.makedirs(DOCS, exist_ok=True)
    header = ("# Evaluation results\n\nGenerated by `python evaluate.py` (fixed seeds, so rerunning gives the same numbers). "
              f"Train stream: {n_train:,} txns; test streams: {n_test:,} txns, different seed and different simulated customers.\n\n"
              "> **Read this honestly.** The data is synthetic and the fraud patterns come from the same generator the rules were written "
              "against, so scores are optimistic. What transfers is the *comparison*: how each component contributes, how precision "
              "falls as fraud gets rarer, and how a supervised model behaves on attacks it has never seen.\n\n")
    with open(os.path.join(DOCS, "RESULTS.md"), "w") as f:
        f.write(header + "\n\n".join(sections) + "\n")

    # ---- PR curve ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y, sc = plot_data
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
        colors = {"Rules only": "#9aa4b2", "Isolation Forest only": "#e0a030", "Hybrid (engine)": "#d6336c",
                  "Supervised GBM": "#2f7fd8", "GBM + rules": "#2a9d6f"}
        for m in METHODS:
            p, r, _ = precision_recall_curve(y, sc[m])
            ax.plot(r, p, label=f"{m} (AP {average_precision_score(y, sc[m]):.2f})", color=colors[m],
                    lw=2.4 if m == "Hybrid (engine)" else 1.5)
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision-recall, demo prevalence")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "pr_curve.png"))
    except ImportError:
        print("matplotlib not installed; skipping plot")
    print(f"Wrote {os.path.normpath(os.path.join(DOCS, 'RESULTS.md'))}")


if __name__ == "__main__":
    main()
