"""Recompute every number in the README from results/, then redraw figure.png.

    python check.py

The numbers need only the standard library; the figure needs matplotlib.

Each tPD run is a folder of four files in results/. Three are the experiments' outputs as written; push.json is cut
down to the backup's own component, values unchanged. The files call the backup "graft" and its three matrices
graft.gate (gate), graft.c_fc (detector) and graft.down_proj (output), and count decompositions ("rounds") from 0.

  decomposition.json         settings ("args") and training record. Every run is the paper's recipe for its 4-block
                             numpy/pandas decomposition (same losses, coefficients and 30,000 steps) with 96
                             subcomponents per matrix and 4 per matrix of the backup, batches of 32 target and 32
                             other sequences, and the adversary on from 20% (160M) or 50% (2.8B) of training;
                             hidden_acts 100 with hidden_relative 3 is the added loss, ci_layerwise 1 the
                             per-module causal-importance network. A run repeats the decomposition on the edited
                             network, up to three in all, until under 5% of the answer is left on its target prompts;
                             a repeat starts from the decomposition before it and trains 9,000 steps.
  rounds.json                per decomposition: the subcomponents selected for ablation (causal importance above 0.1
                             on a target prompt and on at most 0.1% of other tokens), answer probabilities, and
                             graft_share, the share of the backup's weight they hold: sum_m <A_m, W_m> / sum_m |W_m|^2,
                             W_m the backup's matrices, A_m the selected subcomponents' summed rank-1 weights
  backup_subcomponents.json  per subcomponent of the backup: its largest causal importance at the last position of
                             the paper's two prompts (ci_bare) and of the backup's six (ci_exposing), whether it was
                             selected, its projection on its matrix (weight_projection); and Delta's projection
  push.json                  the backup's direct push on the answer's logit, relative to the vocabulary's mean
                             logit, with the final LayerNorm's scale held fixed, in the original and the edited network

results/backup-*.json record each backup as built: answer probabilities with it ("grafted") and without ("base").
The weak 2.8B backup has its output scaled down until it answers its six prompts at a median probability of 0.5;
at full strength it answers them above 0.999, saturated.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
MATRICES = {"graft.gate": "gate", "graft.c_fc": "detector", "graft.down_proj": "output"}

# group, model, seed, causal-importance network, run directory, backup record
RUNS = [
    ("needed", "160M", 0, "global", "pythia160m-graft-gate-exp-ppgd02-s0", "backup-pythia-160m"),
    ("needed", "160M", 1, "global", "pythia160m-graft-gate-exp-ppgd02-s1", "backup-pythia-160m"),
    ("needed", "160M", 0, "per-module", "pythia160m-graft-gate-exp-cilw-ppgd02-s0", "backup-pythia-160m"),
    ("needed", "160M", 1, "per-module", "pythia160m-graft-gate-exp-cilw-ppgd02-s1", "backup-pythia-160m"),
    ("needed", "2.8B", 0, "per-module", "pythia2.8b-graft-gatew-exp-cilw-ppgd05-s0", "backup-pythia-2.8b-weak"),
    ("needed", "2.8B", 1, "per-module", "pythia2.8b-graft-gatew-exp-cilw-ppgd05-s1", "backup-pythia-2.8b-weak"),
    ("needed+loss", "160M", 0, "per-module", "pythia160m-graft-gate-exp-hidrel3c100-cilw-ppgd02-s0", "backup-pythia-160m"),
    ("needed+loss", "160M", 1, "per-module", "pythia160m-graft-gate-exp-hidrel3c100-cilw-ppgd02-s1", "backup-pythia-160m"),
    ("unneeded", "2.8B", 0, "per-module", "pythia2.8b-graft-gate-cilw-ppgd05-s0", "backup-pythia-2.8b"),
    ("unneeded+loss", "2.8B", 0, "per-module", "pythia2.8b-graft-gate-hidrel3c100-cilw-ppgd05-s0", "backup-pythia-2.8b"),
]


def load(run, name):
    return json.loads((RES / run / f"{name}.json").read_text())


def measure(group, model, seed, ci_net, run, backup):
    record = load(run, "decomposition")
    args = record["args"]
    rounds = load(run, "rounds")
    subs = load(run, "backup_subcomponents")["rounds"]
    push = load(run, "push")["sets"]
    built = json.loads((RES / f"{backup}.json").read_text())
    order = [str(r) for r in rounds["rounds"]]
    prompts = [k for k in rounds["sets"] if k.startswith("exposing ")]
    words = [k.split("'")[1] for k in prompts]

    # settings the README states
    assert args["seed"] == seed and args["steps"] == 30000
    assert bool(args["ci_layerwise"]) == (ci_net == "per-module")
    assert (args["target"] == "exposing") == group.startswith("needed") and len(prompts) == 6
    assert args["model"] == {"backup-pythia-160m": "pythia-160m-graft-gate", "backup-pythia-2.8b": "pythia-2.8b-graft-gate",
                             "backup-pythia-2.8b-weak": "pythia-2.8b-graft-gatew"}[backup]
    with_loss = args.get("hidden_acts") == 100.0 and args.get("hidden_relative") == 3
    assert with_loss == group.endswith("+loss") and (with_loss or not args.get("hidden_acts"))

    per_matrix = []  # per decomposition: share of each matrix in the selected subcomponents
    detector_ci, detector_selected = 0.0, 0
    for rd in subs:
        row = {}
        for key, name in MATRICES.items():
            mine = [s for s in rd["graft"] if s["module"] == key]
            assert len(mine) == 4
            row[name] = sum(s["weight_projection"] for s in mine if s["selected"])
            if name == "detector":
                detector_ci = max([detector_ci] + [max(s["ci_bare"], s["ci_exposing"]) for s in mine])
                detector_selected += sum(s["selected"] for s in mine)
        per_matrix.append(row)
    return {
        "group": group, "model": model, "seed": seed, "ci_net": ci_net, "run": run, "backup": backup,
        "decompositions": len(order),
        "selected": [rounds["n_ablated"][r] for r in order],
        "adversary": [rd["diag"]["faithfulness"]["pgd"] for rd in record["rounds"]],
        "share": rounds["graft_share_cumulative"][order[-1]],
        "share_first": rounds["graft_share"][order[0]],
        "per_matrix": per_matrix,
        "detector_ci": detector_ci, "detector_selected": detector_selected,
        "detector_in_delta": [rd["graft_delta_projection"]["graft.c_fc"] for rd in subs],
        "words": words,
        "p_with": [rounds["sets"][k]["b"]["W"][0] for k in prompts],
        "p_without": [built["candidates"]["pandas"][w] for w in words],
        "silent_on_bare": all(u == [0.0, 0.0] for u in built["grafted_units"]["target"]),
        "pile_kl": built["pile_kl_grafted_vs_base"],
        "push_before": [push[k]["original"]["backup_push"][0] for k in prompts],
        "push_after": [push[k]["edited"]["backup_push"][0] for k in prompts],
        "p_after": [push[k]["edited"]["p"][0] for k in prompts],
        "push_numpy_after": push["target 'import numpy as'"]["edited"]["backup_push"][0],
    }


def pct(x, digits=3):
    return f"{100 * x:.{digits}f}%"


def span(xs, digits=2):
    return f"{min(xs):.{digits}f} to {max(xs):.{digits}f}"


def report(rows):
    print("share = share of the backup's weight in the selected subcomponents, over every decomposition of the run")
    print("push  = the backup's push on ' pd' over its six prompts, logits, before -> after the ablation")
    print("adv   = each decomposition's reconstruction loss under a fresh adversary, as recorded when it finished\n")
    print(f"{'':15s}{'model':6s}{'seed':5s}{'CI network':11s}{'selected':12s}{'share':>9s}  "
          f"{'gate / detector / output %, per decomposition':46s}{'detector CI':>11s}  {'push':29s}adv")
    for r in rows:
        mats = "; ".join("/".join(f"{100 * m[k]:.1f}" for k in ("gate", "detector", "output")) for m in r["per_matrix"])
        print(f"{r['group']:15s}{r['model']:6s}{r['seed']:<5d}{r['ci_net']:11s}{'+'.join(map(str, r['selected'])):12s}"
              f"{pct(r['share']):>9s}  {mats:46s}{r['detector_ci']:>11.2f}  "
              f"{span(r['push_before'], 1) + ' -> ' + span(r['push_after'], 1):29s}{', '.join(f'{a:.2f}' for a in r['adversary'])}")
    print()


def table(rows):
    """The README's table, one line per run."""
    push = lambda xs: "0" if max(xs) == 0.0 else span(xs, 1)  # noqa: E731
    lines = []
    for r in rows:
        model = r["model"] + (" \u2020" if r["group"].startswith("unneeded") else "")
        loss = "tPD's + hidden" if r["group"].endswith("+loss") else "tPD's"
        share = pct(r["share"], 0 if r["share"] > 0.1 else 1 if r["share"] > 0.005 else 3)
        lines.append(f"| {model} | {r['seed']} | {r['ci_net'].replace('-', ' ')} | {loss} | {share} | {push(r['push_before'])} "
                     f"| {push(r['push_after'])} | {max(r['p_after']):.2f} |")
    return lines


def check(rows):
    by = lambda g: [r for r in rows if r["group"] == g]  # noqa: E731
    needed, fixed = by("needed"), by("needed+loss")
    (plain28,), (fixed28,) = by("unneeded"), by("unneeded+loss")

    # the six prompts need the backup, and it is not saturated on them
    for r in needed + fixed:
        assert r["silent_on_bare"] and r["pile_kl"] == 0.0
    small, large = [r for r in needed + fixed if r["model"] == "160M"], [r for r in needed if r["model"] == "2.8B"]
    assert max(max(r["p_without"]) for r in small) < 0.002 and max(max(r["p_without"]) for r in large) < 0.0112
    assert all(0.27 < p < 0.92 for r in small for p in r["p_with"]) and all(0.22 < p < 0.93 for r in large for p in r["p_with"])
    weak = sorted(large[0]["p_with"])
    assert abs((weak[2] + weak[3]) / 2 - 0.5) < 0.001
    print(f"answer probability on the six prompts, 160M: model alone at most {max(max(r['p_without']) for r in small):.4f}, "
          f"with the backup {span(small[0]['p_with'])}")
    print(f"answer probability on the six prompts, 2.8B: model alone at most {max(max(r['p_without']) for r in large):.4f}, "
          f"with the weakened backup {span(large[0]['p_with'])}, median {(weak[2] + weak[3]) / 2:.2f}")

    # tPD's objective: six runs, ten decompositions, none of the detector, at most 0.012% of the backup
    assert len(needed) == 6 and sum(r["decompositions"] for r in needed) == 10
    assert all(r["detector_ci"] == 0.0 and r["detector_selected"] == 0 for r in needed)
    assert all(m["detector"] == 0.0 for r in needed for m in r["per_matrix"])
    top = max(r["share"] for r in needed)
    assert 0.00012 < top < 0.000125 and all(r["share"] <= top for r in needed)
    in_delta = [d for r in needed for d in r["detector_in_delta"]]
    assert 0.93 < min(in_delta) and max(in_delta) < 1.02
    print(f"tPD's objective, six runs, ten decompositions: detector causal importance {max(r['detector_ci'] for r in needed):.2f} "
          f"on every target prompt, none of it selected, its projection on \u0394 {span(in_delta)}; largest share {pct(top)}")
    gate_hits = sum(1 for r in needed if any(abs(m["gate"]) > 0 for m in r["per_matrix"]))
    output_hits = [(r["model"], r["seed"], r["ci_net"], i + 1) for r in needed for i, m in enumerate(r["per_matrix"]) if m["output"] > 0]
    assert gate_hits == 5 and output_hits == [("160M", 1, "per-module", 2)]
    print(f"  what it did select of the backup: a gate direction in {gate_hits} runs, an output direction in {output_hits}")
    alive = [r for r in needed if min(r["push_after"]) > 0]
    dead = [r for r in needed if max(r["push_after"]) == 0.0]
    assert len(alive) == 5 and len(dead) == 1 and (dead[0]["model"], dead[0]["seed"], dead[0]["ci_net"]) == ("160M", 1, "per-module")
    lo, hi = min(min(r["push_after"]) for r in alive), max(max(r["push_after"]) for r in alive)
    b_lo, b_hi = min(min(r["push_before"]) for r in needed), max(max(r["push_before"]) for r in needed)
    assert 3.4 < lo < 3.5 and 16.1 < hi < 16.2 and 5.5 < b_lo < 5.7 and 11.5 < b_hi < 11.6
    print(f"  after the ablation the backup still pushes ' pd' by {lo:.1f} to {hi:.1f} logits in five runs "
          f"(before: {b_lo:.1f} to {b_hi:.1f}); in the sixth, 0")
    taken = dead[0]["per_matrix"][1]["output"]
    assert round(taken, 2) == 0.19 and dead[0]["share"] < 0.0001
    print(f"  the sixth: its second decomposition selected {pct(taken, 0)} of the output matrix, {pct(dead[0]['share'])} of the "
          f"backup's weight")

    # with each module's own hidden-activation error x100: found, and the ablation silences it
    assert [round(r["share"], 3) for r in fixed] == [0.354, 0.250]
    assert all(r["detector_ci"] == 1.0 and r["detector_selected"] == 3 and r["decompositions"] == 1 for r in fixed)
    assert all(max(r["push_after"]) == 0.0 and max(r["p_after"]) < 1e-5 for r in fixed)
    assert all(r["push_numpy_after"] > 11 for r in fixed)
    assert [r["selected"] for r in fixed] == [[641], [649]] and [r["selected"][0] for r in needed[2:4]] == [186, 189]
    print(f"with the loss, 160M seeds 0 and 1: share {pct(fixed[0]['share'], 1)} and {pct(fixed[1]['share'], 1)}, detector causal "
          f"importance 1.00, push after the ablation 0 and answer probability under 0.00001 on all six prompts; subcomponents selected {fixed[0]['selected'][0]} and "
          f"{fixed[1]['selected'][0]} against {needed[2]['selected'][0]} and {needed[3]['selected'][0]}")
    print(f"  its numpy unit, which no target prompt needs, still pushes ' np' on 'import numpy as': "
          f"{fixed[0]['push_numpy_after']:.1f} and {fixed[1]['push_numpy_after']:.1f} logits")

    # 2.8B, full-strength backup that no target prompt needs, one seed each
    assert round(plain28["share"], 4) == 0.0121 and round(plain28["share_first"], 6) == 0.000083 and round(fixed28["share"], 3) == 0.304
    assert min(plain28["push_after"]) > 12.5 and max(fixed28["push_after"]) == 0.0 and fixed28["push_numpy_after"] == 0.0
    assert fixed28["decompositions"] == 1 and plain28["decompositions"] == 2
    print(f"2.8B, backup needed by no target prompt, seed 0: share {pct(plain28['share'], 1)} after two decompositions of tPD's "
          f"objective ({pct(plain28['share_first'])} after one), {pct(fixed28['share'], 1)} after one with the loss; "
          f"push after {span(plain28['push_after'], 1)} against {max(fixed28['push_after']):.0f}")
    readme = (HERE / "README.md").read_text()
    missing = [line for line in table(rows) if line not in readme]
    assert not missing, missing
    print("the README's table matches")
    print("\nall checks passed")


def figure(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    red, ink, g1, g2, g4 = "#E73121", "#111111", "#E6E6E6", "#C4C4C4", "#595959"
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    medium = {"fontfamily": "Helvetica Neue", "fontweight": "medium"}  # the fallbacks have no medium weight
    number = "zero one two three four five six seven eight nine ten".split()
    size = lambda m: float(m[:-1]) * {"M": 1e6, "B": 1e9}[m[-1]]  # noqa: E731
    models = lambda runs: " and ".join(f"Pythia-{m}" for m in sorted({r["model"] for r in runs}, key=size))  # noqa: E731

    tpd = [r for r in rows if r["group"] == "needed"]  # tPD's objective on the prompts that need the backup
    fix = [r for r in rows if r["group"] == "needed+loss"]  # the same test with the added loss
    top = max(r["share"] for r in tpd)

    W, H = 8.8, 3.12
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    inches = lambda t: ax.transData.inverted().transform(t.get_window_extent(renderer))  # noqa: E731

    x0, x1 = 0.15, W - 0.15  # 0 and 100% of the backup's weight
    bx = lambda v: x0 + (x1 - x0) * min(max(v, 0.0), 1.0)  # noqa: E731
    band, sep, pad = 0.18, 1.6, 0.08  # a run's band in inches, the white line between two runs in points, inset

    def heading(y, name, detail):
        t = ax.text(x0, y, name, fontsize=12.5, color=ink, va="baseline", **medium)
        ax.text(inches(t)[1][0] + 0.12, y, detail, fontsize=11.5, color=g4, va="baseline")

    def selected(y, value, color, va, after=""):  # "selected:" and its value, the value in the heavier face
        t = ax.text(x0 + pad, y, "selected:", fontsize=11.5, color=color, va=va, zorder=3)
        t = ax.text(inches(t)[1][0] + 0.05, y, value, fontsize=11.5, color=color, va=va, zorder=3, **medium)
        if after:
            ax.text(inches(t)[1][0] + 0.05, y, after, fontsize=11.5, color=color, va=va, zorder=3)

    def separators(n, yt):  # white lines between runs
        for i in range(1, n):
            ax.plot([x0, x1], [yt - i * band] * 2, color="white", lw=sep, solid_capstyle="butt", zorder=2)

    # tPD's objective: what it left unselected of the backup's weight, one band per run, from its share to 100%
    y = H - 0.3
    heading(y, "tPD's objective", f"{number[len(tpd)]} runs, {models(tpd)}")
    yt = top_y = y - 0.17
    for i, r in enumerate(tpd):
        ax.add_patch(Rectangle((bx(r["share"]), yt - (i + 1) * band), x1 - bx(r["share"]), band,
                               facecolor=red, edgecolor="none", zorder=1))
    yb = yt - len(tpd) * band
    label = dict(fontsize=15.5, color="white", fontweight="bold", va="center", zorder=3)
    t = ax.text((x0 + x1) / 2, (yt + yb) / 2, "the backup circuit, left unselected", ha="center", **label)
    probe = ax.text(0, 0, "left unselected", **label)
    (p0, _), (p1, _) = inches(probe)
    probe.remove()
    predicate = inches(t)[1][0] - (p1 - p0) / 2  # the middle of "left unselected", echoed in the grey below
    selected(yb - 0.18, f"at most {100 * top:.3f}%", ink, "baseline", after="in any run")

    # the same test with the added loss, the contrast in greys: the selected share dark, the rest light
    y = yb - 0.68
    heading(y, "tPD's objective plus each module's own hidden-activation error",
            f"{number[len({r['seed'] for r in fix})]} seeds, {models(fix)}")
    yt = y - 0.17
    for i, r in enumerate(fix):
        yr = yt - (i + 1) * band
        ax.add_patch(Rectangle((x0, yr), x1 - x0, band, facecolor=g1, edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((x0, yr), bx(r["share"]) - x0, band, facecolor=g4, edgecolor="none", zorder=1))
        selected(yr + band / 2, f"{100 * r['share']:.0f}%", "white", "center")
    ax.text(predicate, yt - band / 2, "left unselected", fontsize=11.5, color=g4, ha="center", va="center", zorder=3)
    separators(len(fix), yt)
    yb = yt - len(fix) * band

    # the scale, one for both: a guide at 100% from the top of the slab down to its label
    ax.plot([x1, x1], [top_y, yb - 0.1], color=g2, lw=0.7, solid_capstyle="butt", zorder=0)
    ax.text(x0, yb - 0.2, "0", fontsize=10.5, color=g4, va="center", ha="left")
    ax.text(x1, yb - 0.2, "100% of the backup circuit's weight", fontsize=10.5, color=g4, va="center", ha="right")

    fig.savefig(path, dpi=200, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {Path(path).name}")


if __name__ == "__main__":
    rows = [measure(*r) for r in RUNS]
    report(rows)
    check(rows)
    try:
        figure(rows, HERE / "figure.png")
    except ImportError:
        print("matplotlib is not installed: figure not redrawn")
