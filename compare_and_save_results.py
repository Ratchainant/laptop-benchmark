"""
compare_results.py — เปรียบเทียบผลจากหลายเครื่องและคำนวณ Normalized Score

วิธีใช้:
  python compare_results.py macbook_air.json mac_mini.json dell_pro.json
  python compare_results.py *.json --md results.md
"""

import json
import sys
import argparse
import statistics
from datetime import datetime
from pathlib import Path

# ── color ─────────────────────────────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"

def pr(c, m): print(f"{c}{m}{C.RESET}")
def header(m): pr(C.BOLD + C.HEADER, f"\n{'═'*70}\n  {m}\n{'═'*70}")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def short_name(data: dict) -> str:
    cpu      = data["platform"].get("cpu", "")
    hostname = data["platform"].get("hostname", "")
    if "M4" in cpu or "M4" in hostname:
        return "Mac Mini M4"
    if "M3" in cpu or "M3" in hostname:
        return "MacBook Air M3"
    if "Ryzen" in cpu or "AMD" in cpu:
        return "Dell Pro (Ryzen AI)"
    return hostname[:20] or cpu[:20]


def get_test_result(data: dict, test_idx: int) -> dict | None:
    try:
        r = data["results"][test_idx]
        return r if r.get("status") == "ok" else None
    except (IndexError, KeyError):
        return None


def normalize(val: float, best: float, higher_is_better: bool) -> float:
    if best == 0:
        return 0.0
    return min((val / best * 100) if higher_is_better else (best / val * 100), 100)


def bar_terminal(score: float, width: int = 20) -> str:
    filled = max(0, min(round(score / 100 * width), width))
    color  = C.GREEN if score >= 80 else (C.YELLOW if score >= 50 else C.RED)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{C.RESET}"


def bar_md(score: float, width: int = 20) -> str:
    filled = max(0, min(round(score / 100 * width), width))
    return "█" * filled + "░" * (width - filled)


TESTS = [
    ("LLM Inference",                          True,  0.30),
    ("Image Classification (ONNX ResNet-50)",  True,  0.20),
    ("Pandas groupby + join (5M rows)",         False, 0.15),
    ("DuckDB analytical query (50M rows)",      False, 0.15),
    ("Sequential I/O Read",                     True,  0.10),
    ("NumPy matrix multiply",                   True,  0.10),
]

CATEGORIES = {
    "AI Inference":     [0, 1],
    "Data / Analytics": [2, 3],
    "I/O + Memory":     [4, 5],
}


def collect(datasets: list[dict]) -> tuple:
    """คืน (test_medians, machine_scores, machine_weights, final_scores, order)"""
    test_medians = []
    for idx, (_, hib, _) in enumerate(TESTS):
        medians = []
        for d in datasets:
            r = get_test_result(d, idx)
            medians.append(r["median"] if r else None)
        test_medians.append((medians, hib))

    machine_scores  = [0.0] * len(datasets)
    machine_weights = [0.0] * len(datasets)

    for idx, (_, hib, weight) in enumerate(TESTS):
        medians, _ = test_medians[idx]
        valid = [m for m in medians if m is not None]
        if not valid:
            continue
        best = max(valid) if hib else min(valid)
        for i, m in enumerate(medians):
            if m is not None:
                machine_scores[i]  += normalize(m, best, hib) * weight
                machine_weights[i] += weight

    final_scores = [
        round(machine_scores[i] / machine_weights[i], 1) if machine_weights[i] > 0 else 0.0
        for i in range(len(datasets))
    ]
    order = sorted(range(len(datasets)), key=lambda i: final_scores[i], reverse=True)
    return test_medians, machine_scores, machine_weights, final_scores, order


# ── terminal output ────────────────────────────────────────────────────────────
def print_terminal(datasets, test_medians, final_scores, order):
    header("CROSS-MACHINE BENCHMARK COMPARISON")
    print()
    for d in datasets:
        p = d["platform"]
        pr(C.BOLD, f"  {d['_label']}")
        pr(C.DIM,  f"    CPU: {p.get('cpu','?')[:55]}")
        pr(C.DIM,  f"    RAM: {d.get('ram_gb','?')} GB   |   Recorded: {d.get('timestamp','?')[:19]}")
        print()

    header("PER-TEST RESULTS")
    for idx, (test_name, hib, weight) in enumerate(TESTS):
        medians, _ = test_medians[idx]
        valid = [m for m in medians if m is not None]
        if not valid:
            print(f"\n  {test_name}  — all skipped\n")
            continue
        best   = max(valid) if hib else min(valid)
        scores = [normalize(m, best, hib) if m is not None else None for m in medians]
        unit   = next((datasets[i]["results"][idx].get("unit","") for i in range(len(datasets)) if get_test_result(datasets[i], idx)), "")
        direction = "↑ higher better" if hib else "↓ lower better"
        pr(C.BOLD + C.CYAN, f"\n  {test_name}   [{weight*100:.0f}%]   {direction}")
        print(f"  {'─'*66}")
        for i, d in enumerate(datasets):
            m = medians[i]
            s = scores[i]
            if m is None:
                print(f"  {'':1} {d['_label']:<22}  {'skipped':>14}")
                continue
            medal = f"{C.GREEN}★{C.RESET}" if m == best else " "
            print(f"  {medal} {d['_label']:<22}  {f'{m:.2f} {unit}':>14}   {bar_terminal(s)}  {s:.1f}/100")

    header("FINAL NORMALIZED SCORES (Weighted)")
    print()
    print(f"  {'Rank':<5}  {'Machine':<25}  {'Score':>7}   {'':20}   {'Tests run'}")
    print(f"  {'─'*5}  {'─'*25}  {'─'*7}   {'─'*20}   {'─'*9}")
    for rank, i in enumerate(order, 1):
        medal = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"
        ran   = sum(1 for idx in range(len(TESTS)) if get_test_result(datasets[i], idx))
        print(f"  {medal:<5}  {datasets[i]['_label']:<25}  {final_scores[i]:>6.1f}   {bar_terminal(final_scores[i], 20)}   {ran}/{len(TESTS)}")

    print()
    pr(C.DIM, "  Formula: score_i = (result_i / best) × 100  (higher-is-better)")
    pr(C.DIM, "           score_i = (best / result_i) × 100  (lower-is-better)")
    pr(C.DIM, "           final   = Σ(score_i × weight_i) / Σ(weight_i)")

    header("CATEGORY BREAKDOWN")
    for cat_name, test_indices in CATEGORIES.items():
        pr(C.BOLD, f"\n  {cat_name}")
        print(f"  {'─'*50}")
        cat_scores = []
        for i, d in enumerate(datasets):
            total, tw = 0.0, 0.0
            for idx in test_indices:
                medians, hib = test_medians[idx]
                valid = [m for m in medians if m is not None]
                if not valid or medians[i] is None:
                    continue
                best = max(valid) if hib else min(valid)
                total += normalize(medians[i], best, hib) * TESTS[idx][2]
                tw    += TESTS[idx][2]
            cat_scores.append((d["_label"], round(total / tw, 1) if tw > 0 else 0.0))
        for label, s in sorted(cat_scores, key=lambda x: x[1], reverse=True):
            print(f"  {label:<25}  {s:>6.1f}/100  {bar_terminal(s, 18)}")

    print()
    pr(C.GREEN + C.BOLD, "  เสร็จสิ้น")


# ── markdown output ────────────────────────────────────────────────────────────
def build_markdown(datasets, test_medians, final_scores, order) -> str:
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d")

    lines += [
        "# Benchmark Results",
        "",
        f"**Date:** {ts}  ",
        f"**Devices tested:** {len(datasets)}  ",
        f"**Scoring:** Normalized 0–100 (best in group = 100)",
        "",
        "---",
        "",
    ]

    # ── device summary ────────────────────────────────────────────────────────
    lines += ["## Devices", ""]
    lines += ["| Device | CPU | RAM | OS | Recorded |"]
    lines += ["|--------|-----|-----|----|----------|"]
    for d in datasets:
        p   = d["platform"]
        cpu = p.get("cpu", "?")
        ram = f"{d.get('ram_gb','?')} GB"
        os_ = f"{p.get('os','?')} {p.get('machine','')}"
        ts_ = d.get("timestamp", "?")[:10]
        lines.append(f"| {d['_label']} | {cpu} | {ram} | {os_} | {ts_} |")
    lines += ["", "---", ""]

    # ── final scores ──────────────────────────────────────────────────────────
    lines += ["## Final Scores", ""]
    lines += ["| Rank | Device | Score | Bar | Tests run |"]
    lines += ["|------|--------|------:|-----|-----------|"]
    medals = ["🥇", "🥈", "🥉"]
    for rank, i in enumerate(order, 1):
        medal = medals[rank-1] if rank <= 3 else f"#{rank}"
        fs    = final_scores[i]
        ran   = sum(1 for idx in range(len(TESTS)) if get_test_result(datasets[i], idx))
        lines.append(f"| {medal} | {datasets[i]['_label']} | **{fs}** | `{bar_md(fs, 20)}` | {ran}/{len(TESTS)} |")
    lines += ["", "---", ""]

    # ── per-test results ──────────────────────────────────────────────────────
    lines += ["## Per-Test Results", ""]
    for idx, (test_name, hib, weight) in enumerate(TESTS):
        medians, _ = test_medians[idx]
        valid = [m for m in medians if m is not None]
        direction  = "↑ higher is better" if hib else "↓ lower is better"
        lines += [f"### {test_name} — {weight*100:.0f}% weight", ""]
        lines += [f"_{direction}_", ""]

        if not valid:
            lines += ["_All skipped_", ""]
            continue

        best   = max(valid) if hib else min(valid)
        scores = [normalize(m, best, hib) if m is not None else None for m in medians]
        unit   = next((datasets[i]["results"][idx].get("unit","") for i in range(len(datasets)) if get_test_result(datasets[i], idx)), "")

        lines += ["| Device | Result | Score | Bar |"]
        lines += ["|--------|-------:|------:|-----|"]
        for i, d in enumerate(datasets):
            m = medians[i]
            s = scores[i]
            if m is None:
                lines.append(f"| {d['_label']} | — | — | — |")
            else:
                winner = " ★" if m == best else ""
                lines.append(f"| {d['_label']}{winner} | {m:.2f} {unit} | {s:.1f}/100 | `{bar_md(s, 16)}` |")
        lines.append("")

    lines += ["---", ""]

    # ── category breakdown ────────────────────────────────────────────────────
    lines += ["## Category Breakdown", ""]
    lines += ["| Category | Device | Score | Bar |"]
    lines += ["|----------|--------|------:|-----|"]
    for cat_name, test_indices in CATEGORIES.items():
        cat_scores = []
        for i, d in enumerate(datasets):
            total, tw = 0.0, 0.0
            for idx in test_indices:
                medians, hib = test_medians[idx]
                valid = [m for m in medians if m is not None]
                if not valid or medians[i] is None:
                    continue
                best = max(valid) if hib else min(valid)
                total += normalize(medians[i], best, hib) * TESTS[idx][2]
                tw    += TESTS[idx][2]
            cat_scores.append((d["_label"], round(total / tw, 1) if tw > 0 else 0.0))
        for rank, (label, s) in enumerate(sorted(cat_scores, key=lambda x: x[1], reverse=True)):
            cat_col = cat_name if rank == 0 else ""
            lines.append(f"| {cat_col} | {label} | {s}/100 | `{bar_md(s, 16)}` |")
    lines += ["", "---", ""]

    # ── scoring formula ───────────────────────────────────────────────────────
    lines += [
        "## Scoring Formula",
        "",
        "```",
        "score_i = (result_i / best) × 100   # higher-is-better",
        "score_i = (best / result_i) × 100   # lower-is-better",
        "final   = Σ(score_i × weight_i) / Σ(weight_i)",
        "```",
        "",
        "_Each device is scored relative to the best result in the group (= 100)._",
        "_Each device uses its own hardware accelerator (Metal / CoreML / DirectML)._",
        "",
    ]

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results across machines")
    parser.add_argument("files", nargs="+", help="JSON result files")
    parser.add_argument("--md", metavar="OUTPUT.md", default=None,
                        help="บันทึก Markdown report ไปที่ไฟล์นี้ด้วย")
    args = parser.parse_args()

    datasets = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            pr(C.RED, f"ไม่พบไฟล์: {f}")
            sys.exit(1)
        d = load(f)
        d["_label"] = short_name(d)
        datasets.append(d)

    test_medians, machine_scores, machine_weights, final_scores, order = collect(datasets)

    print_terminal(datasets, test_medians, final_scores, order)

    if args.md:
        md = build_markdown(datasets, test_medians, final_scores, order)
        Path(args.md).write_text(md, encoding="utf-8")
        pr(C.GREEN, f"\n  ✓ บันทึก Markdown ไปที่: {args.md}")


if __name__ == "__main__":
    main()
