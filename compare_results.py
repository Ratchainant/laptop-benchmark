"""
compare_results.py — เปรียบเทียบผลจากหลายเครื่องและคำนวณ Normalized Score

วิธีใช้:
  python compare_results.py macbook_air.json mac_mini.json dell_pro.json
"""

import json
import sys
import statistics
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
    """พยายามสร้างชื่อสั้นๆ จาก hostname หรือ CPU"""
    cpu = data["platform"].get("cpu", "")
    hostname = data["platform"].get("hostname", "")

    # Apple Silicon heuristics
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
    if higher_is_better:
        return min(val / best * 100, 100)
    else:
        return min(best / val * 100, 100)


def bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    filled = max(0, min(filled, width))
    empty  = width - filled
    color  = C.GREEN if score >= 80 else (C.YELLOW if score >= 50 else C.RED)
    return f"{color}{'█' * filled}{'░' * empty}{C.RESET}"


def main():
    if len(sys.argv) < 2:
        pr(C.YELLOW, "วิธีใช้: python compare_results.py file1.json file2.json ...")
        pr(C.YELLOW, "ถ้ามีเครื่องเดียว: python compare_results.py benchmark_result.json")
        sys.exit(1)

    files   = sys.argv[1:]
    datasets = []
    for f in files:
        p = Path(f)
        if not p.exists():
            pr(C.RED, f"ไม่พบไฟล์: {f}")
            sys.exit(1)
        d = load(f)
        d["_label"] = short_name(d)
        datasets.append(d)

    header("CROSS-MACHINE BENCHMARK COMPARISON")

    # ── machine summary ───────────────────────────────────────────────────────
    print()
    for d in datasets:
        p = d["platform"]
        label  = d["_label"]
        ram    = d.get("ram_gb", "?")
        cpu    = p.get("cpu", "?")[:55]
        ts     = d.get("timestamp", "?")[:19]
        pr(C.BOLD, f"  {label}")
        pr(C.DIM,  f"    CPU: {cpu}")
        pr(C.DIM,  f"    RAM: {ram} GB   |   Recorded: {ts}")
        print()

    # ── per-test comparison ───────────────────────────────────────────────────
    TESTS = [
        ("LLM Inference (llama.cpp)",              True,  0.30),
        ("Image Classification (ONNX ResNet-50)",  True,  0.20),
        ("Pandas groupby + join (5M rows)",         False, 0.15),
        ("DuckDB analytical query (50M rows)",      False, 0.15),
        ("Sequential I/O Read",                     True,  0.10),
        ("NumPy matrix multiply",                   True,  0.10),
    ]

    # ── collect medians ───────────────────────────────────────────────────────
    test_medians = []   # list of (list of medians per machine, higher_is_better)
    for idx, (_, hib, _) in enumerate(TESTS):
        medians = []
        for d in datasets:
            r = get_test_result(d, idx)
            medians.append(r["median"] if r else None)
        test_medians.append((medians, hib))

    # ── normalized scores ─────────────────────────────────────────────────────
    machine_scores = [0.0] * len(datasets)
    machine_weights = [0.0] * len(datasets)

    header("PER-TEST RESULTS")
    for idx, (test_name, hib, weight) in enumerate(TESTS):
        medians, _ = test_medians[idx]
        valid = [m for m in medians if m is not None]
        if not valid:
            print(f"\n  {test_name}  — all skipped\n")
            continue

        best = max(valid) if hib else min(valid)
        scores = [normalize(m, best, hib) if m is not None else None for m in medians]

        unit = datasets[0]["results"][idx].get("unit", "") if len(datasets) > 0 else ""
        direction = "↑ higher better" if hib else "↓ lower better"

        pr(C.BOLD + C.CYAN, f"\n  {test_name}   [{weight*100:.0f}%]   {direction}")
        print(f"  {'─'*66}")

        col_w = 22
        for i, d in enumerate(datasets):
            label = d["_label"]
            m = medians[i]
            s = scores[i]
            if m is None:
                print(f"  {label:<{col_w}}  {'skipped':<12}  {'':22}")
                continue

            # mark winner
            is_best = (m == best)
            medal   = f"{C.GREEN}★{C.RESET}" if is_best else " "

            val_str  = f"{m:.2f} {unit}"
            score_str = f"{s:.1f}/100" if s is not None else "—"

            print(f"  {medal} {label:<{col_w}}  {val_str:>14}   {bar(s)}  {score_str}")

            # accumulate weighted score
            if s is not None:
                machine_scores[i]  += s * weight
                machine_weights[i] += weight

    # ── final scores ──────────────────────────────────────────────────────────
    header("FINAL NORMALIZED SCORES (Weighted)")
    print()

    final_scores = []
    for i, d in enumerate(datasets):
        w = machine_weights[i]
        fs = round(machine_scores[i] / w, 1) if w > 0 else 0.0
        final_scores.append(fs)

    # sort by score descending
    order = sorted(range(len(datasets)), key=lambda i: final_scores[i], reverse=True)

    print(f"  {'Rank':<5}  {'Machine':<25}  {'Score':>7}   {'':20}   {'Tests run'}")
    print(f"  {'─'*5}  {'─'*25}  {'─'*7}   {'─'*20}   {'─'*9}")

    for rank, i in enumerate(order, 1):
        label = datasets[i]["_label"]
        fs    = final_scores[i]
        w     = machine_weights[i]
        ran   = round(w / 1.0 * 6)  # approx number of tests that ran
        medal = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"
        print(f"  {medal:<5}  {label:<25}  {fs:>6.1f}   {bar(fs, 20)}   {ran}/6")

    # ── scoring formula reminder ──────────────────────────────────────────────
    print()
    pr(C.DIM, "  Formula: score_i = (result_i / best) × 100  (higher-is-better)")
    pr(C.DIM, "           score_i = (best / result_i) × 100  (lower-is-better)")
    pr(C.DIM, "           final   = Σ(score_i × weight_i) / Σ(weight_i)")
    print()

    # ── per-category breakdown ────────────────────────────────────────────────
    header("CATEGORY BREAKDOWN")
    CATEGORIES = {
        "AI Inference":    [0, 1],
        "Data / Analytics": [2, 3],
        "I/O + Memory":    [4, 5],
    }

    for cat_name, test_indices in CATEGORIES.items():
        pr(C.BOLD, f"\n  {cat_name}")
        print(f"  {'─'*50}")
        cat_scores = []
        for i, d in enumerate(datasets):
            total, tw = 0.0, 0.0
            for idx in test_indices:
                _, hib     = test_medians[idx]
                medians, _ = test_medians[idx]
                valid = [m for m in medians if m is not None]
                if not valid:
                    continue
                best = max(valid) if hib else min(valid)
                m = medians[i]
                if m is None:
                    continue
                s = normalize(m, best, hib)
                w = TESTS[idx][2]
                total += s * w
                tw    += w
            fs = round(total / tw, 1) if tw > 0 else 0.0
            cat_scores.append((d["_label"], fs))

        cat_scores.sort(key=lambda x: x[1], reverse=True)
        for label, s in cat_scores:
            print(f"  {label:<25}  {s:>6.1f}/100  {bar(s, 18)}")

    print()
    pr(C.GREEN + C.BOLD, "  เสร็จสิ้น")


if __name__ == "__main__":
    main()
