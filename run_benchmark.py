"""
AI & Data Analytics Benchmark Suite
ทดสอบ: MacBook Air M3 / Mac Mini M4 / Dell Pro 16 (Ryzen AI 5 Pro 340)
รัน: python run_benchmark.py [--runs 5] [--output results.json]
"""

import platform
import sys
import time
import json
import argparse
import statistics
import subprocess
from pathlib import Path
from datetime import datetime

# ── color output ──────────────────────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def pr(color, msg): print(f"{color}{msg}{C.RESET}")
def header(msg):    pr(C.BOLD + C.HEADER, f"\n{'═'*60}\n  {msg}\n{'═'*60}")
def step(msg):      pr(C.CYAN, f"  ▸ {msg}")
def ok(msg):        pr(C.GREEN, f"  ✓ {msg}")
def warn(msg):      pr(C.YELLOW, f"  ⚠ {msg}")
def err(msg):       pr(C.RED, f"  ✗ {msg}")

# ── dependency check ──────────────────────────────────────────────────────────
REQUIRED = {
    "numpy":   "numpy",
    "pandas":  "pandas",
    "pyarrow": "pyarrow",
    "duckdb":  "duckdb",
    "psutil":  "psutil",
}

def check_deps():
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        err(f"Missing packages: {', '.join(missing)}")
        pr(C.YELLOW, f"  Install: pip install {' '.join(missing)}")
        sys.exit(1)
    ok("All required packages found")

# ── platform detection ────────────────────────────────────────────────────────
def detect_platform():
    info = {
        "os":       platform.system(),
        "machine":  platform.machine(),
        "cpu":      platform.processor() or platform.machine(),
        "python":   platform.python_version(),
        "hostname": platform.node(),
        "is_apple_silicon": False,
        "onnx_available": False,
        "onnx_backend": None,
    }

    # Apple Silicon check
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["is_apple_silicon"] = True
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            info["cpu"] = result.stdout.strip() or info["cpu"]
        except Exception:
            pass

    # ONNX Runtime
    try:
        import onnxruntime as ort
        info["onnx_available"] = True
        providers = ort.get_available_providers()
        if "CoreMLExecutionProvider" in providers:
            info["onnx_backend"] = "CoreML"
        elif "DmlExecutionProvider" in providers:
            info["onnx_backend"] = "DirectML (NPU/GPU)"
        elif "CUDAExecutionProvider" in providers:
            info["onnx_backend"] = "CUDA"
        else:
            info["onnx_backend"] = "CPU"
    except ImportError:
        pass

    return info


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── T1: LLM inference (Ollama) ───────────────────────────────────────────────
def bench_llm(runs: int, plat: dict, ollama_model: str = "llama3.2:latest") -> dict:
    """
    วัด token generation speed (tokens/sec) ผ่าน Ollama REST API
    ไม่ต้องติดตั้ง llama-cpp-python — ใช้โมเดลที่ pull ไว้ใน Ollama ได้เลย
    """
    import urllib.request, urllib.error

    NAME = f"LLM Inference (Ollama · {ollama_model})"
    import os
    API  = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    PROMPT   = "Explain the concept of machine learning in simple terms."
    N_TOKENS = 128

    # ── ตรวจว่า Ollama daemon ทำงานอยู่ ──────────────────────────────────────
    try:
        urllib.request.urlopen(f"{API}/api/tags", timeout=3)
    except Exception:
        return {
            "name": NAME, "status": "skip",
            "reason": "Ollama daemon ไม่ตอบสนอง",
            "fix": "รัน `ollama serve` ก่อน หรือเปิดแอป Ollama",
            "weight": 0.30,
        }

    # ── ตรวจว่ามีโมเดลนี้ pull ไว้แล้ว ──────────────────────────────────────
    try:
        with urllib.request.urlopen(f"{API}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read())
        model_names = [m["name"] for m in tags.get("models", [])]
        matched = next(
            (n for n in model_names if n.startswith(ollama_model.split(":")[0])),
            None
        )
        if not matched:
            return {
                "name": NAME, "status": "skip",
                "reason": f"ไม่พบโมเดล '{ollama_model}' ใน Ollama",
                "fix": f"รัน: ollama pull {ollama_model}",
                "available": model_names,
                "weight": 0.30,
            }
        ollama_model = matched
        ok(f"พบโมเดล: {ollama_model}")
    except Exception as e:
        return {"name": NAME, "status": "skip", "reason": f"ตรวจโมเดลไม่ได้: {e}", "weight": 0.30}

    # ── helper: เรียก Ollama generate (non-streaming) ─────────────────────────
    def call_ollama(prompt: str, max_tokens: int) -> dict:
        payload = json.dumps({
            "model":  ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{API}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    # ── warm-up ───────────────────────────────────────────────────────────────
    step("Warm-up run (โหลดโมเดลเข้า Metal GPU)...")
    try:
        call_ollama(PROMPT, 32)
    except Exception as e:
        return {"name": NAME, "status": "skip", "reason": f"warm-up ล้มเหลว: {e}", "weight": 0.30}

    # ── benchmark runs ────────────────────────────────────────────────────────
    samples = []
    for i in range(runs):
        t0  = time.perf_counter()
        out = call_ollama(PROMPT, N_TOKENS)
        t1  = time.perf_counter()
        generated = out.get("eval_count", N_TOKENS)
        elapsed   = t1 - t0
        tps       = generated / elapsed
        samples.append(tps)
        step(f"  run {i+1}/{runs}: {tps:.1f} tokens/sec  ({generated} tokens, {elapsed:.2f}s)")

    backend = "Metal (Apple Silicon)" if plat["is_apple_silicon"] else "CPU"
    return {
        "name": NAME, "status": "ok",
        "unit": "tokens/sec",
        "higher_is_better": True,
        "samples": [round(s, 2) for s in samples],
        "median": round(statistics.median(samples), 2),
        "mean":   round(statistics.mean(samples), 2),
        "stdev":  round(statistics.stdev(samples) if len(samples) > 1 else 0, 2),
        "backend": backend,
        "model": ollama_model,
        "weight": 0.30,
    }


# ── T2: ONNX image classification ────────────────────────────────────────────
def bench_onnx_image(runs: int, plat: dict) -> dict:
    """
    วัด ResNet-50 throughput (images/sec) ด้วย ONNX Runtime
    ใช้ random tensor แทนรูปจริง — วัด inference speed ล้วนๆ
    """
    NAME = "Image Classification (ResNet-50, ONNX)"
    MODEL_PATH = Path.home() / ".cache" / "benchmark_models" / "resnet50.onnx"
    MODEL_URL  = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"

    if not plat["onnx_available"]:
        return {
            "name": NAME, "status": "skip",
            "reason": "onnxruntime ไม่ได้ติดตั้ง",
            "install_mac": "pip install onnxruntime",
            "install_amd": "pip install onnxruntime-directml  (Windows NPU/GPU)",
        }

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        step("กำลังดาวน์โหลด ResNet-50 ONNX (~100MB)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            ok("ดาวน์โหลดโมเดลสำเร็จ")
        except Exception as e:
            return {"name": NAME, "status": "skip", "reason": f"ดาวน์โหลดไม่สำเร็จ: {e}"}

    import onnxruntime as ort
    import numpy as np

    # เลือก provider ตาม platform
    providers = ort.get_available_providers()
    if "CoreMLExecutionProvider" in providers:
        chosen = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif "DmlExecutionProvider" in providers:
        chosen = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        chosen = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(MODEL_PATH), providers=chosen)
    input_name = sess.get_inputs()[0].name
    BATCH = 16
    dummy = np.random.randn(BATCH, 3, 224, 224).astype(np.float32)

    samples = []
    step("Warm-up run...")
    sess.run(None, {input_name: dummy})

    for i in range(runs):
        t0 = time.perf_counter()
        for _ in range(4):  # 4 × batch16 = 64 images per run
            sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        ips = (BATCH * 4) / (t1 - t0)
        samples.append(ips)
        step(f"  run {i+1}/{runs}: {ips:.1f} images/sec")

    return {
        "name": NAME, "status": "ok",
        "unit": "images/sec",
        "higher_is_better": True,
        "samples": [round(s, 2) for s in samples],
        "median": round(statistics.median(samples), 2),
        "mean":   round(statistics.mean(samples), 2),
        "stdev":  round(statistics.stdev(samples) if len(samples) > 1 else 0, 2),
        "backend": plat["onnx_backend"] or "CPU",
        "weight": 0.20,
    }


# ── T3: Pandas groupby + join (5M rows) ──────────────────────────────────────
def bench_pandas(runs: int) -> dict:
    NAME = "Pandas groupby + join (5M rows)"
    import pandas as pd
    import numpy as np

    N = 5_000_000

    def make_data():
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "product_id":  rng.integers(0, 1000, N),
            "region":      rng.choice(["north","south","east","west"], N),
            "amount":      rng.uniform(10, 10000, N).round(2),
            "quantity":    rng.integers(1, 100, N),
            "date":        pd.date_range("2020-01-01", periods=N, freq="1s"),
        })
        lookup = pd.DataFrame({
            "product_id": np.arange(1000),
            "category":   rng.choice(["A","B","C","D"], 1000),
            "cost":       rng.uniform(1, 500, 1000).round(2),
        })
        return df, lookup

    step("กำลังสร้าง dataset (5M rows)...")
    df, lookup = make_data()

    samples = []
    step("Warm-up run...")
    _ = df.merge(lookup, on="product_id").groupby(["region","category"])["amount"].agg(["sum","mean","count"])

    for i in range(runs):
        t0 = time.perf_counter()
        merged  = df.merge(lookup, on="product_id")
        result  = merged.groupby(["region", "category"])["amount"].agg(["sum","mean","count"])
        _       = result.sort_values("sum", ascending=False)
        t1 = time.perf_counter()
        elapsed = round(t1 - t0, 3)
        samples.append(elapsed)
        step(f"  run {i+1}/{runs}: {elapsed:.3f}s")

    return {
        "name": NAME, "status": "ok",
        "unit": "seconds",
        "higher_is_better": False,
        "samples": samples,
        "median": round(statistics.median(samples), 3),
        "mean":   round(statistics.mean(samples), 3),
        "stdev":  round(statistics.stdev(samples) if len(samples) > 1 else 0, 3),
        "weight": 0.15,
    }


# ── T4: DuckDB analytical query (50M rows) ───────────────────────────────────
def bench_duckdb(runs: int) -> dict:
    NAME = "DuckDB analytical query (50M rows)"
    import duckdb
    import numpy as np
    import tempfile, os

    N = 50_000_000
    step("กำลังสร้าง Parquet dataset (50M rows)...")

    rng   = np.random.default_rng(99)
    chunk = 5_000_000
    tmpdir = Path(tempfile.mkdtemp())
    parquet_path = tmpdir / "sales.parquet"

    # เขียนเป็น chunks เพื่อประหยัด RAM
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    for c in range(N // chunk):
        tbl = pa.table({
            "order_id":   pa.array(rng.integers(1, 10**9, chunk), type=pa.int64()),
            "product_id": pa.array(rng.integers(1, 5000, chunk),  type=pa.int32()),
            "region":     pa.array(rng.choice(["N","S","E","W","C"], chunk)),
            "revenue":    pa.array(rng.uniform(10, 50000, chunk).round(2), type=pa.float64()),
            "quantity":   pa.array(rng.integers(1, 500, chunk),   type=pa.int32()),
            "year":       pa.array(rng.integers(2018, 2025, chunk),type=pa.int16()),
        })
        if writer is None:
            writer = pq.ParquetWriter(str(parquet_path), tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()

    QUERY = f"""
        SELECT
            region,
            year,
            SUM(revenue)                                      AS total_revenue,
            AVG(revenue)                                      AS avg_revenue,
            COUNT(*)                                          AS order_count,
            RANK() OVER (PARTITION BY year ORDER BY SUM(revenue) DESC) AS region_rank
        FROM read_parquet('{parquet_path}')
        GROUP BY region, year
        ORDER BY year, region_rank
    """

    con = duckdb.connect()
    samples = []
    step("Warm-up run...")
    con.execute(QUERY).fetchdf()

    for i in range(runs):
        t0 = time.perf_counter()
        con.execute(QUERY).fetchdf()
        t1 = time.perf_counter()
        elapsed = round(t1 - t0, 3)
        samples.append(elapsed)
        step(f"  run {i+1}/{runs}: {elapsed:.3f}s")

    # cleanup
    try:
        parquet_path.unlink()
        tmpdir.rmdir()
    except Exception:
        pass

    return {
        "name": NAME, "status": "ok",
        "unit": "seconds",
        "higher_is_better": False,
        "samples": samples,
        "median": round(statistics.median(samples), 3),
        "mean":   round(statistics.mean(samples), 3),
        "stdev":  round(statistics.stdev(samples) if len(samples) > 1 else 0, 3),
        "weight": 0.15,
    }


# ── T5: I/O throughput (10 GB sequential read) ───────────────────────────────
def bench_io(runs: int) -> dict:
    NAME = "Sequential I/O Read (Parquet, ~1GB)"
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    import tempfile

    N = 8_000_000   # ~1GB — ลดจาก 10GB เพื่อให้รันได้บน RAM ปกติ
    tmpdir  = Path(tempfile.mkdtemp())
    fpath   = tmpdir / "io_test.parquet"

    step("กำลังสร้างไฟล์ทดสอบ (~1GB)...")
    rng = np.random.default_rng(7)
    tbl = pa.table({
        "id":    pa.array(rng.integers(0, 10**9, N), type=pa.int64()),
        "val1":  pa.array(rng.random(N),  type=pa.float64()),
        "val2":  pa.array(rng.random(N),  type=pa.float64()),
        "val3":  pa.array(rng.random(N),  type=pa.float64()),
        "label": pa.array(rng.choice(["X","Y","Z","W"], N)),
    })
    pq.write_table(tbl, str(fpath))
    file_size_gb = fpath.stat().st_size / 1e9

    samples = []
    step("Warm-up run...")
    pq.read_table(str(fpath))

    for i in range(runs):
        t0 = time.perf_counter()
        _ = pq.read_table(str(fpath))
        t1 = time.perf_counter()
        gbps = file_size_gb / (t1 - t0)
        samples.append(gbps)
        step(f"  run {i+1}/{runs}: {gbps:.2f} GB/s  ({file_size_gb:.2f} GB)")

    try:
        fpath.unlink()
        tmpdir.rmdir()
    except Exception:
        pass

    return {
        "name": NAME, "status": "ok",
        "unit": "GB/s",
        "higher_is_better": True,
        "samples": [round(s, 3) for s in samples],
        "median": round(statistics.median(samples), 3),
        "mean":   round(statistics.mean(samples), 3),
        "stdev":  round(statistics.stdev(samples) if len(samples) > 1 else 0, 3),
        "file_size_gb": round(file_size_gb, 2),
        "weight": 0.10,
    }


# ── T6: NumPy matrix multiply (memory bandwidth) ────────────────────────────
def bench_numpy(runs: int) -> dict:
    NAME = "NumPy matrix multiply (GFLOPS)"
    import numpy as np

    # 4K×4K เพื่อให้รันบน RAM 8GB ได้ (เครื่อง Air M3 base มี 8GB)
    N = 4096

    step(f"Matrix size: {N}×{N} float32")
    rng = np.random.default_rng(0)
    A = rng.random((N, N), dtype=np.float32)
    B = rng.random((N, N), dtype=np.float32)

    FLOPS = 2 * N**3   # matmul FLOPs

    samples = []
    step("Warm-up run...")
    np.dot(A, B)

    for i in range(runs):
        t0 = time.perf_counter()
        np.dot(A, B)
        t1 = time.perf_counter()
        gflops = FLOPS / (t1 - t0) / 1e9
        samples.append(gflops)
        step(f"  run {i+1}/{runs}: {gflops:.1f} GFLOPS")

    return {
        "name": NAME, "status": "ok",
        "unit": "GFLOPS",
        "higher_is_better": True,
        "samples": [round(s, 2) for s in samples],
        "median": round(statistics.median(samples), 2),
        "mean":   round(statistics.mean(samples), 2),
        "stdev":  round(statistics.stdev(samples) if len(samples) > 1 else 0, 2),
        "matrix_size": f"{N}×{N}",
        "weight": 0.10,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_score(results: list[dict], all_results_across_machines: list | None = None) -> float:
    """
    คำนวณ normalized weighted score 0–100
    baseline = ค่าของเครื่องนี้เอง (single-machine mode)
    หรือ best across all machines (multi-machine mode)
    """
    total, total_weight = 0.0, 0.0
    for r in results:
        if r.get("status") != "ok":
            continue
        w = r.get("weight", 0)
        val = r.get("median", 0)
        # single machine → score 100 on each (useful for relative comparison later)
        total += 100 * w
        total_weight += w
    return round((total / total_weight) if total_weight > 0 else 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(plat: dict, results: list[dict]):
    header("BENCHMARK RESULTS")
    pr(C.BOLD, f"  Device  : {plat['hostname']}")
    pr(C.BOLD, f"  CPU     : {plat['cpu']}")
    pr(C.BOLD, f"  OS      : {plat['os']} {platform.release()}")
    pr(C.BOLD, f"  Python  : {plat['python']}")

    # ── per-test raw scores ───────────────────────────────────────────────────
    for r in results:
        name   = r["name"]
        status = r.get("status", "?")
        weight = r.get("weight", 0)
        print()
        pr(C.BOLD + C.CYAN, f"  ┌─ {name}  [{weight*100:.0f}%]")

        if status != "ok":
            reason = r.get("reason", "")
            pr(C.YELLOW, f"  │  skipped — {reason}")
            if r.get("install"):
                pr(C.DIM, f"  │  install : {r['install']}")
            pr(C.DIM, "  └─")
            continue

        unit    = r.get("unit", "")
        hib     = r.get("higher_is_better", True)
        median  = r.get("median", 0)
        mean    = r.get("mean", 0)
        stdev   = r.get("stdev", 0)
        samples = r.get("samples", [])
        backend = r.get("backend", "")
        direction = "↑ higher better" if hib else "↓ lower better"

        pr(C.BOLD,  f"  │  median : {median:.3g} {unit}   ({direction})")
        pr(C.DIM,   f"  │  mean   : {mean:.3g} {unit}")
        pr(C.DIM,   f"  │  stdev  : {stdev:.3g} {unit}"
                    + (f"   ({stdev/mean*100:.1f}% CV)" if mean else ""))

        # raw samples on one line
        samples_str = "  ".join(f"{s:.3g}" for s in samples)
        pr(C.DIM,   f"  │  runs   : {samples_str}  [{unit}]")

        if backend:
            pr(C.DIM, f"  │  backend: {backend}")
        pr(C.DIM, "  └─")

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    pr(C.BOLD, f"  {'SUMMARY':─<58}")
    print(f"  {'Test':<40}  {'Median':>10}  {'±stdev':>8}  {'Unit'}")
    print(f"  {'─'*40}  {'─'*10}  {'─'*8}  {'─'*12}")
    for r in results:
        name = r["name"][:40]
        if r.get("status") != "ok":
            print(f"  {name:<40}  {'—':>10}  {'—':>8}  skipped")
        else:
            print(f"  {name:<40}  {r['median']:>10.3g}  {r.get('stdev',0):>8.3g}  {r.get('unit','')}")

    print()
    pr(C.BOLD + C.BLUE, "  ℹ  เพื่อเปรียบเทียบระหว่างเครื่อง รัน script นี้บนทุกเครื่องแล้ว")
    pr(C.BLUE,          "     รวม JSON output ไว้ใน list แล้วใช้ compare_results.py")


def save_json(plat: dict, results: list[dict], path: str):
    import psutil
    out = {
        "timestamp": datetime.now().isoformat(),
        "platform":  plat,
        "ram_gb":    round(psutil.virtual_memory().total / 1e9, 1),
        "results":   results,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    ok(f"บันทึกผลลัพธ์ไปที่: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AI & Data Analytics Benchmark")
    parser.add_argument("--runs",   type=int, default=5, help="จำนวนรอบต่อ test (default 5)")
    parser.add_argument("--output", type=str, default="benchmark_result.json", help="ไฟล์ JSON output")
    parser.add_argument("--model",  type=str, default="llama3.2:latest", help="Ollama model ที่จะใช้ (default: llama3.2:latest)")
    parser.add_argument("--ollama-host", type=str, default=None, help="Ollama API URL — ถ้าไม่ระบุจะใช้ OLLAMA_HOST env var หรือ http://localhost:11434")
    parser.add_argument("--skip-llm",   action="store_true", help="ข้าม LLM test")
    parser.add_argument("--skip-onnx",  action="store_true", help="ข้าม ONNX test")
    parser.add_argument("--quick",      action="store_true", help="รัน 2 รอบ + ข้าม LLM (สำหรับทดสอบ)")
    args = parser.parse_args()

    if args.quick:
        args.runs = 2
        args.skip_llm = True

    # ถ้าส่ง --ollama-host มา ให้ set env var ก่อน bench_llm จะอ่านเอง
    import os
    if args.ollama_host:
        os.environ["OLLAMA_HOST"] = args.ollama_host

    header("AI & DATA ANALYTICS BENCHMARK SUITE")
    pr(C.CYAN, f"  Runs per test : {args.runs}")
    pr(C.CYAN, f"  Ollama model  : {args.model}")
    pr(C.CYAN, f"  Output        : {args.output}")
    print()

    step("ตรวจสอบ dependencies...")
    check_deps()
    plat = detect_platform()
    ok(f"Platform: {'Apple Silicon' if plat['is_apple_silicon'] else plat['cpu']}")

    results = []

    # T1 — LLM
    if not args.skip_llm:
        header(f"T1: LLM Inference (Ollama · {args.model})")
        results.append(bench_llm(args.runs, plat, ollama_model=args.model))
    else:
        warn("ข้าม T1 (LLM) ตามที่ระบุ")
        results.append({"name": f"LLM Inference (Ollama · {args.model})", "status": "skip", "reason": "skipped by user", "weight": 0.30})

    # T2 — ONNX
    if not args.skip_onnx:
        header("T2: Image Classification (ONNX ResNet-50)")
        results.append(bench_onnx_image(args.runs, plat))
    else:
        warn("ข้าม T2 (ONNX) ตามที่ระบุ")
        results.append({"name": "Image Classification (ONNX)", "status": "skip", "reason": "skipped by user", "weight": 0.20})

    # T3 — Pandas
    header("T3: Pandas groupby + join (5M rows)")
    results.append(bench_pandas(args.runs))

    # T4 — DuckDB
    header("T4: DuckDB analytical query (50M rows)")
    results.append(bench_duckdb(args.runs))

    # T5 — I/O
    header("T5: Sequential I/O Read")
    results.append(bench_io(args.runs))

    # T6 — NumPy
    header("T6: NumPy Matrix Multiply")
    results.append(bench_numpy(args.runs))

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(plat, results)
    save_json(plat, results, args.output)
    header("เสร็จสิ้น")
    pr(C.GREEN, f"  รันต่อไปด้วย: python compare_results.py <file1.json> <file2.json> ...\n")


if __name__ == "__main__":
    main()
