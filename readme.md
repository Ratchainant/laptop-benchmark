# AI & Data Analytics Benchmark Suite

ทดสอบและเปรียบเทียบ: MacBook Air M3 · Mac Mini M4 · Dell Pro 16 (Ryzen AI 5 Pro 340)

---

## การติดตั้ง

```bash
pip install numpy pandas pyarrow duckdb psutil
```

### Mac (M3 / M4) — ONNX CoreML backend
```bash
pip install onnxruntime
```

### Windows (Ryzen AI 5 Pro 340) — ONNX DirectML backend
```bash
pip install onnxruntime-directml
```

### T1 LLM test — ต้องติดตั้ง Ollama
ดาวน์โหลด Ollama ได้ที่ https://ollama.com แล้ว pull โมเดลที่ต้องการ

```bash
ollama pull llama3.2:latest
```

ไม่ต้องติดตั้ง `llama-cpp-python` อีกต่อไป — script เรียกผ่าน Ollama API แทน

---

## วิธีรัน

### รันบนแต่ละเครื่อง

```bash
# รัน full benchmark (ทุก test, 5 รอบ)
python run_benchmark.py --output mac_mini_m4.json

# กำหนดจำนวนรอบ
python run_benchmark.py --runs 3 --output mac_mini_m4.json

# ใช้โมเดลอื่นที่ pull ไว้ใน Ollama
python run_benchmark.py --model llama3.1:8b --output mac_mini_m4.json

# ข้าม LLM test
python run_benchmark.py --skip-llm --output mac_mini_m4.json

# ทดสอบด่วน (2 รอบ, ข้าม LLM)
python run_benchmark.py --quick --output mac_mini_m4.json
```

### เปรียบเทียบระหว่างเครื่อง

```bash
python compare_results.py macbook_air_m3.json mac_mini_m4.json dell_pro.json
```

---

## การตั้งค่า Ollama Host

script ไม่มี URL hardcode — อ่าน host จาก 3 แหล่งตามลำดับความสำคัญ

| ลำดับ | วิธี | ตัวอย่าง |
|-------|------|---------|
| 1 | `--ollama-host` argument | `python run_benchmark.py --ollama-host http://192.168.1.10:11434` |
| 2 | `OLLAMA_HOST` env var | `export OLLAMA_HOST=http://192.168.1.10:11434` |
| 3 | fallback (Ollama default) | `http://localhost:11434` |

กรณีปกติที่ Ollama รันบนเครื่องเดียวกัน ไม่ต้องตั้งค่าอะไรเพิ่มเติมครับ

---

## การทดสอบทั้งหมด (น้ำหนักรวม 100%)

| # | การทดสอบ | เครื่องมือ | metric | น้ำหนัก |
|---|----------|-----------|--------|---------|
| T1 | LLM Inference | Ollama API | tokens/sec | 30% |
| T2 | Image Classification (ResNet-50) | ONNX Runtime | images/sec | 20% |
| T3 | Pandas groupby + join (5M rows) | pandas 2.x | seconds | 15% |
| T4 | DuckDB analytical query (50M rows) | DuckDB 1.x | seconds | 15% |
| T5 | Sequential Parquet read (~1GB) | pyarrow | GB/s | 10% |
| T6 | Matrix multiply 4K×4K float32 | NumPy | GFLOPS | 10% |

T1 ใช้โมเดลที่มีอยู่แล้วใน Ollama — ไม่มีการดาวน์โหลดอัตโนมัติ

---

## สูตรคะแนน (Normalized Score)

```
score_i = (result_i / best)  × 100   # higher-is-better (tokens/sec, GFLOPS, GB/s)
score_i = (best / result_i)  × 100   # lower-is-better  (seconds)

final   = Σ(score_i × weight_i) / Σ(weight_i)
```

แต่ละเครื่องได้คะแนนเทียบกับ **ผลดีที่สุดในกลุ่ม = 100 คะแนน**
ทำให้เปรียบได้ข้าม platform โดยไม่ต้องสนใจ raw hardware spec

---

## ไฟล์

```
benchmark/
├── run_benchmark.py      ← รัน benchmark บนเครื่องนี้
├── compare_results.py    ← เปรียบเทียบ JSON ข้ามเครื่อง
├── requirements.txt      ← dependencies
└── README.md             ← ไฟล์นี้
```
