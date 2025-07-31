### Pipeline v1.3 — Quality + Perf + Graph Boost

#### Tasks Checklist

- [x] **1 – Quality Gates Pydantic**
  - [x] Images: min_width>=256, min_height>=256, blur_score<=0.2
  - [x] Audio: duration<=4 h, snr>=10 dB
  - [x] Texte/PDF: entropy>=3.5 bits/char
- [x] **2 – Exactly-once Kafka + Token-Bucket**
- [x] **3 – Image dedup & quality scoring**
- [x] **4 – Smart audio chunking + LangID**
- [x] **5 – Adaptive PID (Redis TTL)**
- [x] **6 – Neo4j idempotence + timestamp**
- [x] **7 – Graph / Fractale enrichissements**
  - [x] Fractal dimension persistance
  - [x] CUDA streams batching Hyper-SAGNN
- [x] **8 – Embeddings optimisation**
- [x] **9 – Dataset governance**
- [x] **10 – Alerts & GDPR delete**

#### Info
KPIs: BLIP save ≥20%, WER gain ≥4%, Hit-ratio oscillation ±1%, P95 ingest ≤200 ms, Graphwave throughput +15%, VRAM –20%, p999 ingest alert active.

#### History
- Implemented quality gates, exactly-once Kafka, image dedup, VAD chunking with langid, PID gain schedule, Neo4j timestamps.
- Added fractal dimension persistence, CUDA stream embeddings, FP8 compression, LakeFS governance, Prometheus alerts, cascade delete.
