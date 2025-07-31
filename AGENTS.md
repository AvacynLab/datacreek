# Development Log

## Tasks

* [x] Étendre schémas `schemas/*.py` avec contraintes dérivées
  * [x] Images : `min_width≥256`, `min_height≥256`, `blur_score≤0.2` (variance Laplacien).
  * [x] Audio : `duration≤4 h`, `snr≥10 dB`.
  * [x] Texte/PDF : `entropy≥3.5 bits/char`.
* [x] Configure Kafka producer `enable.idempotence=true`, `transaction.id`.
* [x] Consumer `transactional.id`, commit offset après `MERGE`.
* [x] Token-Bucket Redis : refill `r=100 msg/s`, capacité `C=500`.
* [x] Perceptual hash `phash` → Bloom m = 1 G bits, k = 7 (FP ≈ 0.01 %).
* [x] Sharpness & Exposure → compute `sharp`, `exposure`, skip BLIP si `Q=√(sharp·exposure)>0.4`.
* [x] Découpe VAD (`webrtcvad.mode=3`), recolle pauses ≤ 300 ms.
* [x] Après Whisper → `fasttext langid` ; tag `lang`.
* [x] Gain scheduling PID : `Kp_day=0.3`, `Kp_night=0.5` ; ajuster via Kalman.
* [x] Neo4j idempotence avec `first_seen` et `last_ingested`.
* [x] Calculer `d_F` (box-counting) pour sous-graphe ; index full-text.
* [x] CUDA streams batching Hyper-SAGNN.
* [x] FP8 compression Poincaré + Online PCA 512→256d.
* [x] Fractal encoder 32 d.
* [x] LakeFS commit ID = SHA ; Delta OPTIMIZE ZORDER + VACUUM.
* [x] Prometheus alerts `ingest_latency_p999 > 5s for 15m`.
* [x] Cascade delete Neo4j + FAISS.

### KPI globaux v1.3-beta

| KPI | Cible |
| --- | ----- |
| BLIP save | ≥ 20 % |
| WER gain | ≥ 4 % |
| Hit-ratio oscill. | ±1 % |
| P95 ingest (Kafka) | ≤ 200 ms |
| Graphwave throughput | +15 % |
| VRAM embeddings | –20 % |
| p999 ingest alert | firing on spike |

## History

- Imported v1.3 checklist and reset log.
- Added quality gates to Pydantic schemas and updated property tests.
- Enabled Kafka producer idempotence and transactional IDs.
- Raised token bucket capacity to 500.
- Neo4j Document merge now records first and last ingest timestamps.
- Added Bloom-filter deduplication with perceptual hash.
- Implemented image quality gating using sharpness/exposure metrics and updated ingestion tests.
- Added fastText language detection after audio transcription; audio nodes now store `lang` property.
- Added VAD-based audio chunking and updated whisper parser tests.
- Introduced day/night gain scheduling in Redis PID controller and tests.
- Added transactional offset handling for Kafka consumer and optional producer
  transactions.
- Stored subgraph fractal dimensions on nodes and provided Neo4j index helper.
- Added burn rate alert for ingest latency and updated Prometheus rule tests.
- Implemented cascade deletion of documents across Neo4j and FAISS with tombstones.
- Recorded LakeFS commit SHA during dataset export and ran Delta OPTIMIZE/VACUUM.
- Added CUDA-stream Hyper-SAGNN batching utility with CPU fallback and tests.
- Implemented FP8 quantization and online PCA reduction for Poincaré embeddings;
  added fractal feature encoder and unit tests.
- Verified tests with added numpy, webrtcvad, fasttext, networkx; heavy tests still failing due to config dependencies.
