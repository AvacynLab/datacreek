### Check-list « Pipeline v1.3 — Quality + Perf + Graph Boost »

*(Chaque bloc ➡️ cases à cocher → sous-étapes → maths + tableau variables → objectif & DoD)*

---

## 1 – Quality Gates Pydantic

* [ ] **Étendre schémas `schemas/*.py` avec contraintes dérivées**
  * [ ] **Images** : `min_width≥256`, `min_height≥256`, `blur_score≤0.2` (variance Laplacien).
  * [ ] **Audio** : `duration≤4 h`, `snr≥10 dB`.
  * [ ] **Texte/PDF** : `entropy≥3.5 bits/char`.
* **Formules** :

  $$
    \text{Blur} = \frac{1}{|I|}\sum (\nabla^2 I)^2,\quad
    \text{Entropy}=-\sum p_i \log_2 p_i
  $$
* **Objectif** : taux de rejet $R<3\%$.
* **DoD** : métrique `ingest_validation_fail_total/N_total` dans Prometheus.

---

## 2 – Exactly-once Kafka + Token-Bucket

* [ ] Configure **Kafka producer** avec `enable.idempotence=true`, `transaction.id`.
* [ ] **Consumer** : `transactional.id`, commit offset après `MERGE`.
* [ ] **Token-Bucket Redis** par tenant : refill $r=100$ msg/s, capacité $C=500$.

  $$
    \dot b = r - \lambda,\quad 0 \le b \le C
  $$
* **Objectif** : 0 duplicat, 0 429 sous burst 10×.
* **DoD** : test Locust, duplicat ratio = 0.

---

## 3 – Image dedup & quality scoring

* [ ] **Perceptual hash** `phash` → Bloom m = 1 G bits, k = 7 (FP ≈ 0.01 %).
* [ ] **Sharpness & Exposure** → compute `sharp`, `exposure`, feed BLIP only if $Q=\sqrt{sharp\cdot exposure}>0.4$.
* **Objectif** : ≥ 20 % appels BLIP évités.
* **DoD** : `blip_skipped_total / blip_called_total ≥ 0.2`.

---

## 4 – Smart audio chunking + LangID

* [ ] Découpe VAD (`webrtcvad.mode=3`), recolle pauses ≤ 300 ms.
* [ ] Après Whisper → `fasttext langid`; tag `lang`.
* **KPI** : WER ↓ ≥ 4 %.
* **DoD** : tests LibriSpeech, `WER_new ≤ 0.96·WER_old`.

---

## 5 – Adaptive PID (Redis TTL)

* [ ] Implémenter **gain scheduling** :
  * `Kp_day=0.3`, `Kp_night=0.5` (traffic drop).
* [ ] Calculer overshoot σ² hit-ratio ; ajuster gains via Kalman.

  $$
    K_p(t)=K_{p0}\frac{σ_e}{σ_{e0}}
  $$
* **Objectif** : oscillation hit-ratio ±1 %.
* **DoD** : `stddev(hit_ratio) ≤ 0.01`.

---

## 6 – Neo4j idempotence + timestamp

* [ ] Colonne `first_seen` (`datetime`) `ON CREATE` + `last_ingested` (`SET`).

  ```cypher
  MERGE (d:Doc {uid:$uid})
  ON CREATE SET d.first_seen=timestamp()
  SET d.last_ingested=timestamp()
  ```
* **DoD** : ré-ingestion ne crée aucune nouvelle arête ; champ mis à jour.

---

## 7 – Graph / Fractale enrichissements

### 7.1  Fractal dimension persistance

* [ ] Calculer `d_F` (box-counting) pour sous-graphe ; écrire propriété.
* [ ] Index full-text `CALL db.index.fulltext.createNodeIndex("idx_fractal",["Subgraph"],["fractal_dim"])`.
* **Objectif** : requête `WHERE fractal_dim>1.5` < 100 ms.

### 7.2  CUDA streams batching Hyper-SAGNN

* [ ] Grouper 8 sous-graphes / stream ; utiliser `cudaStream_t`.
* **Perf target** : +15 % nodes/s.
* **DoD** : benchmark `graphwave_nodes_per_s` amélioration ≥ 1.15×.

---

## 8 – Embeddings optimisation

* [ ] **FP8 compression** Poincaré : scale S, store INT8, de-quant on GPU.
* [ ] **Online PCA (sketch)** : Boutsidis 2016 ; reducer 512 → 256d.
* [ ] **Fractal encoder 32 d** : features `[d_F, spectrum_slope,…]` dense layer.
* **Objectif** : VRAM – 20 %, recall stable ±0.5 %.
* **DoD** : tests recall@10 diff ≤ 0.005.

---

## 9 – Dataset governance

* [ ] LakeFS commit ID = SHA ; Delta `OPTIMIZE ZORDER(org_id,kind)` + `VACUUM RETAIN 30 DAYS`.
* [ ] `ALTER TABLE ADD COLUMN` scripted for schema evolution.
* **KPI** : storage overhead ≤ 1.2×, query scan ms ↓ 30 %.
* **DoD** : benchmark `time_travel_query_ms` old vs new.

---

## 10 – Alerts & GDPR delete

* [ ] **Prometheus alerts** :
  * `ingest_latency_p999 > 5s for 15m` (critical)
  * Burn rate (1 h/6 h) SLO 99 %.
* [ ] **Cascade delete** : Neo4j + FAISS vector tombstone → `doc:deleted`.
* **DoD** : simulated delete request removes all traces < 1 min.

---

### KPI globaux pour v1.3-beta

| KPI                  | Cible           |
| -------------------- | --------------- |
| BLIP save            | ≥ 20 %          |
| WER gain             | ≥ 4 %           |
| Hit-ratio oscill.    | ±1 %            |
| P95 ingest (Kafka)   | ≤ 200 ms        |
| Graphwave throughput | +15 %           |
| VRAM embeddings      | –20 %           |
| p999 ingest alert    | firing on spike |

## History

- Reset checklist with 10 task groups as provided.
