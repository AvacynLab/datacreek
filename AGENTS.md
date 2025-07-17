### 📋 Checklist d’améliorations « GA » — à cocher

*(uniquement les points **non présents** dans le code actuel ; chaque tâche inclut sous‑étapes, formules/variables et objectif mesurable)*

---

## 1 | Surveillance eigsh & fallback Lanczos

### Objectif

Empêcher tout blocage > 60 s sur laplaciens géants et exposer l’incident en métrique + alerte.

### Sous‑étapes

1. **Gauge + counter**

   ```python
   eigsh_timeouts_total = Counter('eigsh_timeouts_total',
                                  'Number of eigsh timeouts')
   eigsh_last_duration  = Gauge('eigsh_last_runtime_seconds', …)
   ```
2. **Décorateur `with_timeout`**

   * Wrappe l’appel `eigsh`; start timer.
   * On timeout : incrémente `eigsh_timeouts_total`, stocke durée limite, bascule `LanczosTop5`.
3. **Alerte Prometheus**

   ```yaml
   - alert: EigshTimeoutStorm
     expr: increase(eigsh_timeouts_total[30m]) > 5
     for: 10m
   ```

---

## 2 | LMDB L2 — logs d’éviction + Soft‑Quota stable

### Sous‑étapes

1. **Eviction log**

   ```python
   logger.debug("LMDB-EVICT %s", key)
   ```
2. **Counter**
   `redis_evictions_l2_total.inc()` pour chaque suppression.
3. **Soft‑Quota**

   * `cfg.cache.l2_max_size_mb` (déjà) → Stop ajoute si size > 0.9\*quota ; log warning.

---

## 3 | fastText lang‑id pooling

### Objectif

Éliminer re‑load modèle (150 ms) sur multi‑workers.

### Sous‑étapes

1. **Singleton**

   ```python
   _ft = fasttext.load_model(PATH) if not hasattr(cache, '_ft') else cache._ft
   ```
2. **Thread‑safe queue** (size = n_cpu) pour accès concurrent.

---

## 4 | GPU ANN – IVFPQ benchmark

### Sous‑étapes

1. **Paramètre YAML**

   ```yaml
   ann:
     backend: "faiss_gpu_ivfpq"  # options: flat, hnsw, ivfpq
   ```
2. **Construction**

   ```python
   quantizer = faiss.IndexFlatIP(d)
   index = faiss.IndexIVFPQ(quantizer, d, nlist=4096, m=16, 8)
   index.train(xb); index.add(xb)
   index.nprobe = 32
   ```
3. **Prometheus gauge** `ann_backend` (label value).
4. **Benchmark script** : mesure P95 latence, recall ≥ 0.9. Met à jour README.

---

## 5 | Model‑card enrichi

### Sous‑étapes

1. **Nouvelles clés JSON**

   ```json
   {
     "bias_wasserstein": 0.08,
     "sigma_db": 1.94,
     "H_wave": 5.12,
     "prune_ratio": 0.48,
     "cca_sha": "a9c4…"
   }
   ```
2. **Génération HTML**

   * Utiliser Jinja2 template + Chart.js mini‑plots (hist demog).
3. **CI artefact** upload dans release.

---

## 6 | Cache L1 TTL — EMA smoothing

### Math

$$
\text{EMA}_t = \alpha\,r_t + (1-\alpha)\,\text{EMA}_{t-1},\;\alpha=0.3
$$

### Sous‑étapes

1. Stocker `hit_ema` global.
2. Appliquer l’EMA avant décision TTL.
3. Log TTL change event.

---

## 7 | Jitter alert tuning

### Sous‑étapes

1. Changer règle :

   ```yaml
   expr: rate(gp_jitter_restarts_total[10m]) > 0.01
   for: 10m
   ```
2. Ajouter label `severity="warning"`.

---

## 8 | Fusion multilingue – seuil probabilité

### Sous‑étapes

1. **cfg.language.min_confidence = 0.7**.
2. Dans fusion nodeSimilarity, accepter si `pred.prob >= min_confidence`.
3. Log `lang_mismatch_total` counter.

---

## 9 | Hyper‑AA index plan stable

### Sous‑étapes

1. Dans requêtes Cypher, toujours :

   ```cypher
   WITH id(a) AS id1, id(b) AS id2
   MATCH ()-[r:SUGGESTED_HYPER_AA]-()
   WHERE r.startNodeId = id1 AND r.endNodeId = id2
   ```
2. Ajouter unit‑test `PROFILE` pour vérifier `IndexSeekByRange` est utilisé.

---

## 10 | Watchdog plus large (eigsh & cache & ANN)

### Sous‑étapes

*Job cron* qui vérifie :

* `eigsh_timeouts_total` growth > 10 /h.
* `ann_latency_seconds_bucket{le="2"}` ratio < 0.95.
* Disk usage > 85 % (soft‑quota) → alerte.

---

### Variables / métriques à ajouter

| Nom                    | Type                | Commentaire                  |
| ---------------------- | ------------------- | ---------------------------- |
| `redis_hit_ratio`      | Gauge               | EMA des hits L1              |
| `eigsh_timeouts_total` | Counter             | Fallback Lanczos             |
| `ann_backend`          | Gauge(label)        | flat/hnsw/ivfpq              |
| `lang_mismatch_total`  | Counter             | Tentatives fusion cross-lang |
| `bias_wasserstein`     | Scalar (model card) | Fairness metric              |

---

#### Fin de feuille de route

## Checklist
- [x] Surveillance eigsh & fallback Lanczos
- [x] LMDB L2 — logs d'éviction + Soft-Quota stable
- [x] fastText lang-id pooling
- [x] GPU ANN – IVFPQ benchmark
- [x] Model-card enrichi
- [x] Cache L1 TTL — EMA smoothing
- [x] Jitter alert tuning
- [x] Fusion multilingue – seuil probabilité
- [x] Hyper‑AA index plan stable
- [x] Watchdog plus large (eigsh & cache & ANN)

## History
- Reset checklist and added GA roadmap.
- Implemented ANN backend gauge with IVFPQ option and TTL EMA smoothing.
- Added eigsh watchdog metrics with timeout decorator and LMDB soft quota.
- Added fastText pooling, model card HTML generation, jitter alert rule and language mismatch metric.
- Added Hyper-AA pair score lookup using indexed query and watchdog cron script.
- Installed dependencies to run tests and verified all GA tasks.
- Verified task implementation and ran tests after installing missing Python dependencies.
