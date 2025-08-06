### Check-list “v3.1 backlog → à implémenter”

*(chaque ☐ = case à cocher ; indentation = sous-étapes ; formules + tableau variables ; **Objectif** & **DoD** à la fin de chaque bloc)*

---

## 1 · Side-car pré-ingestion (safety + lang + gating) ★ haute priorité

* [x] **Repo `ingest-filter/` (Go ou Rust)**

  ☑ HTTP service (gRPC optional) devant Kafka producer.
  ☑ Endpoints `/ingest/img`, `/ingest/audio`, `/ingest/text`.
* [x] **Chaîne scoring**

  1. RegEx blacklist (O(1)).
  2. Mini-transformer `distilroberta-toxic` → $s_\text{tox}$.
  3. CLIP NSFW head → $s_\text{nsfw}$.
  4. **Décision** block si

     $$
       s=\tfrac12(s_\text{tox}+s_\text{nsfw})>0.7
     $$
* [x] **LangID** (fastText) : refuser médias où `lang∉{fr,en}`.
* [x] **Pré-gating audio/image** : calcul SNR/blur **avant** Kafka ; rejeu 4xx.
* [x] **Métriques** Prom : `filter_block_total`, `lang_skipped_total`.
* **Objectif** : −10 % volume Kafka ; faux-négatifs tox < 1 %.
* **DoD** : tests dataset tox ⇒ 99 % bloqué ; Grafana montre nouvelle métrique.

---

## 2 · Hypergraphe à hyperarêtes typées + Laplacien multiplex ★

* [x] **Migration Neo4j**
  ☑ Arêtes `:EDGE_DOC|:EDGE_USER|:EDGE_TAG`.
* [x] **Tensor Laplacien**

  $$
    \Delta^{(t)}\!=\!I\!-\!D_t^{-1/2}B_tW_tD_t^{-1}B_t^\top D_t^{-1/2},
    \qquad
    \Delta_\text{multi}=\sum_{t}\alpha_t\Delta^{(t)},~~\alpha_t\ge0,\!\sum\alpha_t=1
  $$

  | Var        | Signification    |
  | ---------- | ---------------- |
  | $B_t$      | incidence type t |
  | $\alpha_t$ | poids trainables |
* [x] **Meta-grad** : Optimiser $\alpha$ (Adam, lr 1e-2) sur Macro-F1 val.
* [x] **Stockage** : `config/hyper_weights.json`.
* **Objectif** : Macro-F1 +1 pt vs baseline.
* **DoD** : test heavy → +1 pt ; somme $\alpha$=1 ± 1e-4.

---

## 3 · Solver ILP “smart-patch” pour incohérences Sheaf ↔ Hyper ★

* [x] **`scripts/sheaf_repair.py`** (OR-Tools)

  $$
    \max_x \sum_e c_e x_e \quad\text{s.c.}\quad \sum_e w_e x_e \le B,\;x_e\in\{0,1\}
  $$

  | Var   | Description         |   |   |
  | ----- | ------------------- | - | - |
  | $c_e$ | gain Δλ             |   |   |
  | $w_e$ | risque / coût       |   |   |
  | $B$   | budget patch (=10 % | E | ) |
* [x] ☐ Génère Cypher patch set ; écrit `repair_suggestions/<id>.cypher`.
* [x] **UI** `/edge_review` : accept / reject patch set.
* **Objectif** : score S ↑ 10 % sans baisse recall.
* **DoD** : test patch cohérent ; rollback possible via ID.

---

## 4 · Flink watermark & late-event replay ★

* [x] **Update job**
  ☑ `WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofMinutes(10))`.
  ☑ Side-output `late_edges` → Kafka `late_ingest`.
* [x] ☑ Consumer merges late edges into Neo4j.
* **Objectif** : lost events < 0.1 %.
* **DoD** : integration test inject 8 min delay → edge présent.

---

## 5 · Admission controller quota GPU ★

* [x] **Service `gpu-quota-webhook/`** (Python FastAPI)
  ☑ Mutating webhook for `TrainingJob` CRD.
  ☑ Estimate $E_\text{gpu} = \text{epochs}\times\text{time\_per\_epoch}$.
  ☑ Deny if `tenant_credits - E_gpu < 0`.
* [x] **Prom metric** `gpu_minutes_total{tenant}` accumulate.
* **Objectif** : 0 dépassement budgétaire.
* **DoD** : attempt job over-budget → 403 ; credit usage visible.

---

## 6 · SmoothQuant bfloat4 + clipping ☆

* [x] **`training/quant_utils.py`**
  ☑ Group size 128 ; compute scale $s_g=\max|w|/127$.
  ☑ Clip groups > P99 to P99 value.
* [x] Export `model_bf4.gguf`.
* **Objectif** : CPU latency –5 %, PPL Δ < 2 %.
* **DoD** : benchmark script passes.

---

## 7 · Drift threshold adaptatif EWMA ☆

* [x] **`analysis/drift.py`**

  $$
    \mu_t=\lambda d_t+(1-\lambda)\mu_{t-1},~~
    \sigma_t = \sqrt{\lambda(d_t-\mu_t)^2+(1-\lambda)\sigma_{t-1}^2}
  $$

  ☑ λ = 0.1 ; alert if $d_t>\mu_t+3σ_t$.
* **DoD** : faux positifs divisé par 5 sur tenants à faible trafic.

---

## 8 · Metric `embedding_cpu_seconds_total` ☆

* [x] Wrap `vector_fp8=compute()` in `time.process_time()` ; increment Prom counter.
* [x] Histogram `embedding_cpu_seconds_per_call` to capture per-request CPU usage distribution.
* **DoD** : dashboard coût CPU + GPU complet.

---

## 9 · CronJob GC checkpoints (déjà présent) ✔️

* [x] Manifest `k8s/cron/gc.yaml` validé en cluster.

---

### KPI visés v3.1

| KPI                 | Cible   |
| ------------------- | ------- |
| Kafka volume ↓      | -10 %   |
| Tox false-negatives | < 1 %   |
| Macro-F1 commu      | +1 pt   |
| Cohérence S ↑       | +10 %   |
| Quota breach        | 0       |
| Late-event loss     | < 0.1 % |

**Une fois toutes les cases cochées, Datacreek passera en v3.1-alpha fully SaaS-hardened.**

### History
- Reset AGENTS for v3.1 backlog.
- Tracked embedding CPU usage with process_time and updated tests.
- Added default λ=0.1 EWMA drift detector with coverage tests.
- Implemented SmoothQuant bfloat4 quantization with GGUF export and tests.
- Added GPU quota webhook enforcing per-tenant credits and metrics tests.
- Added ILP smart-patch script generating Cypher patch files with tests.
- Replayed late-event edges into graph with consumer helper and tests.
- Implemented Go pre-ingestion sidecar with scoring chain, language gating, SNR/blur checks, and Prometheus metrics.
- Added typed edge support with multiplex Laplacian meta-gradient optimisation and persisted hyper weights.
- Verified sidecar blocks 99% of a toxic dataset and increments Prometheus metrics.
- Added histogram to track per-request embedding CPU seconds and extended tests.

- Verified v3.1 backlog via targeted unit and Go tests; installed networkx, ortools, and torch. Full pytest run still has 96 collection errors due to missing deps.
