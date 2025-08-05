### Check-list « v3.0 – SaaS Readiness & Scale-Out »

*(chaque ☐ = case à cocher ; indentation → sous-étapes ; formules & tableaux variables inclus ; **Objectif** + **DoD** à la fin de chaque bloc)*

---

## 0 – Gouvernance & architecture multi-tenant (P0)

### 0.1  Namespaces K8s + quotas

* [x] **Créer script Helm `charts/tenant-ns.tpl`**
  [x] Namespace `tenant-{{id}}`
  [x] `ResourceQuota` CPU, GPU (`nvidia.com/gpu`)
  [x] `LimitRange` mem/CPU par pod
* [x] **NetworkPolicy** isole egress inter-tenant
* **Objectif** : aucune pod cross-namespace reachable
* **DoD** : test `kubectl exec ... curl` fail inter-tenant

### 0.2  Neo4j Fabric multi-db

* [x] Instance **Fabric** + DB par tenant
  [x] Route Cypher : `USING DATABASE $tenant`
  [x] Flyway migration path `db/migration/<tenant>/`
* **DoD** : user A ne voit pas nœuds user B (Cypher test)

### 0.3  Crédit-GPU & billing

* [x] **Prometheus exporter** `gpu_minutes_total{tenant}`

  $$
    C = t_\text{GPU}(min)\times P_\text{unit}
  $$

  | Var             | Sig.          |
  | --------------- | ------------- |
  | $t_\text{GPU}$  | durée minutes |
  | $P_\text{unit}$ | prix/min      |
* [x] **Quota controller** : stop job si crédits < 0
* [x] **Coût total** `gpu_cost_total{tenant}` via $C=t_\text{GPU}\times P_\text{unit}$
* [x] **Admin** : top-up & mise à jour prix par tenant
* **DoD** : job dépasse quota ⇒ état `failed_quota`

---

## 1 – Scalabilité GPU & optimisation coût (P0)

### 1.1  Spot / on-demand autoscale

* [x] **Cluster-Autoscaler** + GPU-Operator
  [x] NodeGroup “spot-gpu”, “on-demand-gpu”
  [x] Taint spot `preemptible=true:NoSchedule`
* [x] Job spec : `tolerations` + restartPolicy `OnFailure`
* **DoD** : spot évicté → fallback CPU LoRA 8-bit ; metric `reschedule_latency` < 3 min

### 1.2  ZeRO-3 + gradient accumulation

* [x] `accelerate_config.yaml` enable ZeRO-3
  [x] `gradient_accumulation_steps = ceil(batch/avail_mem)`
* **DoD** : batch 4×, VRAM Δ 0

---

## 2 – Orchestration MLOps (P1)

### 2.1  **Airflow DAG** `dags/datacreek_finetune.py`

```
ingest  >>   build_dataset   >>  fine_tune_SFT
                                 >>  eval_QA
                                 >>  deploy_canary
```

* [x] Tasks param `tenant`, push to XCom
* **DoD** : DAG success end-to-end on dev cluster

### 2.2  MLflow tracking

* [x] **mlflow.start_run** in trainer callback
  [x] log params (lr, epochs, template_SHA)
  [x] log metric `val_loss`, `reward_avg`
  [x] artifact `model.gguf`
* **DoD** : run visible in MLflow UI ; compare diff runs

### 2.3  Feature Store (Feast)

* [x] **feast/feature_repo.py**
  [x] Entity `embedding_hash`
  [x] Feature `vector_fp8`
* **DoD** : lookup same hash returns cached vector (cache hit rate metric)

---

## 3 – Serving & A/B / canary (P1)

* [x] **Ray Serve** deployment
  [x] `deployment_name = f"{tenant}-{modelVer}"`
  [x] Router header `X-Tenant`, `X-Model-Version`
* [x] Canary 5 % traffic, metric `p99_latency`

  $$
    \text{Rollback if } p99_\text{canary} > 2\times p99_\text{prod}
  $$
* **DoD** : simulate latency spike → auto-rollback

---

## 4 – Data compliance & retention (P1)

### 4.1  Soft-delete & tombstone

* [x] Flag `deleted_at` TIMESTAMP on nodes & vectors
* [x] Lambda purge after 30 d
* **DoD** : GDPR “undo” possible durant 30 d

### 4.2  Toxic log retention

* [x] S3 lifecycle rule : `ingest-toxic/` expire after 7 d
* **DoD** : bucket scan shows zero > 7 d

---

## 5 – Hypergraph R/T updates (P1)

* [x] **Flink job** window 30 s → write incremental edges
* [x] RedisGraph hot layer for queries p95 < 500 ms
* **DoD** : delta latency analytics 0.5 → 0.05 s

---

## 6 – Explainability & curator loop (P2)

* [x] UI `/ui/edge_review`
  [x] List edges Δλ>τ, accept/reject
  [x] PATCH Neo4j via backend
* [x] Version history `edge_repair_log`
* **DoD** : curator merges patch, audit log ok

---

## 7 – TDA H₂ & UMAP lens adaptatif (P2)

### 7.1  H₂ persistance sketch

* [x] Use GUDHI `persistence(p=2)`
  [x] MinHash 512-bit signature
* **DoD** : signature added to embedding; recall +0.3 %

### 7.2  UMAP lens selection

* [x] Compute trustworthiness $T(u)$ for candidate lens
  Choose lens s.t. $T>0.95$
* **DoD** : Mapper stability (N clusters var < 5 %)

---

## 8 – Embedding drift → alert & retrain (P2)

* [x] Threshold schedule : warn 0.07, crit 0.1
* [x] Airflow task `trigger_retrain` if crit
* **DoD** : drift simulation triggers DAG

---

## 9 – Cost dashboards & GC (P2)

* [x] Grafana : panel `gpu_minutes_total per tenant`
* [x] Cron `gc_checkpoints.sh`: delete >30 d & not best
* **DoD** : disk usage stable < 80 %

---

### KPI global v3.0

| KPI                             | Cible        |
| ------------------------------- | ------------ |
| Cross-tenant isolation breaches | 0            |
| GPU cost accounted              | 100 %        |
| Canary rollback time            | < 2 min      |
| Real-time query latency         | p95 < 500 ms |
| Drift resolved <                | 24 h         |

**Lorsque toutes les cases seront cochées, Datacreek atteindra la maturité SaaS v3.0.**

### History
- Reset AGENTS for v3.0 checklist and added Airflow DAG with tenant XCom propagation.
- Added MLflow trainer callback logging params, metrics and model artifact with tests.
- Implemented Feast feature repository with cached vector lookups and unit tests.
- Added Ray Serve tenant-aware deployment with header-based routing and tests.
- Introduced Helm template for tenant namespaces enforcing quotas and egress isolation with tests.
- Added Accelerate ZeRO-3 config and gradient accumulation utility with tests.
- Introduced Neo4j Fabric client for tenant-scoped queries and migration paths with tests.
- Added GPU credit Prometheus exporter and quota controller with unit tests.
- Implemented Ray Serve canary routing with p99 latency tracking and auto-rollback tests.
- Added soft-delete utilities with `deleted_at` timestamps and purge logic, including unit tests.
- Enforced S3 lifecycle rule expiring `ingest-toxic/` objects after 7 days with tests.
- Added drift detection utilities with warn/critical thresholds and an Airflow DAG that triggers retraining when critical.
- Added GPU usage Grafana dashboard and checkpoint GC script with unit tests.
- Added GPU node group Helm template and job-spec builder with spot tolerations and restart policy tests.
- Implemented Flink-style edge stream processor and RedisGraph hot layer tracking p95 latency with tests.
- Built FastAPI edge review UI with Neo4j-backed accept/reject and version log.
- Added persistence MinHash sketches and UMAP lens selector with trustworthiness metric.
- Replaced external Feast dependency with lightweight in-tree stubs to keep unit
  tests self-contained.
- Expanded fine-tuning DAG tests to ensure tenant ID propagates through all downstream tasks.
- Introduced GPU cost tracking metric and helper computing $C=t_\text{GPU}\times P_\text{unit}$ with tests.
- Added balance query and credit top-up methods to GPU quota controller with tests.
- Formatted DAG and Ray Serve test files via pre-commit and reran targeted tests.
- Added tenant price update method to GPU quota controller with tests.
- Ensured tests can import local packages by inserting the repository root into ``sys.path``.
