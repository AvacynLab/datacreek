----------
### ✔️ Mega‑Checklist à cocher – *Tech‑Debt Sweep* vers une **v1‑beta “production‑grade”**

*(tout ce qui n’est **pas** listé ci‑dessous a déjà été validé)*

---

## 1  | Stabilité & performance des algos spectraux

### 1.1  Fallback eigsh → Lanczos‑5

* [ ] **Condition timeout**

  ```python
  try:
      lmax = eigsh(L, k=1, which='LM', tol=1e-3, maxiter=cfg.spectral.eig_maxit)[0]
  except ConvergenceError:
      lmax = lanczos_top_eigenvalue(L, k=5)
  ```
* [ ] **Implémenter `lanczos_top_eigenvalue`** (5 itérations, orthogonalisation)

  $$
    l_{\max}\approx \frac{v_5^\top L v_5}{v_5^\top v_5}
  $$
* [ ] **Paramètres YAML**

  ```yaml
  spectral:
    eig_maxit: 2000
  ```

### 1.2  Histogramme latence FAISS/HNSW

* [ ] Ajouter histogramme Prometheus

  ```python
  ann_latency = Histogram(
      'ann_latency_seconds', 'ANN query latency',
      buckets=(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10))
  ```
* [ ] Décorer chaque appel `index.search` :

  ```python
  with ann_latency.time():
      index.search(...)
  ```

---

## 2  | Caching – gérer la croissance

### 2.1  TTL adaptatif Redis L1

* [ ] **Ajout métrique `redis_hit_ratio`**.
* [ ] **Ré‑évaluer TT L** toutes les 5 min :

  ```
  if hit_ratio < 0.2: ttl *= 0.5
  elif hit_ratio > 0.8: ttl = min(ttl*1.2, 7200)
  redis.config_set("maxmemory-policy", "allkeys-lru")
  ```

### 2.2  Limite taille LMDB L2

* [ ] Fixer `set_mapsize(cfg.cache.l2_max_size_mb*1024**2)`.
* [ ] Boucle éviction :

  ```python
  if lmdb_stat()['psize_mb'] > cfg.cache.l2_max_size_mb:
      evict_oldest_entries()
  ```

---

## 3  | Bias mitigation raffinée

### 3.1  Clamp Wasserstein

* [ ] Avant re‑pondération :

  ```python
  W = min(W, 0.5)      # clamp
  logits *= math.exp(-W)
  ```

### 3.2  Exporter métrique

* [ ] Gauge `bias_wasserstein_last`.

---

## 4  | Observabilité – métriques manquantes

| Gauge Prometheus           | Source        | Implémentation                    |
| -------------------------- | ------------- | --------------------------------- |
| `gp_jitter_restarts_total` | `autotune.py` | incrémenter à chaque restart SVGP |
| `redis_hit_ratio`          | Cache L1 loop | hits / (hits+miss)                |

---

## 5  | Indexation Neo4j – composite sur Hyper‑AA

* [ ] Créer index :

  ```cypher
  CREATE INDEX haa_pair IF NOT EXISTS
  FOR ()-[r:SUGGESTED_HYPER_AA]-()
  ON (r.startNodeId, r.endNodeId);
  ```

---

## 6  | Reload Hot‑YAML

### 6.1  File watcher

* [ ] Utiliser **watchdog** :

  ```python
  def on_modified(event):
      Config.reload()
  ```
* [ ] Protège par verrou lecteur/écrivain pour la config globale.

---

## 7  | Portabilité / dev‑experience

### 7.1  Variable d’environnement cache

* [ ] Préfixer tous les chemins `cache/` par `os.environ.get("DATACREEK_CACHE", "./cache")`.

### 7.2  Checkpoint GraphRNN sur S3

* [ ] Télécharger au boot :

  ```python
  s3 = boto3.client('s3')
  s3.download_file(bucket, key, local_path, ExtraArgs={'ChecksumMode':'ENABLED'})
  ```
* [ ] Vérifier `hashlib.sha256(local).hexdigest()==cfg.tpl.rnn_ckpt_sha`.

---

## 8  | Pruner FractalNet – rollback sécurisé

### 8.1  Condition

* $\text{ratio} = \frac{\text{ppl\_new}}{\text{ppl\_old}}$.
* Si `ratio > 1.01` → restore checkpoint, log `PRUNE_REVERTED=true`, counter `prune_reverts_total`++.

---

## 9  | Seed reproductible – fractal bootstrap

* [ ] Clé YAML :

  ```yaml
  fractal:
    bootstrap_seed: 42
  ```
* [ ] Avant sampling :

  ```python
  random.seed(cfg.fractal.bootstrap_seed)
  np.random.seed(cfg.fractal.bootstrap_seed)
  ```

---

## 10  | Docs & CI

### 10.1  Readme

* [ ] Ajouter tableau « Rollback policy » (prune + DP).

### 10.2  GitHub Actions

* [ ] Spin‑up Neo4j container, run `pytest -m "e2e"` sur un mini‑graph 50 k edges.

---

### Variables / formules introduites

| Var                        | Description                 |
| -------------------------- | --------------------------- |
| `spectral.eig_maxit`       | max itérations dans `eigsh` |
| `bias_wasserstein_last`    | W après re‑pondération      |
| `gp_jitter_restarts_total` | # restarts SVGP             |
| `redis_hit_ratio`          | cache L1 hits/(hits+miss)   |
| `fractal.bootstrap_seed`   | seed RNG fractal bootstrap  |

---

✅ **Objectif atteint** quand :

* `haa_edges_total` > 0 & index `(startNodeId,endNodeId)` créé.
* Prometheus montre `sheaf_score`, `tpl_w1`, `autotune_cost`, `gp_jitter_restarts_total`, `redis_hit_ratio`.
* CI passe e2e, ANN latence < 0.2 s P95, no eigsh timeout.

## Checklist

- [ ] Stabilité & performance des algos spectraux
- [ ] Caching – gérer la croissance
- [ ] Bias mitigation raffinée
- [ ] Observabilité – métriques manquantes
- [ ] Indexation Neo4j – composite sur Hyper‑AA
- [ ] Reload Hot‑YAML
- [ ] Portabilité / dev‑experience
- [ ] Pruner FractalNet – rollback sécurisé
- [ ] Seed reproductible – fractal bootstrap
- [ ] Docs & CI

## History
- Reset tasks list.
