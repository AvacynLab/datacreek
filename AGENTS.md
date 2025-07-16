## Task List

### 1. Écriture des arêtes Hyper‑Adamic–Adar
- [x] Appel effectif de l’algo GDS
- [x] Filtre de seuil
- [x] Métrique Prometheus `haa_edges_total`

### 2. Persistance des matrices CCA (multi‑vue)
- [x] Sérialisation après entraînement
- [x] Chargement lors de l’inférence
- [x] Traceabilité

### 3. Mitigation de biais Wasserstein dans la génération
- [x] Calculer `W` via `geomloss.SamplesLoss("sinkhorn", blur=0.01)`
- [x] Si `W > 0.1` : appliquer `logits *= np.exp(-W)` avant softmax
- [x] Gauge Prometheus `bias_wasserstein_last`

### 4. Rollback du pruner FractalNet
- [x] Calcul perplexité post‑prune
- [x] Condition de restauration
- [x] Logs `prune_ratio`, `ppl_delta`, `was_reverted`

### 5. Cache hiérarchique — TTL LMDB
- [x] Ajouter `cache.l2_max_size_mb` et `cache.l2_ttl_hours` dans YAML
- [x] Mettre en place un thread d’éviction

### 6. Monitoring — gauges manquants
- [x] `sheaf_score`
- [x] `tpl_w1`
- [x] `autotune_cost`

### 7. YAML — clé manquante
- [x] Ajouter `cleanup.lp_topk` (valeur par défaut 50)

## Information
Objectif terminé lorsque :
1. `haa_edges_total` > 0 dans Prometheus.
2. `cache/cca.pkl` existe et `cca_sha` est loggé.
3. Fichier de génération affiche `bias_wasserstein_last ≤ 0.1`.
4. Journal contient `PRUNE_REVERTED: false`.
5. Redis + LMDB indiquent une éviction TTL active.
6. Les trois gauges (`sheaf_score`, `tpl_w1`, `autotune_cost`) apparaissent dans `/metrics`.

## History
- Completed Hyper-AA, CCA cache, Wasserstein bias, LMDB eviction,
  FractalNet rollback logs, and monitoring gauges
- Implemented LMDB eviction thread with start/stop helpers and added unit test.
- Verified tasks, installed rich dependency and ran targeted tests. All pass.
- Rechecked metrics and TTL eviction; installed deps and executed tests.
- Tried full dependency install and subset tests; missing packages caused import errors.
