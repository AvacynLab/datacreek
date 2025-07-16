## Synthèse rapide

Il ne reste qu’un **noyau compact de sept chantiers** avant d’atteindre la conformité totale : filtrage Hyper-AA, persistance CCA, activation réelle du biais-Wasserstein, histogramme de latence ANN, seed bootstrap fractal, ajout des clés YAML manquantes et documentation du rollback FractalNet. Ci-dessous, chaque tâche est déroulée en sous-étapes sans placeholder ; j’indique les formules à respecter ainsi que les variables de configuration indispensables.

---

## 1 | Filtrage & indexation Hyper-Adamic–Adar

### Sous-étapes

1. **Appel GDS**

   ```cypher
   CALL gds.alpha.hypergraph.linkprediction.adamicAdar.write(
     $graphName,
     {writeRelationshipType:'SUGGESTED_HYPER_AA',
      writeProperty:'score',
      topK: cfg.cleanup.lp_topk});
   ```
2. **Filtre par score**

   ```cypher
   MATCH ()-[r:SUGGESTED_HYPER_AA]-()
   WHERE r.score < cfg.cleanup.lp_sigma
   DELETE r;
   ```
3. **Index Neo4j**

   ```cypher
   CREATE INDEX haa_score IF NOT EXISTS
   FOR ()-[r:SUGGESTED_HYPER_AA]-() ON (r.score);
   ```
4. **Prometheus**

   * Counter : `haa_edges_total`.

### Math

$$
s_{HAA}(u,v)=\sum_{e\ni u,v}\frac{1}{\log(|e|-1)}
$$

Conserver si $s_{HAA}\ge\lambda_{\text{HAA}}$.

### Variables YAML

```yaml
cleanup:
  lp_topk: 50
  lp_sigma: 0.30
```

---

## 2 | Persistance des matrices CCA

### Sous-étapes

1. Après entraînement, sérialiser :

   ```python
   with open('cache/cca.pkl','wb') as f:
       pickle.dump({'Wn2v': W1, 'Wgw': W2}, f)
   ```
2. Calculer le SHA-256 du fichier :

   ```python
   sha = hashlib.sha256(Path('cache/cca.pkl').read_bytes()).hexdigest()
   logger.info(f"cca_sha={sha}")
   ```
3. Au chargement, si le pickle existe, sauter le recalcul CCA.

---

## 3 | Mitigation biais Wasserstein

### Sous-étapes

1. **Histogrammes**

   * `local_hist` : distribution démographique des *k* nœuds du prompt.
   * `global_hist` : distribution globale (stockée en cache).
2. **Distance Sinkhorn**

   ```python
   W = SamplesLoss("sinkhorn", blur=0.01)(local_hist, global_hist)
   ```
3. **Re-pondération**

   ```python
   if W > 0.1:
       logits *= math.exp(-W)
   ```
4. **Gauge**

   * `bias_wasserstein_last = W`.

---

## 4 | Rollback du pruner FractalNet

### Sous-étapes

1. **Tester perplexité**

   ```python
   ppl_new = model.evaluate(val_loader)
   ratio = ppl_new / ppl_old
   ```
2. **Condition**

   * Si `ratio > 1.01` → restaurer checkpoint avant pruning ; log `PRUNE_REVERTED=true`.
3. **Logs**

   * `prune_ratio`, `ppl_delta`, `was_reverted`.

---

## 5 | Cache LMDB — TTL & éviction

### Sous-étapes

1. Ajouter clés YAML :

   ```yaml
   cache:
     l2_max_size_mb: 2048
     l2_ttl_hours: 24
   ```
2. Thread d’entretien :

   ```python
   for key,val in lmdb_cursor:
       if val.age > cfg.cache.l2_ttl_hours or db_size()>cfg.cache.l2_max_size_mb:
           delete(key)
   ```

---

## 6 | Monitoring — trois gauges manquants

### Implémentation (`monitoring.py`)

```python
sheaf_score_g   = Gauge('sheaf_score',   'Average sheaf consistency')
tpl_w1_g        = Gauge('tpl_w1',        'Last TPL Wasserstein-1')
autotune_cost_g = Gauge('autotune_cost', 'Current J(theta)')
```

* **sheaf_score** : moyenne des scores batch Δx=b.
* **tpl_w1** : distance W₁ du dernier cycle TPL.
* **autotune_cost** : valeur finale de J(θ) à chaque itération BO.

---

## 7 | Seed bootstrap fractal

### Sous-étapes

1. Ajouter clé YAML :

   ```yaml
   fractal:
     bootstrap_seed: 42
   ```
2. Dans `core/fractal.py` :

   ```python
   random.seed(cfg.fractal.bootstrap_seed)
   np.random.seed(cfg.fractal.bootstrap_seed)
   ```

---

## 8 | Histogramme de latence ANN

### Sous-étapes

1. Ajouter :

   ```python
   ann_latency = Histogram('ann_latency_seconds',
                           'Latency of ANN queries',
                           buckets=(.01,.02,.05,.1,.2,.5,1,2))
   ```
2. Time chaque `index.search`; observe latence.

---

## 9 | Documentation rollback FractalNet

* Ajouter au `README.md` (section Compression) :

  > *Si la perplexité après pruning dépasse 1 % de la référence, le pruner restaure automatiquement le checkpoint précédent.*

---

### Variables / Formules à créer

| Variable                 | Description           |
| ------------------------ | --------------------- |
| `haa_edges_total`        | Prometheus counter    |
| `bias_wasserstein_last`  | Dernier W Sinkhorn    |
| `sheaf_score`            | Gauge (Δx=b)          |
| `tpl_w1`                 | Gauge Wasserstein TPL |
| `autotune_cost`          | Gauge J(θ)            |
| `fractal.bootstrap_seed` | Seed RNG              |

---

✔️ **Quand toutes ces cases seront cochées**, aucun élément du mémo d’architecture ne restera en suspens.

## Checklist
- [x] Filtrage & indexation Hyper-Adamic–Adar
- [x] Persistance des matrices CCA
- [x] Mitigation biais Wasserstein
- [x] Rollback du pruner FractalNet
- [x] Cache LMDB — TTL & éviction
- [x] Monitoring — trois gauges manquants
- [x] Seed bootstrap fractal
- [x] Histogramme de latence ANN
- [x] Documentation rollback FractalNet

## History
- Reset tasks list and cleared previous progress.
- Renamed monitoring gauges with `_g` suffix and updated imports.
- Installed missing dependencies and ran tests successfully.
