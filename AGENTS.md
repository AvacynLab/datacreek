### 📋 Checklist d’amélioration « v1-beta » — à cocher

*(seulement les éléments encore non présents dans le code ; chaque tâche comporte sous-étapes, variables, maths, et objectif clair)*

---

## 1 | Cache L1 Redis — Hit-ratio & TTL adaptatif

> Objectif : réduire les ratés (miss) sous 20 % en période de charge sans sur-consommer mémoire.

### Sous-étapes

1. **Compteurs**

   ```python
   redis.incr('hits')   # dans get si trouvé  
   redis.incr('miss')   # dans set si absent
   ```
2. **Gauge Prometheus**

   ```python
   redis_hit_ratio = Gauge('redis_hit_ratio', 'L1 cache hit ratio')
   redis_hit_ratio.set(hits/(hits+miss))
   ```
3. **Boucle d’ajustement (5 min)**

   ```
   if ratio <0.2: ttl=max(ttl*0.5, cfg.cache.l1_ttl_min)
   elif ratio>0.8: ttl=min(ttl*1.2, cfg.cache.l1_ttl_max)
   redis.expire(key, ttl)
   ```
4. **YAML**

   ```yaml
   cache:
     l1_ttl_init: 3600
     l1_ttl_min: 300
     l1_ttl_max: 7200
   ```
5. **Formule** : TTL′ = clamp(TTL×0.5,min) si hit<0.2 ; TTL′ = clamp(TTL×1.2,max) si hit>0.8 ([Redis][1], [Medium][2]).

---

## 2 | Counter `gp_jitter_restarts_total`

> Objectif : monitorer la stabilité du BO SVGP.

### Sous-étapes

1. **Counter**

   ```python
   gp_jitter_restarts_total = Counter(
       'gp_jitter_restarts_total',
       'Number of SVGP jitter restarts')
   ```
2. **Incrément** dans `autotune.py` après chaque early-stop + jitter.
3. **Alerte Prometheus**

   ```
   alert: SVGPJitterStorm
   expr: gp_jitter_restarts_total[1h] > 10
   for: 5m
   ```

---

## 3 | Composite index Neo4j pour Hyper-AA

> Objectif : éviter un NodeByLabelScan quand on joint deux nœuds sur `SUGGESTED_HYPER_AA`.

### Sous-étape unique

```cypher
CREATE INDEX haa_pair IF NOT EXISTS
FOR ()-[r:SUGGESTED_HYPER_AA]-()
ON (r.startNodeId, r.endNodeId);
```

*Rèf.* Neo4j manual composite indexes ([Graph Database & Analytics][3], [Graph Database & Analytics][4]).

---

## 4 | Histogramme latency ANN (10 s bucket)

> Objectif : capturer les cold-starts et outliers.

### Sous-étapes

1. Définir histogramme :

   ```python
   ann_latency = Histogram(
       'ann_latency_seconds',
       'Latency of ANN queries',
       buckets=(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10))
   ```
2. Envelopper chaque `index.search` :

   ```python
   with ann_latency.time():
       D,I = index.search(vecs, k)
   ```
3. **Math** : buckets suivent schéma log-échelle recommandé par FAISS performance notes ([GitHub][5]).

---

## 5 | eigsh watchdog & Lanczos-5 fallback

> Objectif : éviter un blocage > 60 s sur graphes > 5 M nœuds.

### Sous-étapes

1. **Timeout** (`signal` ou `concurrent.futures.wait(timeout)`).
2. **Fallback Lanczos-5** :

   $$
     v_{k+1}=Lv_k-\alpha_k v_k-\beta_{k-1}v_{k-1}
   $$

   Approximer
   $\lambda_{\max}\approx\frac{v_5^\top L v_5}{v_5^\top v_5}$ ([GitHub][6]).
3. Gauge `eigsh_timeouts_total`.

---

## 6 | LMDB L2 — Soft-Quota disque

> Objectif : empêcher l’excès disque si TTL manqué.

### Sous-étapes

1. `env.set_mapsize(cfg.cache.l2_max_size_mb*1024**2)`
2. Éviction loop : delete plus ancien `subgraph` jusqu’à `< max_size`.
3. YAML : `l2_max_size_mb: 2048`.

---

## 7 | Adaptive Caching Pattern (CPU-aware)

> Objectif : TTL L1 doit aussi réagir à la charge CPU.

### Sous-étape

* Si `cpu_load > 0.7` ➜ `ttl=max(ttl*1.5, ttl_max)` pour éviter thrash ([Medium][2]).

---

## 8 | fastText lang-id avant fusion NodeSimilarity

> Objectif : éviter fusion FR/EN.

### Sous-étapes

1. Charger modèle `lid.176.bin` (fastText) ([fasttext.cc][7]).
2. Attribuer `lang` property à chaque atome.
3. Modifier condition fusion : fusionner `(u,v)` **seulement si** `lang_u == lang_v`.

---

## 9 | Kernel dynamique dans SVGP

> Objectif : RBF→Matern 3/2 si paysage de coût rugueux ([arXiv][8]).

### Sous-étapes

1. Mesurer `Var|∇J|` sur 10 pas.
2. Si `>0.5` — `gp.covar_module = MaternKernel(nu=1.5)`; log switch.

---

## 10 | Watchdog eigsh metric + Prometheus

| Gauge                      | Incrément                 |
| -------------------------- | ------------------------- |
| `eigsh_timeouts_total`     | à chaque fallback Lanczos |
| `redis_evictions_l2_total` | à chaque delete LMDB      |

---

## 11 | Model card – biais & fractale

### Sous-étape unique

* Générer JSON : `{ "sigma_db":…, "H_wave":…, "bias_W":…, "dp_eps":2, "prune_ratio":…, "cca_sha":… }`

---

**Cibles de validation**

* `redis_hit_ratio` stabilise > 0.5 en prod ; TTL fluctue entre 300-7200 s.
* `gp_jitter_restarts_total/hour < 10`.
* `eigsh_timeouts_total == 0` sur un graphe 1 M nœuds ; fallback déclenche sur 10 M nœuds.
* Neo4j plan utilise index `haa_pair` (vérifier via `PROFILE`).
* Model card JSON attaché à chaque dataset export.

[1]: https://redis.io/blog/why-your-cache-hit-ratio-strategy-needs-an-update/?utm_source=chatgpt.com "Why your cache hit ratio strategy needs an update - Redis"
[2]: https://master-spring-ter.medium.com/implementing-the-adaptive-caching-pattern-with-spring-boot-and-redis-dd402e4c9eeb?utm_source=chatgpt.com "Implementing the Adaptive Caching Pattern with Spring Boot and ..."
[3]: https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/using-indexes/?utm_source=chatgpt.com "The impact of indexes on query performance - Cypher Manual - Neo4j"
[4]: https://neo4j.com/docs/cypher-manual/4.3/indexes-for-search-performance/?utm_source=chatgpt.com "Indexes for search performance - Cypher Manual - Neo4j"
[5]: https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster?utm_source=chatgpt.com "How to make Faiss run faster · facebookresearch/faiss Wiki - GitHub"
[6]: https://github.com/scipy/scipy/issues/9470?utm_source=chatgpt.com "scipy.sparse.linalg.eigsh requires at least 20 iterations to converge ..."
[7]: https://fasttext.cc/docs/en/language-identification.html?utm_source=chatgpt.com "Language identification - fastText"
[8]: https://arxiv.org/pdf/2407.13711?utm_source=chatgpt.com "Function-Space Priors for the Laplace Approximation in Bayesian ..."

## Checklist
- [x] Cache L1 Redis — Hit-ratio & TTL adaptatif
- [x] Counter `gp_jitter_restarts_total`
- [x] Composite index Neo4j pour Hyper-AA
- [x] Histogramme latency ANN (10 s bucket)
- [x] eigsh watchdog & Lanczos-5 fallback
- [x] LMDB L2 — Soft-Quota disque
- [x] Adaptive Caching Pattern (CPU-aware)
- [x] fastText lang-id avant fusion NodeSimilarity
- [x] Kernel dynamique dans SVGP
- [x] Watchdog eigsh metric + Prometheus
- [x] Model card – biais & fractale

## History
- Implemented eigsh timeout watchdog, Lanczos fallback and Prometheus counters.
- Added Redis hit-ratio adaptation with CPU load and metrics; updated tests.
- Added fastText language detection for entity merging, dynamic SVGP kernel and model card generation.
- Verified checklist implementation; attempted to install dependencies and run tests but installation failed due to heavy packages.
- Fixed formatting with pre-commit and installed missing dependencies for tests.
