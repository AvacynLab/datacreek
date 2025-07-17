----------
Voici la **check‑list finale** : quatre chantiers, chacun découpé en sous‑tâches concrètes ; pour chaque point je rappelle l’objectif, le correctif précis et les variables / formules associées. À la fin, quand toutes les cases seront cochées, la code‑base atteindra la conformité complète fixée précédemment.

---

## 1  | Métrique `redis_hit_ratio` + TTL adaptatif (Cache L1)

| ☑ | Sous‑tâche                      | Détail / code                                                                                                                                          | Objectif / variable            |
| - | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| ☑ | **1.1 Enregistrer hits / miss** | Décorateur `@cache_l1` :<br>`redis.incr('hits')` si key existe,<br>`redis.incr('miss')` sinon.                                                         | Compteur précis pour ratio.    |
| ☑ | **1.2 Gauge Prometheus**        | `python
redis_hit_ratio = Gauge('redis_hit_ratio', 'Hit ratio L1')
`                                                                                 | Exposé : `hits / (hits+miss)`. |
| ☑ | **1.3 Boucle d’ajustement TTL** | Toutes les 5 min :<br>`python
ratio = hits / max(1,hits+miss)
if ratio < 0.2: ttl = max(ttl*0.5, 300)
elif ratio > 0.8: ttl = min(ttl*1.2, 7200)
` | Formule TTL adaptatif.         |
| ☑ | **1.4 Appliquer nouveau TTL**   | `redis.expire(key, ttl)` pour les clés nouvellement mises en cache.                                                                                    | TTL dynamique par charge.      |
| ☑ | **1.5 Config YAML**             | `yaml
cache:
  l1_ttl_init: 3600
  l1_ttl_min: 300
  l1_ttl_max: 7200
`                                                                           | Paramètres centralisés.        |

---

## 2  | Compteur `gp_jitter_restarts_total` (Autotuner)

| ☑ | Sous‑tâche               | Détail / code                                                                                                     | Objectif                 |
| - | ------------------------ | ----------------------------------------------------------------------------------------------------------------- | ------------------------ |
| ☑ | **2.1 Déclarer Counter** | `python
gp_jitter_restarts_total = Counter('gp_jitter_restarts_total',
    'Number of SVGP jitter restarts')
` | Visibilité des restarts. |
| ☑ | **2.2 Incrémenter**      | Dans `autotune.py`, bloc early‑stop :<br>`gp_jitter_restarts_total.inc()` juste avant le refit GP.                | Métrique fiable.         |

---

## 3  | Index Neo4j composite pour Hyper‑AA

| ☑ | Sous‑tâche          | Cypher / script                                                                                                     | Objectif                       |
| - | ------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| ☑ | **3.1 Créer index** | `cypher
CREATE INDEX haa_pair IF NOT EXISTS
FOR ()-[r:SUGGESTED_HYPER_AA]-()
ON (r.startNodeId, r.endNodeId);
` | Recherche pair (u,v) O(log n). |
| ☑ | **3.2 Migration**   | Script de maintenance exécuté une fois lors du déploiement.                                                         | Pas de scan complet.           |

---

## 4  | Histogramme latence ANN

| ☑ | Sous‑tâche                 | Détail / code                                                                                                                                 | Objectif                   |
| - | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| ☑ | **4.1 Déclarer Histogram** | `python
ann_latency = Histogram(
  'ann_latency_seconds',
  'Latency of ANN queries',
  buckets=(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10))
` | Traquer P95 et cold‑start. |
| ☑ | **4.2 Décorer search**     | `python
with ann_latency.time():
    index.search(q, k)
`                                                                                  | Mesure automatique.        |

---

### Variables et formules introduites

| Variable                   | Description / formule                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------- |
| `redis_hit_ratio`          | hits / (hits + miss)                                                                              |
| TTL adaptatif              | `ttl' = clamp(ttl×0.5, ttl_min)` si ratio < 0.2 ; `ttl' = clamp(ttl×1.2, ttl_max)` si ratio > 0.8 |
| `gp_jitter_restarts_total` | Counter Prometheus                                                                                |
| `ann_latency_seconds`      | Histogram latence ANN                                                                             |

---

## Checklist
- [x] Métrique redis_hit_ratio + TTL adaptatif
- [x] Compteur gp_jitter_restarts_total
- [x] Index Neo4j composite pour Hyper-AA
- [x] Histogramme latence ANN

## History
- Added Neo4j HAA index migration script and test.
- Reset tasks list as per user specification.
- Implemented cache_l1 decorator, adaptive TTL config and tests.
- Verified metrics implementations and ran unit tests.
- Added TTL adjustment unit test and installed test dependencies.
- Verified test environment after dependency installs.
