# Agent Tasks

## 1 │ `datacreek/core/knowledge_graph.py`

| Tâche                              | Sous-étapes détaillées                                                                                                                                           | Maths / variables                                                  | Status |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------ |
| **1-A Démarrer le watcher YAML**   | 1. Dans `datacreek/__init__.py` (ou point d’entrée CLI), appeler `start_cleanup_watcher(cfg_path)`. 2. Log INFO : "CFG-HOT watcher started" ([PyPI][1])        | `cfg_path = os.getenv("DATACREEK_CONFIG", "configs/default.yaml")` | [x] |
| **1-B Vérification seuils actifs** | 1. Ajouter `verify_thresholds()` juste avant chaque appel `cleanup_graph`. 2. Lever `RuntimeError` si `CleanupConfig.tau` ne correspond plus aux valeurs en BDD. |                                                                    | [x] |

## 2 │ `datacreek/analysis/generation.py`

| Tâche                                 | Sous-étapes                                                                                                                                                                           | Formule / var.                              | Status |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------ |
| **2-A Appliquer les logits corrigés** | 1. Dans `generate_chatml` et `generate_alpaca`, remplacer `payload["logits"]` par `scaled_logits` retourné par `bias_wasserstein`. 2. Ajouter test ▶ logits majoritaire ↓ si `W>0.1`. | $β = e^{-W}$ → `scaled_logits = logits * β` | [x] |

## 3 │ `datacreek/analysis/monitoring.py`

| Tâche                          | Sous-étapes                                                                                                                                             | Var. | Status |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------ |
| **3-A Activer gauge `j_cost`** | 1. Déclarer `j_cost = Gauge('autotune_cost','Current J(theta)')` ([prometheus.github.io][2]). 2. Dans `autotune.update_theta`, appeler `j_cost.set(J)`. |      | [x] |

## 4 │ `datacreek/analysis/mapper.py`

| Tâche                    | Sous-étapes                                                                                                                                                                                        | Réf.                                                                      | Status |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------ |
| **4-A Eviction L2 LMDB** | 1. Après chaque écriture, vérifier `env.stat()['psize'] * env.info()['map_size'] > cfg.cache.l2_max_mb<<20` (Mo). 2. Si dépassement, supprimer la plus ancienne clé via curseur puis `env.sync()`. | LMDB map_size & freelist pages ([Stack Overflow][3])([Google Groups][4]) | [x] |

## 5 │ `tests/test_pipeline_e2e.py`

| Tâche                         | Assertions à coder                                                                                                | Notes                            | Status |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------ |
| **5-A Exécution CLI dry-run** | • `sheaf_score >= 0.8` • `recall10 >= 0.9` • `tpl_w1 <= cfg.tpl.eps_w1` • `(index_type=="HNSW") == (latency>0.1)` | Utiliser dataset `samples/mini`. | [x] |
| **5-B Intégration CI**        | Ajouter job `pipeline-e2e` dans `.github/workflows/python.yml`.                                                   |                                  | [x] |

## 6 │ `core/knowledge_graph.py` (validation)

| Tâche               | Étapes                                                                                                                      | Maths | Status |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----- | ------ |
| **6-A Reload test** | 1. Modifier `configs/default.yaml` (tau +=1). 2. Vérifier via log que la modification est prise en compte sans redémarrage. | —     | [x] |

## 7 │ Docs & Config

| Tâche                       | Contenu                                                                                                     | Status |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | ------ |
| **7-A README – Hot reload** | • Décrire la variable `DATACREEK_CONFIG`. • Expliquer que le watcher recharge `cleanup.*` toutes les 5 min. | [x] |

### Variables / Paramètres ajoutés
```yaml
watcher:
  enabled: true
cache:
  l2_max_mb: 256  # taille max LMDB avant eviction
```

### Références rapides
- [watchdog](https://stackoverflow.com/questions/73406981/restart-a-file-on-change-python-watchdog?utm_source=chatgpt.com)
- [Redis SETEX](https://redis.io/docs/latest/commands/setex/?utm_source=chatgpt.com)
- [Prometheus Gauge](https://prometheus.github.io/client_python/instrumenting/gauge/?utm_source=chatgpt.com)
- [LMDB size / eviction](https://stackoverflow.com/questions/63552889/maximum-lmdb-value-size?utm_source=chatgpt.com) ([Google Groups](https://groups.google.com/g/caffe-users/c/0RKsTTYRGpQ?utm_source=chatgpt.com))
- [FAISS HNSW](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes?utm_source=chatgpt.com)
- [GraphRNN in PyG](https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/nn.html?utm_source=chatgpt.com)
- [NumPy seed](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html?utm_source=chatgpt.com)

## History
- Initial integration of YAML watcher, verify_thresholds checks, logit scaling and monitoring gauge.
- Implemented LMDB eviction, pipeline dry-run tests and CI job.
- Added hot reload documentation and config parameters.
