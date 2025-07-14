Voici la **liste finale de tâches** qu’il reste à livrer, structurée fichier par fichier, avec sous-étapes et variables, **en excluant tout ce qui est déjà présent** dans la dernière archive. Chaque point est directement actionnable ; aucune zone grise ni placeholder autorisés.

---

## 1 │ `datacreek/core/knowledge_graph.py`

| Tâche                              | Sous-étapes détaillées                                                                                                                                           | Maths / variables                                                  |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **1-A Démarrer le watcher YAML**   | 1. Dans `datacreek/__init__.py` (ou point d’entrée CLI), appeler `start_cleanup_watcher(cfg_path)`. 2. Log INFO : "CFG-HOT watcher started" ([PyPI][1])        | `cfg_path = os.getenv("DATACREEK_CONFIG", "configs/default.yaml")` |
| **1-B Vérification seuils actifs** | 1. Ajouter `verify_thresholds()` juste avant chaque appel `cleanup_graph`. 2. Lever `RuntimeError` si `CleanupConfig.tau` ne correspond plus aux valeurs en BDD. |                                                                    |

---

## 2 │ `datacreek/analysis/generation.py`

| Tâche                                 | Sous-étapes                                                                                                                                                                           | Formule / var.                              |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **2-A Appliquer les logits corrigés** | 1. Dans `generate_chatml` et `generate_alpaca`, remplacer `payload["logits"]` par `scaled_logits` retourné par `bias_wasserstein`. 2. Ajouter test ▶ logits majoritaire ↓ si `W>0.1`. | $β = e^{-W}$ → `scaled_logits = logits * β` |

---

## 3 │ `datacreek/analysis/monitoring.py`

| Tâche                          | Sous-étapes                                                                                                                                             | Var. |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| **3-A Activer gauge `j_cost`** | 1. Déclarer `j_cost = Gauge('autotune_cost','Current J(theta)')` ([prometheus.github.io][2]). 2. Dans `autotune.update_theta`, appeler `j_cost.set(J)`. |      |

---

## 4 │ `datacreek/analysis/mapper.py`

| Tâche                    | Sous-étapes                                                                                                                                                                                        | Réf.                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **4-A Eviction L2 LMDB** | 1. Après chaque écriture, vérifier `env.stat()['psize'] * env.info()['map_size'] > cfg.cache.l2_max_mb<<20` (Mo). 2. Si dépassement, supprimer la plus ancienne clé via curseur puis `env.sync()`. | LMDB map\_size & freelist pages ([Stack Overflow][3])([Google Groups][4]) |

---

## 5 │ `tests/test_pipeline_e2e.py`

| Tâche                         | Assertions à coder                                                                                                | Notes                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **5-A Exécution CLI dry-run** | • `sheaf_score >= 0.8` • `recall10 >= 0.9` • `tpl_w1 <= cfg.tpl.eps_w1` • `(index_type=="HNSW") == (latency>0.1)` | Utiliser dataset `samples/mini`. |
| **5-B Intégration CI**        | Ajouter job `pipeline-e2e` dans `.github/workflows/python.yml`.                                                   |                                  |

---

## 6 │ `core/knowledge_graph.py` (validation)

| Tâche               | Étapes                                                                                                                      | Maths |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----- |
| **6-A Reload test** | 1. Modifier `configs/default.yaml` (tau +=1). 2. Vérifier via log que la modification est prise en compte sans redémarrage. | —     |

---

## 7 │ Docs & Config

| Tâche                       | Contenu                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **7-A README – Hot reload** | • Décrire la variable `DATACREEK_CONFIG`. • Expliquer que le watcher recharge `cleanup.*` toutes les 5 min. |

---

### Variables / Paramètres à ajouter

```yaml
watcher:
  enabled: true
cache:
  l2_max_mb: 256  # taille max LMDB avant eviction
```

---

#### Références rapides

* **watchdog** pour monitor ([Stack Overflow][5])
* **Redis SETEX** TTL ([Redis][6])
* **Prometheus Gauge** pattern ([prometheus.github.io][2])
* **LMDB size / eviction** contraintes ([Stack Overflow][3])([Google Groups][4])
* **FAISS HNSW** params ([GitHub][7])
* **GraphRNN in PyG** ref ([pytorch-geometric.readthedocs.io][8])
* **NumPy seed best practice** ([numpy.org][9])

---

Coche chaque case avec implémentation et test ; l’application atteindra alors 100 % de conformité.

[1]: https://pypi.org/project/watchdog/?utm_source=chatgpt.com "watchdog - PyPI"
[2]: https://prometheus.github.io/client_python/instrumenting/gauge/?utm_source=chatgpt.com "Gauge | client_python - Prometheus"
[3]: https://stackoverflow.com/questions/63552889/maximum-lmdb-value-size?utm_source=chatgpt.com "Maximum LMDB value size - python - Stack Overflow"
[4]: https://groups.google.com/g/caffe-users/c/0RKsTTYRGpQ?utm_source=chatgpt.com "How to assign proper mapsize for LMDB? - Google Groups"
[5]: https://stackoverflow.com/questions/73406981/restart-a-file-on-change-python-watchdog?utm_source=chatgpt.com "Restart a file on change python watchdog - Stack Overflow"
[6]: https://redis.io/docs/latest/commands/setex/?utm_source=chatgpt.com "SETEX | Docs - Redis"
[7]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes?utm_source=chatgpt.com "Faiss indexes · facebookresearch/faiss Wiki - GitHub"
[8]: https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/nn.html?utm_source=chatgpt.com "torch_geometric.nn - PyTorch Geometric - Read the Docs"
[9]: https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html?utm_source=chatgpt.com "numpy.random.seed — NumPy v2.2 Manual"
