### ✅ "ULTIME TODO" — ce qu’il reste encore à livrer

*(format identique aux listes précédentes ; seules les **trois tâches** encore incomplètes sont listées)*

---

## 1 │ `datacreek/analysis/mapper.py` — Eviction LMDB (cache L2)

| Action | Sous‑étapes détaillées | Variables / formules |
| --- | --- | --- |
| **Mettre en place la purge LRU** | 1. Après chaque insertion :`with env.begin(write=True) as txn:` récupérer :<br>  `stat = env.stat()`, `info = env.info()` → `current_mb = (stat["psize"] * info["map_size"]) / (1<<20)`.<br>2. Si `current_mb > cfg.cache.l2_max_mb` :<br>  a. Ouvrir un curseur `cur = txn.cursor()` (ordre par défaut = insertion).<br>  b. Supprimer clés jusqu’à retomber sous la limite :`cur.delete()` ; compter `n_deleted`.<br>  c. `env.sync()`.<br>3. Log INFO : `"[L2-EVICT] %d keys purged (%.1f MB→%.1f MB)"`. | `l2_max_mb` (YAML) — quota mémoire MB pour LMDB. |

---

## 2 │ `datacreek/core/knowledge_graph.py` — Vérification seuils actifs

| Action | Sous‑étapes | Variables |
| --- | --- | --- |
| **Brancher `verify_thresholds()` dans la pipeline** | 1. Dans le script orchestrateur (`build_dataset.py` ou `pipeline.py`), juste avant l’appel de `cleanup_graph()`, insérer :`knowledge_graph.verify_thresholds()`.<br>2. `verify_thresholds()` doit lire les propriétés Neo4j (labels `CFG_TAU`, etc.) ou le singleton `CleanupConfig`, comparer aux valeurs YAML et lever :<br>`raise RuntimeError("CFG-HOT mismatch: tau≠%d" % yaml_tau)` si divergence. | |

---

## 3 │ `docs/README.md` (ou `DOCS.md`) — Documentation du hot‑reload

| Action | Contenu requis |
| --- | --- |
| **Ajouter section “♻ Hot reload des paramètres de nettoyage”** | 1. Expliquer la variable d’environnement **`DATACREEK_CONFIG`** : chemin absolu/relatif vers le fichier YAML à surveiller.<br>2. Décrire le watcher : délai = 5 minutes, message log “CFG‑HOT watcher started”, rechargement à la volée de `tau`, `sigma`, `k_min`, `lp_sigma`, `hub_deg`.<br>3. Mentionner la commande rapide :<br>`export DATACREEK_CONFIG=configs/cleanup_prod.yaml` puis `make run`. |

---

#### Après ces trois livraisons :

*Eviction L2, vérification de cohérence avant chaque cleanup, et documentation utilisateur complètent la conformité à 100 % ; la branche peut passer en **release candidate**.*

---

## Checklist

- [x] Eviction L2 LMDB
- [x] Brancher `verify_thresholds()` dans la pipeline
- [x] Documentation du hot-reload

## Historique

- Reinstalled missing Python packages and reran targeted tests.
- Installed runtime dependencies (networkx, fakeredis, lmdb, numpy, etc.) to run targeted tests and verified they pass.
- Installed dependencies for pytest and ensured hot-reload tests pass.
- Installed minimal packages (requests, pydantic, rich) and executed targeted tests successfully.
- Reinstalled dependencies (networkx, python-dateutil, watchdog, PyYAML) and ran targeted tests with ``PYTHONPATH=.`` – all pass.
