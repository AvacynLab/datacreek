# AGENTS Checklist

### \ud83d\udccb Checklist “Version Finale” – Tout ce qu’il reste à livrer

*(structure identique à tes précédentes listes : fichier → tâches → sous-tâches / maths / variables)*

---

## 1 \u2502 `datacreek/core/knowledge_graph.py`

### 1-A  Watcher YAML (rechargement à chaud)

* [ ] **Créer le watcher**

  1. `from watchdog.observers import Observer, FileSystemEventHandler`.
  2. Classe `ConfigReloader(FileSystemEventHandler)` : si `on_modified`, appeler `load_config()` puis `apply_cleanup_config()`.
* [ ] **Fonction `apply_cleanup_config()`**

  * Mettre à jour les attributs du singleton `CleanupConfig` : `tau`, `sigma`, `k_min`, `lp_sigma`, `hub_deg`.
  * Log INFO : « [CFG-HOT] cleanup thresholds updated at {timestamp} ».
* [ ] **Thread de fond**

  * Démarrer l’`Observer` dans `__init__.py` du package.
  * Config YAML path : `cfg_path = os.environ.get("DATACREEK_CONFIG", "configs/default.yaml")`.

### 1-B  Revalidation seuils actifs

* [ ] Ajouter dans le pipeline un step « `verify_thresholds()` » qui assert :

  ```python
  assert cleanup.tau == CleanupConfig.tau
  ```

---

## 2 \u2502 `datacreek/analysis/generation.py`

### 2-A  Application effective des logits rescalés

* [ ] **Modifier la fonction appelant le LLM**

  1. Récupérer `scaled_logits` et `W`.
  2. Remplacer la clé `logits` du payload par `scaled_logits` **avant** l’appel modèle.
* [ ] **Log du facteur de biais**

  * `logger.info(f"Bias factor \u03b2={np.exp(-W):.3f} applied; W={W:.4f}")`.

### 2-B  Test unitaire « bias \u2192 logits »

* [ ] Données factices : distribution locale 90 \% A / 10 \% B, globale 50 \% / 50 \%.
* [ ] Vérifier : `\u03b2 < 1` et `scaled_logits[A] < logits[A]`.

---

## 3 \u2502 `tests/test_pipeline_e2e.py`

### 3-A  Mini-dataset & dry-run

* [ ] Créer un dossier `samples/mini/` (3 PDF, 2 images, 1 audio).
* [ ] Exécuter :

  ```bash
  python -m datacreek.build_dataset \
         --source samples/mini \
         --config configs/default.yaml \
         --output /tmp/out \
         --dry-run
  ```

### 3-B  Assertions automatisées

1. `assert sheaf_score >= 0.8`.
2. `assert recall10 >= 0.9`.
3. `if latency > 0.1: assert index.type == "HNSW"`.
4. Vérifier écriture : `fractal_sigma < 0.02`.

### 3-C  CI hook

* [ ] Ajouter à `.github/workflows/python.yml` : job « pipeline-e2e ».

---

## 4 \u2502 `datacreek/analysis/monitoring.py`

### 4-A  Gauges manquants

* [ ] Déclarer :

  ```python
  tpl_w1       = Gauge('tpl_w1',       'Wasserstein-1 TPL')
  sheaf_score  = Gauge('sheaf_score',  'Sheaf consistency score')
  gw_entropy   = Gauge('gw_entropy',   'GraphWave entropy')
  autotune_cost= Gauge('autotune_cost','Current J(theta)')
  ```
* [ ] **Push** dans :

  * `tpl.run_tpl` \u2192 `tpl_w1.set(W1)`
  * sheaf solver CG \u2192 `sheaf_score.set(score)`
  * GraphWaveRunner \u2192 `gw_entropy.set(H_wave)`
  * Autotuner update \u2192 `autotune_cost.set(J)`

---

## 5 \u2502 Docs & Config

### 5-A  YAML nouvelles clés

* [ ] Ajouter :

  ```yaml
  watcher:
    enabled: true
  ```

### 5-B  README mise à jour

* [ ] Section « Hot-reload config » expliquant la variable `DATACREEK_CONFIG`.

---

### \u{231a} Rappels mathématiques

| Variable      | Formule                                        | Contexte                                       |
| ------------- | ---------------------------------------------- | ---------------------------------------------- |
| `\u03b2`           | $\u03b2 = e^{-W}$                                   | Ré-pondération des logits après biais Sinkhorn |
| `J(\u03b8)`        | $+\; w_{\text{lat}}\max(0,\text{latency}-0.1)$ | Nouvelle pénalité latence                      |
| `sheaf_score` | $1/(1+\|b-\u0394x\|_2)$                             | Surveillance CG                                |

---

**Livrables attendus :**

* Code 100 % opérationnel (aucun `TODO`, aucun placeholder).
* Tests unitaires couvrant hot-reload, bias logits, pipeline end-to-end.
* CI GitHub Actions verte.

