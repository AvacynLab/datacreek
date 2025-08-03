### Check-list complète « v2.1 – Hardening Qualité / Perf / Gouvernance »

*(chaque case ☐ = action à cocher ; indentation = sous-étape ; formules + objectifs + DoD inclus)*

---

## 1 – Filtre sémantique & sûreté (⭑⭑⭑)

* [x] **Module `ingest/safety_filter.py`**

  * [x] ☐ Charger modèle tiny HF « toxicity » (`distilroberta-toxic`).
  * [x] ☐ Couper contenu NSFW : regex + score > 0.7.
  * [x] ☐ Log métrique `ingest_toxic_blocks_total`.
* **Maths** : score global

  $$
    s = \tfrac12(s_\text{tox}+s_\text{nsfw})
  $$

  | Var             | Description              |
  | --------------- | ------------------------ |
  | $s_\text{tox}$  | prob toxicité modèle     |
  | $s_\text{nsfw}$ | proba NSFW (regex heur.) |
* **Objectif** : < 0.5 % toxic payloads passent.
* **DoD** : test dataset toxique → 95 % blocage, CI.

---

## 2 – SNR dynamique (PID soft-gating)

* [x] **Calcul écart-type SNR** :

  $$
    \sigma_{SNR} = \sqrt{\tfrac1N\sum (SNR_i-\overline{SNR})^2}
  $$
* [x] ☐ Adapter seuil : `thr = 6 + 0.5·σ_{SNR}`.
* **DoD** : taux faux positifs audio silence < 2 %.

---

## 3 – Power iteration λ_max (hypergraph Laplacien) (⭑⭑)

* [x] **`analysis/hypergraph_conv.py`**
  * [x] ☐ Ajouter `estimate_lambda_max(Δ, it=3)` : $v_{k+1}=Δv_k/‖Δv_k‖$.
* **Objectif** : speed gain > 10× vs eigsh.
* **DoD** : erreur λ̂/λ < 5 %, benchmark.

---

## 4 – Exporter arêtes incohérentes Sheaf/Hyper (explain)

* [x] **`analysis/sheaf_hyper_bridge.py`**
  * [x] ☐ Fonction `top_k_incoherent(k, τ)` => edges Δλ_i > τ.
* [x] ☐ Route `/explain/sheaf_diff?top=50`.
* **DoD** : JSON liste edges, latence < 100 ms pour k=50.

---

## 5 – Kernel PCA sur vecteurs PI (curse of dim)

* [x] **`analysis/tda_vectorize.py`**
  * [x] `reduce_pca(X_PI, n=50)` – sklearn incremental.
* **DoD** : recall ANN chute < 0.5 % ; dim tot – 75 %.

---

## 6 – Re-évaluation dynamique dimension fractale latente

* [x] **Callback entraînement** : tous les 2 epochs → `fractal_dim_embedding`.
* [x] Loss fractale :

  $$
    \mathcal{L}_{frac} = \beta\,|\hat D_f - D_f^{target}|
  $$
* **DoD** : log `fractal_loss` ; dim se stabilise ±0.05.

---

## 7 – Overlap Mapper auto-tuning

* [x] **Grid-search** overlap ∈ {0.2,0.3,0.4,0.5}.
  * [x] Choisir overlap max silhouette score.
* **DoD** : silhouette ↑ ≥ 0.03 vs overlap fixe.

---

## 8 – LakeFS schema registry (⭑⭑)

* [x] **`.lakefs/schema.yaml`** : champs + types.
* [x] **Git hook** : refuser commit si breaking change.
* **DoD** : CI fail sur schema break, OK sinon.

---

## 9 – Snapshot tokenizer

* [x] **`utils/dataset_export.py`**
  * [x] Sauver `tokenizer.json` dans branch LakeFS (same commit).
* **DoD** : hash tokenizer stable ; reproduce exact split.

---

## 10 – Alias-aware fact-check reward (⭑⭑⭑)

* [x] **`training/reward_fn.py`**
  * [x] Mapper alias via Neo4j index `apoc.text.clean`.

  $$
    R = \frac{\#facts\ validés+\#alias\_validés}{\#facts}
  $$
* **DoD** : hallucination rate QA ↓ 30 %.

---

## 11 – Entailment check paraphrases (active learning)

* [x] **`augmenter.py`**
  * [x] Passer paraphrase + phrase org → model `facebook/bart-large-mnli`, drop si `contradiction`.
* **DoD** : drift label < 1 %.

---

## 12 – Metric `training_eta_seconds`

* [x] **Callback** : ETA = `(steps_total - step_done)/steps_per_sec`.
* **DoD** : Grafana montre barre progression.

---

## 13 – Checkpoint pruning (disk)

* [x] **Keep top-k (k=2) checkpoints** basé `val_metric`.
* **DoD** : Espace disque run heavy ≤ +5 GB.

---

## 14 – Notebook ex : fine-tune QA

* [x] `examples/fine_tune_QA.ipynb` : de l’ingestion à inference.
* **DoD** : exécute sans erreur sur GPU Colab.

---

## 15 – Property-tests reward_fn

* [x] Hypothesis : si réponse == vérité alors R=1 ; si vide → R=0.
* **DoD** : tests verts.

---

### KPI v2.1 (post-hardening)

| KPI                        | Cible       |
| -------------------------- | ----------- |
| Toxic payloads non filtrés | < 0.5 %     |
| λ̂ calc time               | −90 %       |
| ANN recall drop après PCA  | < 0.5 %     |
| Hallucination QA           | −30 %       |
| Disk footprint checkpoints | ≤ +5 GB/run |

**Cochez toutes les cases → Datacreek v2.1-beta prêt.**

### History
- Reset AGENTS for v2.1 tasks.
- Implemented training ETA metric and callback with Prometheus logging; added unit tests.

- Added semantic safety filter with Hugging Face model, NSFW regex, Prometheus counter, and unit tests.
- Added dynamic SNR soft-gate with adaptive threshold and tests.
- Added power iteration estimator for hypergraph Laplacian λ_max and accompanying tests.
- Added sheaf/hypergraph incoherence exporter with API route `/explain/sheaf_diff` and tests.
- Added incremental PCA reducer `reduce_pca` for persistence-image vectors with unit tests.
- Added fractal dimension callback with loss logging and Prometheus gauge; added unit tests.
- Implemented checkpoint pruner keeping top-2 checkpoints by validation metric with unit tests.
- Added Mapper overlap auto-tuning via silhouette-based grid search with unit test.
- Implemented alias-aware fact-check reward with Neo4j alias resolution and
  accompanying unit/property tests.
- Added contradiction check for paraphrases with bart-large-mnli and unit test.
- Snapshot tokenizer export saving `tokenizer.json` with LakeFS commit and unit tests.
- Added LakeFS schema registry with pre-commit guard to block breaking changes; included unit tests.
- Created fine-tuning QA example notebook demonstrating ingestion, training, and inference.
- Exposed `EtaCallback` via the top-level `training` package and added test for import.
- Verified all checklist implementations via targeted unit tests; no further changes required.
