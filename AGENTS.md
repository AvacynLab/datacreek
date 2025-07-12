#### ✅ **MASTER CHECKLIST** — Branche `0.0.1` à porter au niveau du mémo d’architecture

*(aucun **placeholder** ne sera accepté, chaque point doit être livré en code exécutable et testé)*

---

### 1 │ `datacreek/core/ingest.py`

* [x] **Externaliser les constantes**

  1. Lire `ingest.chunk_size`, `ingest.chunk_overlap`, `ingest.ocr_lang` de `configs/default.yaml`.
  2. Supprimer toute valeur chiffrée en dur.
* [x] **OCR de secours PDF scannés**

  * Implémenter : `pytesseract.image_to_string(page, lang=cfg.ingest.ocr_lang)`.
  * Injecter texte OCR dans la liste des éléments `unstructured`.
* [x] **Prometheus — métriques d’ingestion**

  * Gauges : `atoms_total`, `avg_chunk_len`.

---

### 2 │ `datacreek/core/knowledge_graph.py`

* [x] **Seuils dynamiques**

  * Remplacer `TAU / SIGMA / K_MIN / LP_SIGMA / HUB_DEG` par lecture YAML.
* [x] **Projection complète Neo4j GDS**

  ```python
  relationshipProjection = {
      "INSIDE": {...},
      "NEXT":   {...},
      "HYPER":  {"type": "HYPER", "orientation": "UNDIRECTED", "aggregation": "MAX"},
  }
  ```
* [x] **Ecriture Hyper-Adamic–Adar**

  ```cypher
  CALL gds.alpha.hypergraph.linkprediction.adamicAdar.write(
        $graph, {writeRelationshipType:'SUGGESTED_HYPER_AA',
                 topK:$cfg.cleanup.lp_topk,
                 writeProperty:'score'})
  ```
* [x] **Filtre “attention adaptatif”**

  * Supprimer l’arête si `T < τ ∧ attention < median(attention)`.

---

### 3 │ `datacreek/core/fractal.py`

* [x] **Bootstrap de la dimension fractale**

  1. `for i in range(30):` → `gds.beta.graph.sample` (80 % des nœuds).
  2. `dB_i = colour_box.cover(sample)`
  3. $$
       \bar d_B=\frac1{30}\sum dB_i,\;
       σ_{d_B}=\sqrt{\frac1{29}\sum(dB_i-\bar d_B)^2}
     $$
  4. Écrire deux propriétés Neo4j : `fractal_dim`, `fractal_sigma`.

---

### 4 │ `datacreek/core/dataset.py`

#### 4-A Node2VecRunner

* [x] Lire `p,q,d` depuis la config.
* [x] Exporter `var_norm = np.var(np.linalg.norm(embs, axis=1))` → Prometheus.

#### 4-B GraphWaveRunner

* [x] **Série de Chebyshev (m = 7)**

  ```python
  L̃ = 2*L / lmax - sp.eye(N)
  Tk_0, Tk_1 = I, L̃
  psi = a0*Tk_0 + a1*Tk_1
  for k in range(2, 8):
      Tk_2 = 2*L̃.dot(Tk_1) - Tk_0
      psi += ak[k]*Tk_2
      Tk_0, Tk_1 = Tk_1, Tk_2
  ```
* [x] Calculer l’entropie

  $$
    H_{\text{wave}}=-\frac1N\sum_u \log\lVert\psi_u\rVert_2
  $$

#### 4-C PoincaréRunner

* [x] Re-centrage si `radius_mean > 0.9`

  $$
     x_i\leftarrow 0.8\,x_i/\bigl(\lVert x_i\rVert+10^{-8}\bigr)
  $$

---

### 5 │ `datacreek/analysis/hypergraph.py`

* [x] **Test interne** : hyper-AA d’un hyper-triangle doit valoir $1/\log 2$.

---

### 6 │ `datacreek/analysis/sheaf.py`

* [x] **Réduction Block-Smith**

  * Découpe `δ` en blocs de 40 000 colonnes.
  * `smith_normal_form` de `sympy` → rang partiel.
  * Recombine via suite de Mayer-Vietoris.
* [x] **Raccourci spectral**

  * Si `eigsh(Δ, k=5)[1] > cfg.sheaf.lam_thresh` ⇒ abandonner Smith.

---

### 7 │ `datacreek/analysis/fractal.py`

* [x] **GraphRNN_Lite**

  1. Charger checkpoint (`tpl.rnn_size`).
  2. Générer sous-graphe ; tag `:RNN_PATCH`.
* [x] **Validation faisceau après injection**

  * `S = 1/(1+‖b-Δx‖₂)` ; rollback si `<0.8`.

---

### 8 │ `datacreek/analysis/multiview.py`

* [x] **Sauvegarder CCA** (`cache/cca.pkl`) pour l’inférence.

---

### 9 │ `datacreek/analysis/autotune.py`

* [x] **Intégrer pénalités**

  ```python
  J += cfg.penalty.lambda_sigma * max(0, sigma_db - 0.02)
  J += cfg.penalty.lambda_cov   * max(0, C_min - coverage_frac)
  J += cfg.penalty.w_rec        * max(0, 0.9 - recall10)
  ```
* [x] **Early-stop** : ΔJ < 1e-3 sur 5 itérations ⇒ jitter ↑ et restart.

---

### 10 │ `datacreek/analysis/index.py`

* [x] **Fallback HNSW** si `query_latency > 0.1 s`

  ```python
  hnsw = faiss.IndexHNSWFlat(128, 32)
  hnsw.hnsw.efSearch = 200
  ```
* [x] Export `recall10` gauge.

---

### 11 │ `datacreek/analysis/generation.py`

* [x] **Solveur sheaf**

  ```python
  def sheaf_score(b, Delta):
      x, _ = cg(Delta, b, tol=1e-5)
      return 1/(1+np.linalg.norm(b-Delta@x))
  ```
* [x] **Ré-pondération biais**

  * Wasserstein démographie locale<>globale ; re-scale logits si > 0.1.

---

### 12 │ `datacreek/analysis/compression.py`

* [x] **Prune FractalNet**

  1. Zero-out poids < `cfg.compression.magnitude`.
  2. Fine-tune 1 epoch ; vérifier Δ perplexité < 1 %.

---

### 13 │ `datacreek/analysis/mapper.py`

* [x] **Cache tiers**

  * Redis L1 (TTL 1 h) ; LMDB L2 ; reconstruction SSD L3.

---

### 14 │ `datacreek/core/scripts/build_dataset.py`

* [x] **Ré-ordonner** : `TPL` doit précéder `embeddings`.
* [x] **CLI** : `--config`, `--source`, `--output`, `--dry-run`.

---

### 15 │ `datacreek/analysis/monitoring.py`

* [x] Ajouter : `sigma_db`, `sheaf_score`, `gw_entropy`, `tpl_w1`, `autotune_cost`.

---

### 16 │ `configs/default.yaml`

* [x] Vérifier que **toutes** les clés ci-dessus y figurent et sont documentées.

---

#### Mathématiques de référence

| Symbole            | Expression                   | Où l’utiliser      |
| ------------------ | ---------------------------- | ------------------ |
| $σ_{d_B}$          | cf. § 3                      | Autotuner pénalité |
| $W_1$              | Sinkhorn                     | TPL & monitoring   |
| $S_{\text{sheaf}}$ | $1/(1+‖b-Δx‖_2)$             | Generation & TPL   |
| $H_{\text{wave}}$  | $-\frac1N\sum\log\|\psi_u\|$ | Monitoring         |
| $J(θ)$             | voir Autotune                | SVGP-EI            |

---

### Historique
- Initialisation du fichier et import de la checklist.
- Ajout de l'option --dry-run au script build_dataset et test associé.
- Implémentation de la projection complète dans knowledge_graph et mise à jour de la checklist.
- Installé FastAPI et tenté d'exécuter tous les tests (erreurs d'import).

- Installed testing dependencies and ran individual CLI test (passed).
- Ran full test suite: many import errors remain.
- Verified checklist items and installed missing dependencies; CLI test passes after setup.
