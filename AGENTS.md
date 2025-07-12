Below is an **exhaustive “to-do” checklist** for every source file in the `datacreek 0.0.1` code-base (tests excluded).
Each bullet is an **actionable task** — no placeholders are allowed once you implement.
For complex topics (GraphWave Chebyshev, Sheaf solver, GraphRNN, caching…), I break work into sub-steps and restate the exact mathematics or algorithm you must respect, with variable names that must appear in code or config.

---

## 1 │ `datacreek/core/ingest.py`

* [x] **Move hard-coded parameters to config**

  * add `ingest.chunk_size`, `ingest.chunk_overlap`, `ingest.ocr_lang` in `configs/default.yaml`; read them in `DatasetBuilder`.
* [x] **Add OCR fallback**

  * use `pytesseract.image_to_string(page_img, lang=cfg.ingest.ocr_lang)` if `unstructured.partition_*` returns `[]`.
* [x] **Prometheus logging**

  * export counters `atoms_total`, `avg_chunk_len` at INFO after each file.

---

## 2 │ `datacreek/core/knowledge_graph.py`

* [x] **Dynamic thresholds**

  * read `cleanup.{tau,sigma,k_min,lp_sigma,hub_deg}`; delete literals.
* [x] **Full Neo4j GDS projection**

  * add `"HYPER": {type:'HYPER', orientation:'UNDIRECTED', aggregation:'MAX'}` to `relationshipProjection`.
* [x] **Hyper-Adamic–Adar write-back**

  * call

    ```cypher
    CALL gds.alpha.hypergraph.linkprediction.adamicAdar.write(
          $graphName,
          { writeRelationshipType:'SUGGESTED_HYPER_AA',
            writeProperty:'score', topK: cfg.cleanup.lp_topk })
    ```
* [x] **Edge-attention adaptive τ**

  * keep an edge only if `T < tau` **and** `attention < median(attention)`.

---

## 3 │ `datacreek/core/fractal.py`

* [x] **Bootstrap σ_{dB}**

  1. Sample 30 subgraphs via `gds.beta.graph.sample` (ratio 0.8).
  2. Compute each `dB_i` with `colour_box.cover`.
  3. Compute

     $$
       \overline{d_B}=\frac{1}{30}\sum dB_i,\quad
       σ_{d_B}=\sqrt{\frac{1}{29}\sum(dB_i-\overline{d_B})^{2}}
     $$
  4. Write `fractal_dim` = `overline_dB`, `fractal_sigma` = `σ_dB` on graph.

---

## 4 │ `datacreek/core/dataset.py`

### 4-A Node2VecRunner

* [x] **Link to config** — read `(p,q,d)` from `embeddings.node2vec.*`.
* [x] **Variance export**

  * compute `var_phi = np.var(np.linalg.norm(embs, axis=1))`; push metric `n2v_var_norm`.

### 4-B GraphWaveRunner

* [x] **Chebyshev approximation (7 terms)**

  ```python
  L_norm = 2*L / lmax - sp.eye(L.shape[0])
  Tk_minus, Tk = sp.eye(n), L_norm
  psi = a0*Tk_minus + a1*Tk
  for k in range(2, 8):
      Tk_plus = 2*L_norm.dot(Tk) - Tk_minus
      psi += ak[k]*Tk_plus
      Tk_minus, Tk = Tk, Tk_plus
  ```

  where `ak[k] = 2*I_k(t*lmax/2)` (modified Bessel).
* [x] **Entropy metric**

  $$
    H_{\text{wave}}=-\frac1N\sum_{u}\log\|\psi_u\|_2
  $$

  push gauge `gw_entropy`.

### 4-C PoincaréRunner

* [x] **Crowding guard**

  * if `radius_mean > 0.9`, re-embed with re-centering:

    $$
      x_i \leftarrow \frac{0.8}{‖x_i‖+10^{-8}} x_i
    $$

---

## 5 │ `datacreek/analysis/hypergraph.py`

* [x] **Solidify unit test**

  * add `assert abs(hyper_adamic_adar(u,v)-1/np.log(2))<1e-6`.

---

## 6 │ `datacreek/analysis/sheaf.py`

* [x] **Implement `block_smith(delta, block_size)`**

  * slice columns, compute Smith normal form with `sympy`, merge ranks via Mayer-Vietoris.
* [x] **Spectral shortcut**

  * if `eigsh(Delta, k=5, which='SM')` returns any `λ_k > lam_thresh`, skip Smith.

Variables:
`delta` incidence, `Delta = delta.T @ delta`.

---

## 7 │ `datacreek/analysis/fractal.py`

* [x] **GraphRNN motif injection**

  * load `GraphRNN_Lite(input_size, hidden_size)`; generate subgraph of size `tpl.rnn_size`.
  * merge nodes into Neo4j with label `RNN_PATCH`.
* [x] **Re-validate sheaf**

  * call `sheaf.validate_section(graph, nodes)`; rollback if score < 0.8.

---

## 8 │ `datacreek/analysis/multiview.py`

* [x] **Persist CCA weights**

  * `pickle.dump({'Wn2v':W1,'Wgw':W2}, 'cache/cca.pkl')`.

---

## 9 │ `datacreek/analysis/autotune.py`

* [x] **Add penalties**

  ```python
  J += cfg.penalty.lambda_sigma * max(0, sigma_db - 0.02)
  J += cfg.penalty.lambda_cov   * max(0, C_min - coverage_frac)
  J += cfg.penalty.w_rec        * max(0, 0.9 - recall10)
  ```
* [x] **Early-stop + restart**

  * if `abs(J_t−J_{t-1})<1e-3` for 5 steps → `likelihood.noise += 1e-3` and re-fit GP.

---

## 10 │ `datacreek/analysis/index.py`

* [x] **HNSW fallback**

  ```python
  if query_latency > 0.1:
      hnsw = faiss.IndexHNSWFlat(dim=128, M=32)
      hnsw.add(xb); hnsw.hnsw.efSearch=200; index = hnsw
  ```
* [x] **Prometheus recall gauge** — expose `recall10`.

---

## 11 │ `datacreek/analysis/generation.py`

* [x] **Real Sheaf consistency**

  ```python
  def sheaf_score(b, Delta):
      x, _ = cg(Delta, b, tol=1e-5, maxiter=1000)
      return 1 / (1 + np.linalg.norm(b - Delta @ x))
  ```
* [x] **Bias Wasserstein**

  * compute `W = geomloss.SamplesLoss("sinkhorn", blur=0.01)(loc_hist, glob_hist)`; if `W>0.1`, rescale logits.

---

## 12 │ `datacreek/analysis/compression.py`

* [x] **Complete `FractalNetPruner.prune`**

  1. Collect linear/conv layers.
  2. Compute global threshold `λ = cfg.compression.magnitude`.
  3. Zero weights `< λ`; fine-tune one epoch (`lr=1e-5`).
  4. Ensure `perplexity_new ≤ 1.01 * perplexity_old`.

---

## 13 │ `datacreek/analysis/mapper.py`

* [x] **Tiered cache**

  * **Redis L1** (`ttl=3600s`) → `nerve_hash`.
  * **LMDB L2** file `lmdb/hot_graph.mdb`.
  * On miss, rebuild nerve, store in both tiers.

---

## 14 │ `datacreek/core/scripts/build_dataset.py`

* [x] **Correct ordering** (TPL before embeddings).
* [x] **Argparse** — require `--config`, `--source`, `--output`.
* [x] **Stop on any sheaf failure** (`sys.exit(1)` if `sheaf_score<0.8`).

---

## 15 │ `datacreek/analysis/monitoring.py`

* [x] **Add gauges**:

  * `sigma_db`, `sheaf_score`, `gw_entropy`, `recall10`, `tpl_w1`, `autotune_cost`.
  * Export via `prometheus_client.start_http_server(cfg.monitor.port)`.

---

## 16 │ `configs/default.yaml`

Add all missing keys with sensible defaults:

```yaml
ingest:
  chunk_size: 512
  chunk_overlap: 64
  ocr_lang: "fra+eng"
cleanup:
  k: 4
  tau: 5
  sigma: 0.95
  lp_sigma: 0.3
  lp_topk: 5
  hub_deg: 500
tpl:
  eps_w1: 0.05
  rnn_size: 128
sheaf:
  lam_thresh: 1e-3
embeddings:
  node2vec: {p: 1.0, q: 2.0, d: 128}
  product_alpha: 0.6
score:
  gamma: 0.6
  eta: 0.3
penalty:
  lambda_sigma: 10
  lambda_cov: 5
  w_rec: 5
compression:
  magnitude: 0.03
monitor:
  port: 8000
```

---

### NO PLACEHOLDERS

Implement every bullet exactly; stubs or TODOs are unacceptable.
Check off each item in your PR description so the reviewer sees full coverage.

Voici, étape par étape et dans l’ordre **strict** d’exécution, la logique complète du pipeline ; je m’adresse directement à toi, développeur·se, pour que tu implémentes ou relises chaque brique exactement comme décrite. Les dépendances entre modules sont indiquées : **chaque étape produit les artefacts requis par la suivante**. J’inclus les formules essentielles et les variables de configuration que tu dois exposer dans `configs/default.yaml`.

---

## 1. Ingestion multimodale → atomes

### 1.1 Détection & partition

1. `detect_modality(path) → TEXT|IMAGE|AUDIO|CODE`.
2. `unstructured.partition_*` coupe le contenu en blocs.
3. *Fallback* : OCR `tesserocr` si un PDF scanné n’a renvoyé aucun texte.

### 1.2 Enrichissement sémantique

* **AUDIO** : `Whisper` → `transcript`.
* **IMAGE** : `BLIP` → `alt_text`.

### 1.3 Atomisation

Chaque bloc devient un atome `A = (d, m)` avec

```yaml
m:
  id: uuid4
  media: TEXT|IMAGE|AUDIO|CODE
  lang: ISO-639
  ts: ISO-8601
  chunk_size: cfg.ingest.chunk_size
  chunk_overlap: cfg.ingest.chunk_overlap
```

### 1.4 Hiérarchie

* Crée un nœud « molécule » pour chaque paragraphe / bloc code.
* Arêtes `(:ATOM)-[:INSIDE]->(:MOLECULE)` et `(:ATOM)-[:NEXT]->(:ATOM)` pour la séquence.

**Sortie →** liste d’atomes + liens INSIDE/NEXT.

---

## 2. Construction du graphe hyper-faisceau

### 2.1 Hyper-arêtes (relations ≥3 nœuds)

* Encode via **Hyper-SAGNN** ; chaque hyper-arête contient un vecteur `attention`.
* **Pruning HEAD-Drop** : désactive les têtes où `mean(attention_head)<0.1` ou `drop_prob = 0.3`.

### 2.2 Faisceau minimal

* Fibres : $F(v)=\mathbb R^{d_\text{local}}$.
* Restrictions : $ρ_e(x)=W_e x$.
* Laplacien : $Δ = δ^{\!\top} δ$.

### 2.3 Cohomologie scalable

1. Découpe `δ` en blocs 40 k colonnes → **block-Smith**; calcule rang $H^1$.
2. Shortcut : si la 5ᵉ plus petite VP $\lambda_5^{\mathcal F} > \text{cfg.sheaf.lam_thresh}$, saute Smith.

**Sortie →** graphe Neo4j + matrices `δ`, `Δ`, rang $H^1$.

---

## 3. Nettoyage GDS (ordre inviolable)

| # | Algo               | Condition de purge / fusion                            | Param. cfg         |
| - | ------------------ | ------------------------------------------------------ | ------------------ |
| 1 | **WCC**            | composante < `k_min`                                   | `cleanup.k`        |
| 2 | **TriangleCount**  | arête supprimée si $T < τ$ **et** `attention < median` | `cleanup.tau`      |
| 3 | **nodeSimilarity** | fusion si $J(u,v) ≥ σ$ ou $cos ≥ σ$                    | `cleanup.sigma`    |
| 4 | **LinkPred**       | écrire `:SUGGESTED_HYPER_AA` si score > `lp_sigma`     | `cleanup.lp_sigma` |
| 5 | **Hub tagging**    | `degree > hub_deg` → `hub:true`                        | `cleanup.hub_deg`  |

**Sortie →** graphe nettoyé, propriétés `hub`, `SUGGESTED_HYPER_AA`.

---

## 4. Topological Perception Layer

1. **Persistance** : diagramme $D(G)$ (clique complex).
2. **Wasserstein-1** :

   $$
     W_1=\min_\gamma\sum_{i,j}\gamma_{ij}\lVert x_i-y_j\rVert_\infty.
   $$
3. Si $W_1>\varepsilon=\text{tpl.eps_w1}$ :

   * Génère sous-graphe (|V| = `tpl.rnn_size`) via **GraphRNN_Lite**.
   * Injecte, puis résout $Δx=b$ (CG) ; score

     $$
      S_{\text{sheaf}}=\frac1{1+\lVert b-Δx\rVert_2}.
     $$

     Annule injection si $S<0.8$.

**Sortie →** graphe topologiquement cohérent.

---

## 5. Fractalisation

* **COLOUR-box GPU** → dimension $\bar d_B$.
* **Bootstrap** 30 échantillons → $σ_{d_B}$.
* Stocke : `fractal_dim = bar_dB`, `fractal_sigma = σ_dB`.

---

## 6. Embeddings triples corrélés

### 6.1 Node2Vec (local euclidien)

$$
\max_\Phi\sum_{u}\sum_{v\in\mathcal N(u)}\log\frac{e^{\Phi_u^\top\Phi_v}}{\sum_w e^{\Phi_u^\top\Phi_w}}
$$

Variables : `p,q,d` (config `embeddings.node2vec.*`).

### 6.2 GraphWave (rôle spectral)

$$
  e^{-tL}\approx\sum_{k=0}^{7}a_k T_k(\tilde L),\quad
  ψ_u = [e^{-tL}]_{u,:}
$$

Entropie : $H_{\text{wave}}=-\tfrac1N\sum_u\log\lVertψ_u\rVert_2$.

### 6.3 Poincaré (hiérarchique)

Distance : $d_{\mathbb B}(x,y)=\operatorname{arcosh}\bigl(1+2\frac{\lVert x-y\rVert^2}{(1-\lVert x\rVert^2)(1-\lVert y\rVert^2)}\bigr)$.
Crowding : re-centrer si rayon moyen > 0.9.

### 6.4 Fusion multi-vue

1. **Produit** $\mathbf z=(x_H,x_E)$; perte
   $\mathcal L_{\text{prod}}=\alpha d_{\mathbb B}+(1-\alpha)(1-\cos)$.
2. **A-CCA** : $W_{n2v},W_{gw}$ (rank 32).
3. **InfoNCE tri-vue** ($\tau=0.07$).

---

## 7. Index ANN + recherche hybride

1. **FAISS IP** sur `n2v`; latence > 100 ms → HNSW (M = 32, efSearch = 200).
2. Score

   $$
    S = γ\cos_{N2V} + η (1-d_{\mathbb B}) + (1-γ-η)\cos_{GW}.
   $$
3. Stocke `recall10`; rebuild si < 0.9.

---

## 8. Métriques informationnelles

* **Entropie** $H(G)$.
* **MDL incrémental** — motif-cover LRU.
* **Information-Bottleneck**
  $\mathcal L_{\text{IB}}=\mathbb E\log p(y|z)-βI(X;Z)$.

---

## 9. Autotuner SVGP-EI

$$
 \!J(θ)=w_1[-H]+w_2W_1+w_3βI+w_4Δ\text{MDL}
        +w_5[-\operatorname{Var}‖Φ_{N2V}‖]
        +λ_σ(σ_{d_B}-0.02)_+
        +λ_C(C_{\min}-C_{\text{frac}})_+
        +w_{\text{rec}}(0.9-\text{recall})_+.
$$

* SVGP (m = 100) → proposition $θ'$.
* Soft-update α = 0.3.
* Gradients KW sur τ, ε.
* Early-stop : ΔJ < 1e-3 (5×) → jitter ↑ & restart.

---

## 10. Génération LLM

1. Sélection Cypher + ANN → chemin de justification ≤ 3 sauts.
2. `PromptBuilder` produit ChatML ; Self-Instruct + Toolformer.
3. **Sheaf consistency** : solveur Δx=b, rejette si $S<0.8$.
4. **Bias Wasserstein** : re-pondère si $W>0.1$.

---

## 11. Compression & stockage

* **FractalNet pruning** λ = 0.03 ; fine-tune ; perplexity +1 % max.
* **Mapper inverse** : nerf dans SQLite.
* Cache L1 (Redis 1 h) → L2 (LMDB) → L3 (reconstruct SSD).

---

## 12. Privacy & rollback

* k-out randomized response (k = 5) → ε_DP ≈ 2.
* Rollback `gitpython` + `gremlinpython diff`.
* Alert Slack si correcteur faisceau déclenché > 3 fois / 24 h.

---

## 13. Monitoring

Expose : `sigma_db`, `sheaf_score`, `gw_entropy`, `recall10`, `tpl_w1`, `j_cost`.
Déclencheurs :

* σ_{d_B} > 0.02 → recalcul fractal
* recall < 0.9 → rebuild ANN
* sheaf_score < 0.8 → bloquer génération
* latence > 100 ms → basculer HNSW.

---

## Ordre opératoire final

```
1  ingest              # multimodal → atomes
2  build_graph         # hyper + faisceau
3  cleanup_graph       # WCC→Triangle→Similarity→LP→Hubs
4  tpl_fix             # persistance, Wasserstein, GraphRNN, sheaf check
5  fractal_dimension   # COLOUR-box + bootstrap
6  embeddings          # Node2Vec, GraphWave, Poincaré
7  multiview_fusion    # product, A-CCA, InfoNCE
8  ann_index           # FAISS/HNSW, recall
9  autotune            # SVGP-EI optimise θ
10 generate_dataset    # LLM + sheaf validate
11 compress_cache      # FractalNet + Mapper cache
12 export_final        # ChatML/Alpaca + meta
```

**Tu dois respecter cet ordre ;** chaque produit intermédiaire (topologie, embeddings, recall, métriques) sert directement de composant d’entrée à l’étape suivante. Toute exécution hors séquence invalidera les hypothèses de l’autotuner, de la cohérence faisceau et du contrôle topologique.

### Historique
- Added CCA weight persistence, monitoring gauges, and hyper-Adam test precision.
- Implemented fractal dimension bootstrap with Neo4j sampling, added real sheaf score and Wasserstein bias scaling, and marked index fallback metrics.
- Implemented block-Smith reduction with spectral shortcut, GraphRNN motif injection to Neo4j, tiered mapper cache keys, and completed FractalNet pruning logic.
- Marked spectral shortcut and GraphRNN re-validation as complete in AGENTS checklist.
- Verified checklist completion and executed targeted tests successfully.
- Installed missing deps for test execution and verified full suite passes.
- Patched compression to handle numpy parameters gracefully and added pydantic dep; confirmed targeted tests pass.
