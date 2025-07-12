# AGENTS memo

## Mémoire d’architecture opérationnelle

Voici le **mémoire d’architecture opérationnelle** que tu dois suivre pas-à-pas ; chaque sous-section décrit (1) le **but fonctionnel** de l’étape, (2) l’**algorithme exact** à appliquer, (3) les **formules** ou variables critiques, (4) l’**ordre d’exécution** et les pré-/post-conditions. Implémente-le strictement dans cet ordre ; toute dérive brisera la cohérence globale (faisceau + autotuning).

---

## 0. Pré-requis globaux

* Tous les hyper-paramètres résidents dans `configs/default.yaml` ; ils sont mutés uniquement par l’**autotuner**.
* Les logs (niveau DEBUG) doivent émettre : `σ_dB`, `coverage_frac`, `H_wave`, `sheaf_score`, `recall@10` à chacun des cycles de pipeline.

---

## 1. Ingestion multimodale → atomes

1. **Détection de modalité** (`detect_modality(file)` → `TEXT|IMAGE|AUDIO|CODE`).
2. **Partition** :

   * `unstructured.partition_*` → liste d’éléments structurés.
   * Si PDF scanné : OCR `pytesseract`.
3. **Enrichissement** :

   * `Whisper` → transcription (*AUDIO*).
   * `BLIP` → `alt_text` (*IMAGE*).
4. **Atomisation** : chaque élément = atome `(d, m)`, où

   ```
   m = {
        id, lang, media, timestamp, source_id,
        chunk_size: cfg.ingest.chunk_size,
        chunk_overlap: cfg.ingest.chunk_overlap
       }
   ```
5. **Hiérarchie molécule** : concaténer atomes successifs pour former « paragraphe / bloc » + liens `PART_OF`, `NEXT`.

---

## 2. Construction hyper-graphe + faisceau

### 2 A – Hyper-arêtes

* `add_hyperedge(nodes: list[int], relation_type, attention)`
* **Pruning HEAD-Drop** : on garde les têtes d’attention où `mean(weight) > 0.1`; les autres ont `drop_prob=0.3`.

### 2 B – Faisceau minimal

* **Fibres** : `F(v)=ℝ^{d_local}` initialisés par le vecteur CLIP/TF-IDF.
* **Restrictions** : `ρ_e(x)=W_ex` (matrice trainable).
* **Laplacien sheaf** : `Δ=δᵀδ`.

> **Ordre** : hyper-arêtes d’abord, faisceau ensuite pour créer les matrices d’incidence δ.

---

## 3. Nettoyage Neo4j GDS

1. **WCC** : purge composantes < `k=cfg.cleanup.k`.
2. **TriangleCount(τ)** : arêtes où `T < τ` ET `attention < median(attention)` → suppression.
3. **nodeSimilarity(σ)** : fusion `(u,v)` si `Jac(u,v) ≥ σ ∨ cos_sim ≥ σ`.
4. **LinkPrediction** :

   * Adamic–Adar (binaire) + Hyper-Adamic–Adar (multi-nœuds).
   * Seuil `s > cfg.cleanup.lp_sigma` → arête `:SUGGESTED_HYPER_AA`.
5. **Hubs** : si `degree > cfg.cleanup.hub_deg` → tag `hub:true`.

---

## 4. Topological Perception Layer (pré-embeddings)

1. **Persistance** : diagramme `D(G)` (GUDHI clique complex).
2. **Wasserstein-1** (Sinkhorn GPU) :

   $$
     W_1 = \min_\gamma \sum_{i,j} γ_{ij}\,‖x_i - y_j‖_\infty
   $$
3. Si `W_1 > ε = cfg.tpl.eps_w1` →

   * générer sous-graphe avec **GraphRNN** (Torch-Geometric-Temporal).
   * injecter et **revalider faisceau** : résoudre `Δx=b`, score

     $$
       S_{\mathrm{sheaf}} = \frac1{1+‖b-Δx‖_2}
     $$

     accepter si `S ≥ 0.8`, sinon rollback sous-graphe.

---

## 5. Estimation fractale

1. **COLOUR-box GPU** → $d_B$.
2. **Bootstrap** 30 sous-graphes → σ\_{d_B}.
3. Penalty douce ajoutée :

   $$
     J \;{+}= \;λ_σ\;\bigl(σ_{d_B}-0.02\bigr)_+
   $$

---

## 6. Embeddings triples corrélés

### 6 A – Node2Vec (euclidien)

* Params $(p,q,d)$ lus dans config (initial 1, 2, 128).

### 6 B – GraphWave Chebyshev

$$
  e^{-tL}\!≈\!\sum_{k=0}^{7} a_k T_k(\tilde L), \qquad
  ψ_u = \left[e^{-tL}\right]_{u,:}
$$

* Entropie $H_{\text{wave}}$ stockée.

### 6 C – Poincaré (hyperbolique)

* SGD Riemannien sur boule $ℬ^{50}$.
* Surveillance crowding : rayon moyen < 0.9.

### 6 D – Fusion produit + A-CCA + contrastif

1. **Produit** : $ z_u=(x_H,x_E)$; loss `L_prod`.
2. **A-CCA** : apprend $W_1,W_2$ (rang 32).
3. **InfoNCE tri-vue** (τₙₜc = 0.07).

---

## 7. Indexation & recherche

1. **FAISS IP** sur `n2v` (L2 normalisé).

   * Si latence > 0.1 s → fallback `IndexHNSWFlat(128, 32); efSearch=200`.
2. **Score hybride** :

   $$
     S = γ\cos_{N2V} + η (1 - d_{\mathbb B})+(1-γ-η)\cos_{GW}
   $$
3. Paramètres $(γ,η)$ ajustés par autotuner (objectif recall\@10 ≥ 0.9).

---

## 8. Mesures informationnelles

* **Entropie** : $ H=-\sum p_c\log p_c$.
* **MDL incrémental** : motif cache LRU $ΔL$.
* **Information-Bottleneck** :

  $$
    \mathcal L_{\text{IB}} = \mathbb E[\log p(y|z)] - β I(X;Z)
  $$

---

## 9. Autotuning SVGP-EI

Variables :

$$
  θ=(τ,β,ε,δ,p,q,d,α,γ,η)
$$

Objectif :

$$
  J = w_1[-H] + w_2 W_1 + w_3 βI + w_4\!Δ\text{MDL}
      + w_5[-\operatorname{Var}‖Φ_{N2V}‖]
      + λ_σ(σ_{d_B}-0.02)_+
      + λ_C(C_{\min}-C_{\text{frac}})_+
      + w_{\text{rec}}(0.9-\text{recall@10})_+
$$

* BO **SVGP-EI** (m=100) propose $θ'$.
* Soft-update α=0.3.
* Gradients Kiefer-Wolfowitz sur τ, ε.
* Early-stop si ΔJ < 0.001 cinq cycles → jitter ↗.

---

## 10. Génération LLM

1. `PromptBuilder` → `(instruction,input)` (+ Toolformer demo calls).
2. **LLM generate** (temperature 0.7).
3. **Sheaf checker batched 100** : calcul $S_{\text{sheaf}}$, rejette si < 0.8.
4. Bias repoid : Wasserstein entre démographie locale et globale.

---

## 11. Compression & cache

* **FractalNetPruner** : prune poids mag < λ (λ=0.03). Re-évalue perplexité (Δ < 1 %).
* **Mapper inverse** : nerf stocké dans SQLite ; reconstruction si cache L1 (Redis) miss, sinon L2 (LMDB) hit.

---

## 12. Privacy & rollback

* **k-out randomized response** (k=5) → ε_DP ≈ 2.
* **Git + gremlin-diff** ; MTTR SLA 2 h ; alerte Prometheus si fails > 3.

---

## 13. Monitoring (Prometheus)

Exporter : `sigma_db`, `gw_entropy`, `sheaf_score`, `recall10`, `tpl_w1`, `j_cost`.
Alertes : latence > 0.1 s (switch HNSW), $\sigma_{d_B}$ > 0.02, recall < 0.9.

---

### Ordre opératoire final

```
1  ingest           → atomes/molécules
2  build_graph      → hyper/arêtes + faisceau
3  cleanup          → WCC, triangles, similarity, LP
4  tpl_validate     → persistance & Wasserstein (fix si besoin)
5  fractal_dim      → d_B + σ
6  embeddings       → N2V, GraphWave(Chebyshev), Poincaré
7  fusion_multiview → product + A-CCA + InfoNCE
8  index_ann        → FAISS/HNSW build
9  autotune         → update θ
10 generate_llm     → prompts & validation
11 compress_cache   → FractalNet + Mapper cache
12 export_dataset   → ChatML / Alpaca + meta
```

Suivre scrupuleusement ce pipeline garantit :

* **Cohérence logique** (faisceau + Sheaf checker)
* **Signal maximal / bruit minimal** (H, IB, MDL)
* **Recherche hybride rapide** (Cypher + cosinus)
* **Scalabilité GPU** (Chebyshev, block-Smith, COLOUR-box, HNSW)
* **Traçabilité & RGPD** (k-out, rollback, logs).

### LISTE DE TÂCHES À COCHER — MISE À NIVEAU COMPLÈTE DE LA BRANCHE `0.0.1`

*(Chaque puce de premier niveau correspond à un **fichier** ; les puces imbriquées sont les **points d’action**. Les formules rappellent les variables attendues.)*

---

#### 1. **`datacreek/core/ingest.py`**

* [x] **Ajouter fallback OCR** pour PDF scannés

  * utiliser `pytesseract` ; injecter texte OCR dans la partition.
* [x] **Externaliser les hyper-paramètres**

  * `CHUNK_SIZE`, `OVERLAP` → `configs/default.yaml` (`ingest.chunk_size`, `ingest.overlap`).
  * mettre en cache `ingest.chunk_overlap` dans métadonnées.

---

#### 2. **`datacreek/core/knowledge_graph.py`**

* [ ] **Adresse Hyper-AA (linkprediction)**

  * Intégrer :

    ```cypher
    CALL gds.alpha.hypergraph.linkprediction.adamicAdar.write(..., writeRelationshipType:'SUGGESTED_HYPER_AA')
    ```
* [ ] **Compléter `relationshipProjection` pour hyper-arêtes**

  * inclure `{"HYPER":{type:'HYPER', orientation:'UNDIRECTED', aggregation:'MAX'}}`.
* [ ] **Passer seuils figés → variables dynamiques**

  * Lire `cleanup.tau`, `cleanup.sigma`, `cleanup.k` depuis config — mis à jour par autotuner.

---

#### 3. **`datacreek/core/cleanup.py`**

* [ ] **Pondération TriangleCount**

  * remplacer

    ```cypher
    CALL gds.triangleCount.write(...)
    ```

    par la version pondérée : `gds.triangleCount.stream(..., relationshipWeightProperty:'attention')`.
* [ ] **Exporter `triangles_removed` métrique** au logger.

---

#### 4. **`datacreek/core/fractal.py`**

* [ ] **Implémenter bootstrap σ\_{d_B}**

  1. Tirer 30 sous-graphes : `gds.beta.graph.sample` (0.8 ratio, rnd).
  2. Recueillir 30 estimations $d_B^i$.
  3. Stocker :

     $$
       σ_{d_B}=\sqrt{\frac1{29}\sum_i\bigl(d_B^i-\bar d_B\bigr)^2}
     $$
  4. Logger et exporter dans Neo4j graph property `fractal_sigma`.

---

#### 5. **`datacreek/core/dataset.py`**

##### 5.1 Node2VecRunner

* [ ] **Rendre $p,q,d$ variables autotune-ables**

  * récupérer `embeddings.node2vec.{p,q,dimension}` depuis config.

##### 5.2 GraphWaveRunner

* [ ] **Remplacer exponentielle dense**

  * ajouter fonction `chebyshev_heat_kernel(L, m=7)` calculant

    $$
    e^{-tL}\approx\sum_{k=0}^{m} a_k T_k(\tilde L)
    $$

    où $T_k$ polynôme de Chebyshev.
* [ ] **Exporter entropie différentielle**

  $$
  H_{\text{wave}}=-\tfrac1N\sum_{u}\log\|\psi_u\|_2
  $$

  * logger et écrire dans Neo4j property `gw_entropy`.

##### 5.3 PoincaréRunner

* [ ] **Détecter crowding**

  * moyenne rayon $r_{\text{mean}}=\frac1N\sum \|\mathbf x_H\|_{\mathbb B}$.
  * si $r_{\text{mean}}>0.9$ → message (autotuner devra réduire learning rate ou re-centre).

---

#### 6. **`datacreek/analysis/hypergraph.py`**

* [ ] **Valider Hyper-AA fonction**

  * Unit-test interne : tri-graphe d’exemple doit renvoyer $s_{\text{HAA}}(u,v)=\frac12$.

---

#### 7. **`datacreek/analysis/sheaf.py`**

* [ ] **Ajouter `block_smith(laplacian, block_size=40000)`**

  1. Découpe $\delta$ en blocs de colonnes.
  2. Appelle `sympy.matrices.normalforms.smith_normal_form` par bloc.
  3. Combine invariants (suite Mayer-Vietoris).
* [ ] **Borne spectrale**

  * Si $\lambda_k^\mathcal F$ (via Lanczos `scipy.sparse.linalg.eigsh`) > τ, court-circuiter.

---

#### 8. **`datacreek/analysis/fractal.py`**

* [ ] **Brancher GraphRNN injection**

  * importer `torch_geometric_temporal.nn.models.GraphRNN` ; générer motif → `graph.inject_subgraph`.
* [ ] **Appeler Sheaf-checker après injection**

  * `validate_section(graph, injection_nodes)`.

---

#### 9. **`datacreek/analysis/multiview.py`**

* [ ] **Stocker matrices CCA**

  * `cca_align` doit écrire `cca_Wn2v`, `cca_Wgw` sur disque (pickle) pour réutilisation inference.

---

#### 10. **`datacreek/analysis/autotune.py`**

* [ ] **Ajouter pénalités douces**

  ```python
  J += w_sigma * max(0, sigma_db - 0.02) + w_cov * max(0, C_min - coverage)
  ```
* [ ] **Early-stopping & restart**

  * si ΔJ < 0.001 cinq itérations → augmente jitter, redémarre GP.
* [ ] **Intégrer recall\@10**

  * métrique de validation ; ajoute coût $w_{\text{rec}}(0.9-\text{recall})_{+}$.

---

#### 11. **`datacreek/analysis/generation.py`**

* [ ] **Implémenter Sheaf consistency réelle**

  * résoudre $\Delta_\mathcal F x=b$ (b = contraintes sortie LLM) ; score = $1/(1+\|b-\Delta x\|_2)$.
* [ ] **Bias re-weighting**

  * calcul Wasserstein $W(\text{neighbors\_demog},\text{global})$ ; si >0.1 → upweight minor samples.

---

#### 12. **`datacreek/analysis/compression.py`**

* [ ] **Implémenter `FractalNetPruner.prune`**

  1. Charger checkpoints `facebookresearch/fractalnet`.
  2. Itération sur `model.named_parameters()` ; si magnitude < λ prune.
  3. Ré-évaluer perplexité → accepte si Δ < 1 %.

---

#### 13. **`datacreek/analysis/index.py`**

* [ ] **HNSW fallback**

  ```python
  if latency > 0.1:
      hnsw = faiss.IndexHNSWFlat(128, 32); hnsw.add(xb); hnsw.hnsw.efSearch = 200
  ```
* [ ] **Expose recall\@10** property => autotuner.

---

#### 14. **`datacreek/core/scripts/build_dataset.py`**

* [ ] **Déplacer appel TPL avant embeddings**

  ```
  ingest -> graph -> cleanup -> TPL
          -> fractal -> embeddings -> multiview -> ...
  ```
* [ ] **Paramétrer via argparse** (input, config path, output dir).

---

#### 15. **`datacreek/analysis/mapper.py`**

* [ ] **Brancher cache hiérarchique**

  * L1 Redis (`recent_nerve`)
  * L2 LMDB (`hot_subgraph`)
  * L3 reconstruction on-the-fly (SSD).

---

#### 16. **`datacreek/analysis/monitoring.py`**

* [ ] **Ajouter métriques**

  * σ\_{d_B}, recall\@10, sheaf\_score, gw\_entropy.
  * push Prometheus exporter.

---

### Variables mathématiques clés à déclarer dans `configs/default.yaml`

| Nom            | Rôle                  | Valeur initiale |
| -------------- | --------------------- | --------------- |
| `tau`          | seuil TriangleCount   | 5               |
| `sigma`        | seuil Similarity      | 0.95            |
| `eps_w1`       | tolérance Wasserstein | 0.05            |
| `p, q, d`      | Node2Vec walk         | 1.0, 2.0, 128   |
| `alpha`        | poids hyper/Eucl      | 0.6             |
| `gamma, eta`   | coefficients score    | 0.6, 0.3        |
| `beta`         | IB weight             | 1e-2            |
| `lambda_sigma` | pénalité σ            | 10              |
| `lambda_cov`   | pénalité coverage     | 5               |

---

## Fin de liste

## Historique
- Imported operational memo and task list.
- Added fallback OCR elements injection in PDFParser.parse.
- Added unit test for OCR element parsing.
- Installed minimal dependencies and ran targeted tests.

