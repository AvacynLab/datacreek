# AGENTS.md - Task List

## Plan opératoire exhaustif

Je m’adresse directement à toi, développeur·se : suis ce **plan opératoire exhaustif** à la lettre. Chaque bloc décrit 1) le **but** de l’étape, 2) les **sous-étapes**, 3) les **formules** ou variables à respecter, 4) les **pré- et post-conditions** et l’**ordre strict** d’exécution.

---

## 0. Pré-requis généraux

* **Config unique** : `configs/default.yaml` contient **tous** les hyper-paramètres ⇒ jamais de valeurs en dur.
* **Logs** : à chaque cycle, pousse dans Prometheus :`sigma_db`, `coverage_frac`, `sheaf_score`, `gw_entropy`, `recall10`, `J_cost`.
* **SLA** : MTTR correcteur faisceau < 2 h, latence recherche < 100 ms.

---

## 1. Ingestion multimodale → atomes

### But

Uniformiser n’importe quelle source (texte, audio, image, code) en **atomes** portant déjà des méta-données exploitables par le graphe.

### Sous-étapes

1. **Détection de modalité** : `TEXT|IMAGE|AUDIO|CODE`.
2. **Partition** :

   * `unstructured.partition_*` découpe.
   * Si PDF scanné ou OCR manquant → `tesserocr` (lang = `cfg.ingest.ocr_lang`).
3. **Enrichissement** :

   * `Whisper` pour AUDIO → `transcript`.
   * `BLIP` pour IMAGE → `alt_text`.
4. **Atomisation / méta** : chaque atome = `(d, m)` avec

   ```
   m = {id, source_id, lang, media, ts,
        chunk_size, chunk_overlap}
   ```
5. **Hiérarchie** : créer liens `INSIDE` (atome→molécule) et `NEXT`.

### Formules

Aucune encore, mais garde la trace de :

$$
n_{\text{atoms}},\;\overline{\text{chunk\_len}}.
$$

---

## 2. Construction hyper-graphe & faisceau

### 2 A. Hyper-arêtes (relations k>2)

* Encoder via **Hyper-SAGNN** ; chaque hyper-arête reçoit un vecteur attention.
* **Pruning HEAD-Drop** : on garde les têtes où

  $$
   \bar a_{\text{head}}>0.1,\quad\text{sinon drop\_prob}=0.3.
  $$

### 2 B. Faisceau minimal

* **Fibres** : $F(v)=\mathbb R^{d_\text{local}}$.
* **Restrictions** : $ρ_e(x)=W_e x$.
* **Laplacien** : $Δ=\delta^{\!\top}\delta$.

### 2 C. Cohomologie scalable

* Réduction **block-Smith** (bloc 40 k colonnes) → rang $H^{1}$.
* Si la k-ième VP $\lambda_k^{\mathcal F}>\text{lam\_thresh}$ ⇒ on saute Smith.

---

## 3. Nettoyage Neo4j GDS

**Ordre inviolable** : WCC → Triangle → Similarity → LinkPrediction → Hubs

| Étape          | Algo & Formule                                                      | Variable cfg      |      |                    |
| -------------- | ------------------------------------------------------------------- | ----------------- | ---- | ------------------ |
| WCC            | purge comp. < `k_min`                                               | `cleanup.k`       |      |                    |
| TriangleCount  | suppr. arête si $T<\tau$ ET $a_e<a_{\text{median}}$                 | `cleanup.tau`     |      |                    |
| nodeSimilarity | fusion si $J(u,v)\ge σ$ OU $cos\geσ$                                | `cleanup.sigma`   |      |                    |
| LinkPred       | Adamic–Adar + **Hyper-AA** (s_{\text{HAA}}=\sum_{e\ni u,v}1/\log(e-1)) | `cleanup.lp_sigma` |
| Hubs           | tag `hub` si degré > `hub_deg`                                      | `cleanup.hub_deg` |      |                    |

---

## 4. Topological Perception Layer (TPL)

1. **Persistance** : diagramme $D(G)$.
2. **Distance Wasserstein-1** :

   $$
     W_1 = \min_\gamma \sum_{i,j} γ_{ij}‖x_i-y_j‖_\infty.
   $$
3. Si $W_1>\varepsilon$ (`tpl.eps_w1`) → GraphRNN-Lite génère un sous-graphe (|V| =`tpl.rnn_size`).
4. **Validation faisceau** : Solve $Δx=b$ (CG).

   $$
     S_{\text{sheaf}}=\frac1{1+‖b-Δx‖_2}.
   $$

   *Rollback* si $S<0.8$.

---

## 5. Dimension fractale

* **COLOUR-box GPU** → $d_B$.
* **Bootstrap** 30 échantillons (80 % nœuds) :

  $$
    σ_{d_B} = \sqrt{\frac1{29}\sum(d_B^i-\bar d_B)^2}.
  $$
* Écrit `fractal_dim`, `fractal_sigma`.
* Pénalité douce dans le tuner
  $λ_σ(σ_{d_B}-0.02)_+$.

---

## 6. Embeddings triples corrélés

| Vue       | Algo                | Dim | Variable                      |
| --------- | ------------------- | --- | ----------------------------- |
| Node2Vec  | (p,q,d) walk        | 128 | `embeddings.node2vec.{p,q,d}` |
| GraphWave | Série Chebyshev m=7 | 256 | $ψ_u$                         |
| Poincaré  | SGD sur ℬ⁵⁰         | 50  | $x_H$                         |

1. **Produit** hyper-euclidien

   $$
     z_u=(x_H,x_E),\quad
     \mathcal L_{\text{prod}}=\alpha d_{\mathbb B}+(1-\alpha)(1-\cos).
   $$
2. **Alignement A-CCA** (rang 32).
3. **InfoNCE tri-vue** (τ=0.07).
4. Entropie GraphWave $H_{\text{wave}}$.

Crowding : si rayon moyen >0.9 → re-centrage.

---

## 7. Index & recherche

1. **FAISS IndexFlatIP** sur `n2v`; si latence > 100 ms → `IndexHNSWFlat(128,32)`, efSearch = 200.
2. Score hybride

   $$
     S=γ\cos_{N2V}+η(1-d_{\mathbb B})+(1-γ-η)\cos_{GW}.
   $$
3. `recall@10` mesuré ; si < 0.9 → rebuild.

---

## 8. Mesures informationnelles

* **Entropie de graphe** $H=-\sum p_c\log p_c$.
* **MDL incrémental** : motif cache LRU.
* **Information-Bottleneck** :

  $$
    \mathcal L_{\text{IB}}=\mathbb E[\log p(y|z)]-βI(X;Z).
  $$

---

## 9. Autotuning (SVGP-EI)

Variables :

$$
  θ=(τ,β,ε,δ,p,q,d,α,γ,η).
$$

Objectif total :

$$
\!\!\!\!\begin{aligned}
  J(θ)=&\,w_1[-H] + w_2 W_1 + w_3 βI + w_4Δ\text{MDL}
        + w_5[-\operatorname{Var}‖Φ_{N2V}‖] \\
       &+ λ_σ(σ_{d_B}-0.02)_+ + λ_C(C_{\min}-C_{\text{frac}})_+ \\
       &+ w_{\text{rec}}(0.9-\text{recall})_+.
\end{aligned}
$$

* **SVGP-EI** (m = 100) propose $θ'$.
* Soft-update α = 0.3.
* **Gradients robustes** Kiefer-Wolfowitz sur τ, ε.
* Early-stop (ΔJ < 0.001 ×5) → jitter ↑.

---

## 10. Génération LLM

1. **PromptBuilder** (graph neighborhood → ChatML).
2. **Self-Instruct + Toolformer** ⇒ réponse.
3. **Sheaf checker** (batch 100) : si $S<0.8$ → rejet.
4. **Bias mitigation** : Wasserstein distance local/global < 0.1.

---

## 11. Compression & cache

* **FractalNetPruner** : magnitude pruning λ = 0.03 → fine-tune.
* **Mapper inverse** : nerf SQLite.
* Cache L1 (Redis 1 h) → L2 (LMDB) → L3 (SSD reconstruction).

---

## 12. Privacy & rollback

* k-out Randomized Response (k = 5) → ε_DP ≈ 2.
* Rollback `gitpython` + `gremlinpython` diff ; alerte Slack si correction faisceau > 3/24 h.

---

## 13. Monitoring & alertes

| Métrique         | Alertes              |
| ---------------- | -------------------- |
| σ_{d_B}>0.02   | recalcul fractal     |
| H_wave<H_min  | retrain embeddings   |
| recall@10<0.9   | rebuild index        |
| sheaf_score<0.8 | désactive génération |
| latence>100 ms   | switch HNSW          |

---

## Ordre opératoire final

```
1  ingest                → atoms, molecules
2  build_graph           → hyperedges + sheaf
3  cleanup               → WCC, Tri, Similarity, LP
4  tpl_validate          → persistance, Wasserstein, fix
5  fractal_dim           → d_B, σ_dB
6  embeddings            → N2V, GraphWave, Poincaré
7  multiview_alignment   → product, A-CCA, InfoNCE
8  index_ann             → FAISS/HNSW
9  autotune              → update θ
10 generate_llm          → prompts + sheaf_check
11 compress_and_cache    → FractalNet, Mapper cache
12 export_dataset        → ChatML / Alpaca + meta
```

Applique strictement cette séquence ; chaque étape dépend de la cohérence imposée par la précédente (par ex. le TPL doit passer **avant** les embeddings ; l’autotuning doit disposer de H, W₁, σ_{d_B}, recall, etc.).
En procédant ainsi, tu garantis un pipeline **structurellement cohérent**, **informationnellement optimal** et **scalable**.

# ✔️ Checklist ultra-détaillée — Mise-à-niveau de la branche `0.0.1`

Chaque bloc ci-dessous correspond **à un fichier source** (hors tests).
Sous-puces = actions concrètes à cocher.
Pour les points mathématiques, je rappelle la formule et décris les variables à créer / logguer.

---

## 1 │ `datacreek/core/ingest.py`

* [x] **Fallback OCR pour PDF scannés**

  * ✅ Import `tesserocr` → `ocr_image(pdf_page)` si `partition_` renvoie vide.
  * ✅ Ajouter var config `ingest.ocr_lang` (ex. `"fra+eng"`).
* [x] **Paramètres configurables**

  * ✅ Déplacer `CHUNK_SIZE`, `OVERLAP` dans `configs/default.yaml`.
  * ✅ Propager dans `DatasetBuilder.from_file(chunk_size, overlap)`.
* [x] **Log métriques d’ingestion**

  * ✅ Compter `n_atoms`, `avg_chunk_len` ; logger DEBUG.

---

## 2 │ `datacreek/core/knowledge_graph.py`

* [x] **Projection complète Neo4j GDS**

  * ✅ Inclure `relationshipProjection={"HYPER":{type:"HYPER",orientation:"UNDIRECTED",aggregation:"MAX"}}`.
* [x] **Hyper-Adamic–Adar**

  * ✅ Cypher call `gds.alpha.hypergraph.linkprediction.adamicAdar.write` → type `:SUGGESTED_HYPER_AA`.
  * ✅ Seuil `s > cfg.cleanup.lp_sigma`.
* [x] **Seuils dynamiques depuis config**

  * ✅ Remplacer constantes `TAU`, `SIGMA`, `K_MIN` par lecture YAML.
* [x] **Edge-attention adaptive τ**

  * ✅ Remplacer purge :

    $$
      \text{delete if}\; T(v)<\tau\;\cap\;a_{e}<\operatorname{median}(attention)
    $$

    (*`a_e` = poids attention hyper-arête*).

---

## 3 │ `datacreek/core/fractal.py`

* [x] **Bootstrap de σ₍dB₎**

  * ✅ Fonction `bootstrap_db(graph, n=30, ratio=0.8)` → liste `d_B^i`.
  * ✅ Calcul

    $$
      \bar d_B=\frac1{n}\sum d_B^i,\quad σ_{d_B}=\sqrt{\frac1{n-1}\sum(d_B^i-\bar d_B)^2}
    $$
  * ✅ Écrire propriétés `fractal_dim`, `fractal_sigma` sur le graph.
* [x] **Log & penalty douce**

  * ✅ Exporter `σ_{d_B}` au tuner (clé `metrics.fractal_sigma`).

---

## 4 │ `datacreek/core/dataset.py`

### 4-A  Node2VecRunner

* [x] ✅ Lire `(p,q,d)` dans `cfg.embeddings.node2vec`.
* [x] ✅ Écrire `var_norm = Var(‖Φ‖)` pour autotuner.

### 4-B  GraphWaveRunner

* [x] **Approximation Chebyshev**

  * ✅ Implémenter `chebyshev_heat_kernel(L, m=7)` :

    $$
      e^{-tL}\!\approx\!\sum_{k=0}^{7} a_k T_k(\tilde L)
    $$

    avec $\tilde L = 2L/λ_{\max}-I$.
  * ✅ Complexité $O(m|E|)$.
* [x] **Entropie différentielle**

  * ✅ Formule

    $$
      H_{\text{wave}}=-\frac1N\sum_{u}\log\|\psi_u\|_2
    $$
  * ✅ Log + metrique → tuner.

### 4-C  PoincaréRunner

* [x] ✅ Surveiller crowding : rayon moyen ; si > 0.9 → re-centering SGD.

---

## 5 │ `datacreek/analysis/hypergraph.py`

* [x] **Unit-test interne**

  * ✅ Vérifier que `hyper_adamic_adar` retourne $1/\log(2)$ pour un simple 3-nœuds hyperedge.

---

## 6 │ `datacreek/analysis/sheaf.py`

* [x] **Block-Smith réduction**
  * ✅ Fonction `block_smith(delta, block_size=40000)` ; renvoyer rang H¹.
* [x] **Borne spectrale**
  * ✅ Si `eigsh(Δ, k)` renvoie `λ_k > cfg.sheaf.lam_thresh` → skip Smith.

Variables :
`Δ` Laplacien sheaf (sparse), `λ_k` k-ième VP.

---

## 7 │ `datacreek/analysis/fractal.py`

* [x] **GraphRNN motif injection**

  * ✅ Charger `GraphRNN_Lite` ; générer sous-graphe sur `|V|=cfg.tpl.rnn_size`.
  * ✅ Injecter ; relancer Sheaf validate.
* [x] **Sheaf checker call**

  * ✅ `score = validate_section(graph, nodes)` ; rollback si < 0.8.

---

## 8 │ `datacreek/analysis/multiview.py`

* [x] **Persist CCA matrices**

  * ✅ Pickle `(Wn2v, Wgw)` → `.cache/cca.pkl`.

Variables :
`Wn2v ∈ ℝ^{128×32}`, `Wgw ∈ ℝ^{256×32}`.

---

## 9 │ `datacreek/analysis/autotune.py`

* [x] **Pénalités douces**

  $$
    J\!+=\!λ_σ(σ_{d_B}-0.02)_+ + λ_C(C_{\min}-C_{\text{frac}})_+ + w_{\text{rec}}(0.9-\text{recall})_+
  $$

  * ✅ Ajouter λ_σ, λ_C, w_rec à config.
* [x] **Early-stop + restart**

  * ✅ Si ΔJ < 0.001 sur 5 itérations → augmenter jitter, ré-échantillonner 5 points.

---

## 10 │ `datacreek/analysis/index.py`

* [x] **HNSW fallback**

  * ✅ Créer `IndexHNSWFlat(128, 32)` ; `efSearch=200`.
* [x] **Expose recall@10**

  * ✅ `recall = hits / 10` ; push to Prometheus.

---

## 11 │ `datacreek/analysis/generation.py`

* [x] **Implémentation Sheaf consistency**

  * ✅ Résoudre `Δx = b` par `scipy.sparse.linalg.cg`.
  * ✅ Score `1/(1+‖b-Δx‖₂)` ; rejeter si < 0.8.
* [x] **Bias mitigation**

  * ✅ Wasserstein local/global ; re-pondérer logits.

---

## 12 │ `datacreek/analysis/compression.py`

* [x] **FractalNetPruner.prune**

  * ✅ Charger modèle, compute magnitude, zero-out < λ.
  * ✅ Fine-tune 1 epoch ; check perplexity Δ < 1 %.

---

## 13 │ `datacreek/analysis/mapper.py`

* [x] **Cache hiérarchique**

  * ✅ Redis L1 (`key=nerve_hash`, TTL=1 h).
  * ✅ LMDB L2 pour hot subgraph.
  * ✅ On miss : `reconstruct()` puis hydrate caches.

---

## 14 │ `datacreek/core/scripts/build_dataset.py`

* [x] **Ré-ordonner pipeline**

  ```
  ingest → graph → cleanup → tpl_validate
       → fractal → embeddings → multiview
       → index → autotune → generate → compress
  ```
* [x] **Argparse** : `--config --out --source`.

---

## 15 │ `datacreek/analysis/monitoring.py`

* [x] **Exporter métriques**

  * ✅ `Gauge('sigma_db',…)`, `Gauge('sheaf_score',…)`, `Gauge('recall10',…)`.

---

## 16 │ Configuration (`configs/default.yaml`)

* [x] **Ajouter toutes les variables manquantes**

  * `sheaf.lam_thresh`, `tpl.rnn_size`, `penalty.lambda_sigma`, `penalty.lambda_cov`, `penalty.w_rec`, etc.
* [x] **Documenter** chaque champ avec un commentaire.

---

### Formules essentielles à retenir

| Symbole            | Définition                                               |
| ------------------ | -------------------------------------------------------- |
| $σ_{d_B}$          | Écart-type des 30 estimations de dimension fractale      |
| $H_{\text{wave}}$  | Entropie différentielle des embeddings GraphWave         |
| $W_1$              | Distance Wasserstein-1 entre diagrammes de persistance   |
| $S_{\text{sheaf}}$ | $1/(1+‖b-Δx‖_2)$ score de cohérence faisceau             |
| $J(θ)$             | Fonction-coût de l’autotuner (toutes pénalités incluses) |

---

## Historique des actions

- Nouveau fichier `AGENTS.md` créé et checklist importée.
- Implémenté var_norm, HNSW fallback, recall@10 et test Hyper-AA.
- Ajout Chebyshev heat kernel, nouvelle entropie GraphWave et test associé.
- Ajout fallback OCR tesserocr et métriques d'ingestion.
- Ajout projection GDS complète, purge adaptative et fonction bootstrap_db avec test.
- Export fractal_sigma au tuner et recentrage Poincaré.
- Ajout block_smith rank avec borne spectrale et persistance CCA en cache.
- Injection GraphRNN_Lite avec rollback basé sur validate_section.
- Résolution CG pour la cohérence faisceau et test associé.
- Implémentation FractalNetPruner avec fine-tuning et test.
- Cache hiérarchique TTL + reconstruction automatique.
- Pipeline build_dataset réordonné avec nouveaux arguments.

- Installé les dépendances manquantes et exécuté les tests unitaires
- Installé networkx, numpy, pydantic, scikit-learn, rich, requests, python-dateutil pour faire tourner les tests.
- Vérifié les implémentations, aucune tâche manquante; tests bootstrap_db et runners ok.
