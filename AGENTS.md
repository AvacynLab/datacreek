Voici un **rapport de recroisement très minutieux** entre les idées d’amélioration proposées précédemment (TDA/hypergraphes/fractal/embeddings/efficience) et **votre code actuel** (archive `datacreek-0.0.1.zip`, 647 fichiers).
J’indique **ce qui existe déjà**, **ce qui est partiel**, **ce qui manque** et **ce qu’il serait pertinent d’ajouter** tout de suite, avec formules, objectifs et critères de recette.

---

## 1) Cartographie “idées ↔ code actuel” (vérifié dans l’archive)

| Thème                                     | Attendu (idée précédente)                                                         | Dans le code ? | Où / preuves                                                                                                                                                                 | Écart(s)                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **TDA – persistance**                     | Diagrammes de persistance, entropie de persistance, distance Wasserstein/Sinkhorn | **Oui**        | `datacreek/analysis/fractal.py` (≈ `persistence_diagrams`, `persistence_entropy`, `persistence_wasserstein_distance`) ; `datacreek/analysis/tpl.py` (approximation Sinkhorn) | **Vectorisation** (persistence images / landscapes) **absente**           |
| **Fractal – dimension/box-counting**      | Box-cover/box-counting multi-échelles + stockage                                  | **Oui**        | `datacreek/analysis/fractal.py` (`box_counting_dimension`, `box_cover`) ; `tests/integration/test_fractal_dimension.py`                                                      | —                                                                         |
| **Fractal – persistance en base**         | `fractal_dim` sur nœuds / sous-graphes + index | **Oui**        | `datacreek/core/knowledge_graph.py` → `subgraph_fractal_dimension` stocke `fractal_dim`; index full-text mentionné                                                           | Migration via Cypher ajoutée pour `fractal_dim`            |
| **Mapper / vue topologique**              | Graphe de clusters Mapper                                                         | **Oui**        | `datacreek/analysis/mapper.py` (+ tests heavy/intégration/unit)                                                                                                              | Implémentation “greedy cover” simplifiée (pas de lens/overlap paramétrés) |
| **Hypergraphes – embeddings**             | Hyper-SAGNN / CUDA streams                                                        | **Oui**        | `datacreek/analysis/hypergraph.py` (CPU) + `analysis/hyper_sagnn_cuda.py` (streams batched)                                                                                  | Pas de **convolution/Laplacien d’hypergraphe**                            |
| **Sheaves / cohomologie**                 | Outils sheaf + cohomologie bloc-Smith                                             | **Oui**        | `datacreek/analysis/sheaf.py` (laplacien, conv, cohomologie, block-Smith)                                                                                                    | —                                                                         |
| **GraphWave – Chebyshev streaming**       | Budget VRAM contrôlé (blocage m→b)                                                | **Oui**        | `datacreek/analysis/graphwave_cuda.py` (formule de blocage en docstring)                                                                                                     | —                                                                         |
| **Embeddings – Poincaré**                 | Recentering Möbius + clamps                                                       | **Oui**        | `datacreek/analysis/poincare_recentering.py`                                                                                                                                 | —                                                                         |
| **Embeddings – compression & PCA sketch** | FP8 + PCA incrémental                                                             | **Partiel/OK** | `datacreek/analysis/compression.py` (fp8\_quantize/dequantize) ; `analysis/fractal_encoder.py` (online\_pca\_reduce)                                                         | FP8 dépend de `numpy`; pas de **bench rappel vs FP16**                    |
| **ANN – Hybrid HNSW→IVFPQ**               | nprobe\_multi CPU + rerank PQ                                                     | **Oui**        | `datacreek/analysis/hybrid_ann.py` (nprobe\_multi, rerank)                                                                                                                   | —                                                                         |
| **Audio – VAD & LangID**                  | VAD “smart” + Tag langue                                                          | **Oui**        | `datacreek/utils/audio_vad.py` (webrtcvad) ; `utils/text.py` (fastText) ; tests intégration                                                                                  | —                                                                         |

**Conclusion cartographie :** la base “TPL/Fractal/Sheaf/GraphWave/Hyper-SAGNN/Mapper” **existe et est solide**. Les principaux manques par rapport aux pistes proposées sont :

1. **Vectorisation TDA** (persistence **images/landscapes**),
2. **Convolution d’hypergraphe** (Laplacien $\Delta$ d’hypergraphe),
3. **Mapper complet** (lens/overlap, pas seulement greedy cover),
4. **Perfs TDA H₀/H₁** (Union-Find, cache, incrémental),
5. **Embeddings hyperboliques “appris”** (HNN/HGCN hyperboliques) + **régularisation fractale**.

---

## 2) Ce qu’il est pertinent d’ajouter maintenant (et comment)

### 2.1 TDA – Persistence Images / Landscapes (vectorisation robuste)

**Pourquoi** : transformer les diagrammes de persistance en vecteurs stables pour enrichir les embeddings nœud/sous-graphe/graphe.

**Maths (image de persistance)**
Soit $D = {(b_i, d_i)}$ un diagramme ($b$ naissance, $d$ mort), et $p_i = d_i - b_i$ la persistance. L’image de persistance (Adams et al.) :

$$
I(x,y) = \sum_i w(p_i)\;\exp\!\left(-\frac{(x - b_i)^2 + (y - d_i)^2}{2\sigma^2}\right),
$$

avec $w(\cdot)$ un poids (p. ex. $w(p)=p$) et une grille $(x,y)$ discrète.
**Landscape** : $\lambda_k(t)$ = $k$-ième enveloppe supérieure des triangles noyautés par $(b_i,d_i)$.

**À faire (très concret)**

* [x] `analysis/tda_vectorize.py` : `persistence_image(diag, sigma, grid)` et `persistence_landscape(diag, k_max)`.
* [x] **Concat** aux embeddings existants : `Φ ← [Φ ; vec_PI(H0) ; vec_PI(H1)]`.
* [x] **Tests** : stabilité L2 à perturbation (±ε) et gain de séparation (silhouette).
* [x] `analysis/tda_vectorize.py` : `persistence_silhouette(diag, t, p)` pour une troisième vectorisation.
* [x] **Tests** : silhouette stable sous perturbation.
  **DoD** : AUC/ARI ↑ ≥ **+2 pts** sur un bench clustering de sous-graphes.

---

### 2.2 Hypergraphes – Convolution spectrale (Laplacien d’hypergraphe)

**Pourquoi** : vos hyper-edges sont traitées par attention (Hyper-SAGNN) ; ajoutez la composante spectrale pour capter des motifs d’ordre supérieur globaux.

**Maths (Laplacien normalisé)**
Avec matrice d’incidence $B\in{0,1}^{|V|\times|E|}$, degrés $D_v, D_e$, poids $W$ :

$$
\Delta = I - D_v^{-1/2}\, B\, W\, D_e^{-1}\, B^\top\, D_v^{-1/2}.
$$

**Convolution** (K-Chebyshev) :

$$
X' = \sum_{k=0}^K \theta_k\, T_k(\tilde \Delta)\,X
\quad\text{où}\quad \tilde\Delta=\frac{2}{\lambda_{\max}}\,\Delta - I.
$$

**À faire**

* [x] `analysis/hypergraph_conv.py` : `hypergraph_laplacian(B, w)` + `chebyshev_conv(X, Δ, K)`.
* [x] **Hook** dans pipeline : combiner `Hyper-SAGNN` (local) + `HGCN-spectral` (global).
* [x] **Tests** : orthogonalité empirique modes + amélioration Macro-F1 (≥ +1.5 pts).
  **DoD** : **Recall@100** sur requêtes “hyper-communautés” ↑ **+1 pt**, sans latence > 5 %.

---

### 2.3 Mapper complet (lens/overlap/cover paramétrables)

**Pourquoi** : la version actuelle couvre mais simplifie Mapper. Un Mapper configurable structure mieux la projection et donc l’explainabilité.

**À faire**

* [x] `analysis/mapper.py` : ajouter `lens: Callable`, `cover=(n_intervals, overlap)`, `clusterer='dbscan|single'`.
* [x] **Lens** par défaut : 1ʳᵉ composante PCA ou énergie GraphWave (norme $‖Φ_\text{GW}‖$).
* [x] **UI** : export JSON (nœuds=clusters, arêtes=overlap) pour `/explain`.
  **DoD** : temps de construction ~O(N log N) ; **Nombre de composantes Mapper** corrélé (ρ > 0.6) aux labels.

---

### 2.4 TDA efficiente (H₀/H₁) : Union-Find + incrémental + cache

**Pourquoi** : calculer H₀/H₁ vite est critique si l’on densifie les filtrations.

**À faire**

* [x] `analysis/tda_fast.py` : H₀ par Union-Find sur arêtes triées par seuil (quasi-linéaire amorti).
* [x] H₁ borne supérieure via spanning-forest + cycles fondamentaux (peigne sur cordes).
* [x] **Cache** par hachage de sous-graphe (frozenset edges) → persistance réutilisée.
  **DoD** : ×3 speed-up sur H₀/H₁ vs `gudhi` par défaut, erreurs < 1 %.

---

### 2.5 Embeddings hyperboliques appris + régularisation fractale

**Pourquoi** : vous avez Poincaré & recentrage ; aller plus loin en apprenant la projection + contraindre la dimension fractale latente.

**Maths (perte composite)**

* **Hyperbolique** (modèle de Lorentz) : distances $d_{\mathbb{H}}(x,y)$ ; **loss** de préservation :

$$
\mathcal{L}_\text{geo} = \sum_{(i,j)} \big(d_{\mathbb{H}}(z_i,z_j) - d_G(i,j)\big)^2.
$$

* **Régularisation fractale** (dimension box-counting latente $\hat D_f$) :

$$
\mathcal{L}_\text{frac} = \big|\hat D_f(\{z_i\}) - D_f(G)\big|.
$$

* **Total** : $\mathcal{L}= \mathcal{L}_\text{task} + \alpha \mathcal{L}_\text{geo} + \beta \mathcal{L}_\text{frac}$.

**À faire**

* [x] `analysis/hyp_embed.py` : projection apprise (MLP) + géométrie hyperbolique (Lorentz ops en NumPy/JAX/torch si dispo).
* [x] `analysis/fractal.py` : estimateur $\hat D_f$ rapide en espace latent (box-counting approximé).
  **DoD** : **NMI/ARI** clustering +2 pts ; VRAM stable (FP8 activé) ; pas de NaN (clamps).

---

### 2.6 “Sheaf → Hypergraph” pont spectral

**Pourquoi** : vous avez sheaf Laplacian & cohomologie et hypergraphes ; offrir un pont invariant (features communs).

**Maths**

* Construire $B$ (incidence hypergraphe) signé par contraintes de faisceau (sheaf) et comparer spectres $(\lambda_i)$ de $\Delta_\text{sheaf}$ et $\Delta_\text{hyper}$ (corrélation de spectres).
* **Score de cohérence** :

$$
S = 1 - \frac{\sum_i|\lambda_i^\text{sheaf} - \lambda_i^\text{hyper}|}{\sum_i \lambda_i^\text{sheaf} + \lambda_i^\text{hyper}}.
$$

**À faire**

* [x] `analysis/sheaf_hyper_bridge.py` : génère $B$ signé, calcule spectres & score $S$.
  **DoD** : $S>0.7$ sur sous-graphes denses ; expose métrique Prometheus.

---

### 2.7 Migration Cypher pour l'index fractal

**Pourquoi** : disposer d'une migration versionnée pour l'index full-text sur ``Subgraph.fractal_dim`` plutôt que d'un appel ad-hoc.

**À faire**

* [x] Créer un fichier de migration Neo4j créant l'index full-text ``idx_fractal`` sur ``Subgraph(fractal_dim)``.

---

### 2.8 Augmentation multi-vectorisation des diagrammes

**Pourquoi** : enrichir les embeddings en permettant différents résumés de
diagrammes de persistance (image, paysage, silhouette).

**À faire**

* [x] Généraliser `augment_embeddings_with_persistence` avec un paramètre
  ``method`` (`"image"`, `"landscape"`, `"silhouette"`).
* [x] Ajouter des tests heavy vérifiant la forme des embeddings augmentés pour
  chaque méthode.

---

### 2.9 Courbe de Betti

**Pourquoi** : suivre l'évolution des nombres de Betti sur une grille de
filtrations pour un résumé vectoriel simple.

**À faire**

* [x] `analysis/tda_vectorize.py` : ajouter `betti_curve(diag, t)` et prise en
  charge de ``method="betti"`` dans `augment_embeddings_with_persistence`.
* [x] Tests heavy vérifiant la courbe de Betti et l'augmentation des
  embeddings.
* [x] Exposer `betti_curve` via `datacreek.analysis` pour chargement lazy.

---

### 2.10 Entropie de persistance comme vecteur

**Pourquoi** : résumer les diagrammes par l'entropie de persistance, simple à
calculer et informative sur la répartition des durées.

**À faire**

* [x] `analysis/tda_vectorize.py` : ajouter `diagram_entropy(diag)` calculant
  l'entropie normalisée des persistances.
* [x] Ajouter ``method="entropy"`` à `augment_embeddings_with_persistence`.
* [x] Tests heavy vérifiant la valeur d'entropie et la forme des embeddings
  augmentés.

---

## 3) Petites optimisations utiles (peu de code, gros effet)

1. **Mapper lens = énergie GraphWave** : réutilise `graphwave_cuda.py`, 0 coût modèle, meilleure lisibilité. **[x]**
2. **Cache TDA** : décorateur LRU sur `persistence_diagrams` clé = hash sous-graphe. **[x]**
3. **Tests property-based** TDA/Fractal : monotonicité filtrations, invariance isomorphismes. **[x]**
4. **Bench FP8** : ajouter test rappel@K (Δ ≤ 0.005) pour sécurité qualité après quantif. **[x]**

---

## 4) Critères de succès (KPI & recette)

| Axe                    | KPI attendu                                   | Mesure / Test                           |
| ---------------------- | --------------------------------------------- | --------------------------------------- |
| TDA vectorisée         | +2 pts AUC/ARI (clustering sous-graphes)      | `tests/heavy/test_tda_vectorize.py`     |
| HGCN spectral          | +1.5 pts Macro-F1 (communautés)               | `tests/heavy/test_hgcn.py`              |
| Mapper complet         | Corrélation (ρ > 0.6) nb composantes ↔ labels | `tests/integration/test_mapper_full.py` |
| TDA rapide             | ×3 speed-up H₀/H₁ vs baseline                 | micro-bench CI heavy                    |
| Hyperbolique + fractal | +2 pts NMI/ARI, FP8 stable                    | `tests/heavy/test_hyp_embed.py`         |
| Sheaf↔Hyper bridge     | S > 0.7 sur sous-graphes denses               | `tests/unit/test_sheaf_hyper_bridge.py` |

---

## 5) TL;DR “ce que vous avez / ce qu’on ajoute”

* ✅ Déjà en place et très bien : GraphWave (Chebyshev streaming), TPL persistance (Gudhi), entropie/OT wasserstein approx, fractal dimension/box-counting stockée dans Neo4j, Hyper-SAGNN CPU/GPU (streams), Sheaf laplacian/cohomologie, Mapper (version simple), PCA sketch, FP8 (partiel), ANN hybride (nprobe\_multi).
* ➕ À ajouter pour la v1.4 (à fort levier sans tout re-écrire) :
  **(1)** Persistence Images/Landscapes (vectorisation TDA),
  **(2)** Convolution spectrale d’hypergraphe (Laplacien + K-Chebyshev),
  **(3)** Mapper “complet” (lens/overlap/clusterer),
  **(4)** TDA H₀/H₁ rapide (Union-Find + cache + incrémental),
  **(5)** Embedding hyperbolique appris avec régularisation fractale,
  **(6)** Score pont sheaf↔hypergraphe (cohérence spectrale).

Si vous le souhaitez, je peux vous livrer les squelettes de modules/tests correspondant aux points 2.1 → 2.6 pour intégrer proprement ces briques dans votre pipeline actuel.

### History
- Reset AGENTS with new roadmap as per user instructions.
- Added persistence image/landscape utilities with heavy tests.
- Implemented hypergraph spectral convolution and tests.
- Added LRU cache for persistence diagrams with property-based tests.
- Added persistence-image augmentation for embeddings with heavy test.
- Added pipeline hook combining Hyper-SAGNN with HGCN spectral convolution.
- Implemented configurable Mapper with PCA/GraphWave lens and JSON export; added integration tests.
- Implemented fast persistence diagrams with union-find, caching and unit tests.
- Implemented sheaf-hypergraph bridge with Prometheus metric and unit test.
- Added learned hyperbolic embeddings with fractal regularization and heavy tests.
- Added FP8 recall benchmark to guard quantization quality.
- Added migration for fractal_dim full-text index.
- Added persistence silhouette vectorization with stability test.
- Extended persistence augmentation with landscape and silhouette options and
  added corresponding heavy tests.
- Added Betti curve vectorization and corresponding augmentation tests.
- Exposed betti_curve via analysis package and added import test.
- Added diagram entropy vectorization and entropy-based embedding augmentation
  with heavy tests.

- Verified roadmap task implementations; pre-commit and targeted tests pass.
