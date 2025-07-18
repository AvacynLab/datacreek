# AGENTS Checklist

----------
Voici la **check-list de corrections / améliorations à intégrer** (code + tests + tput) couvrant *tous* les points que j’ai signalés dans mon dernier audit.
Structure : cases à cocher → sous-tâches → éventuellement sous-sous-tâches.
Chaque bloc inclut : **Maths**, **Variables**, **Objectif / Correctif attendu**, et **Définition de fin de tâche (DoD)**.
Je regroupe par domaine; cochez lorsque toutes les sous-cases d’un bloc sont accomplies.

---

## 1. Compléter & durcir l’implémentation Hutch++/Diag++ pour estimation diagonale de f(L)

* [ ] **Implémenter la phase “range finder” basse-dimension (Q) manquante**
  * [ ] Générer $s_1$ vecteurs gaussiens/Rademacher $G$ ; calculer $Y = A G$ (ici $A = f(L)$).
  * [ ] Orthonormaliser $Y$ → $Q = \text{qr}(Y)$ (block Gram-Schmidt ou Lanczos court).
  * [ ] Optionnel : puissance $q$ pour renforcer séparation spectrale ($Y = (A A^\top)^q A G$) si cond. mal.

* [ ] **Projeter bas-rang & corriger résidu (Hutch++ core)**
  * [ ] $B = Q^\top A Q$ (petite matrice).
  * [ ] Contribution bas-rang à la diagonale : $\mathrm{diag}(Q B Q^\top)$.
  * [ ] Résidu : $\tilde A = A - Q B Q^\top$ ; approx. diag résidu par Hutchinson classique sur $s_2$ vecteurs $Z$.
  * [ ] Estimateur final : $\widehat{\mathrm{diag}}(A) = \mathrm{diag}(Q B Q^\top) + \frac{1}{s_2}\sum_i (Z_i \odot (\tilde A Z_i))$.

* [ ] **Paramétrage échantillons**
  * [ ] Choisir $(s_1, s_2)$ en fonction de tolérance variance cible $\sigma^2_{\text{rel}}$ (visée ≤ 1e-3).
  * [ ] Heuristique : $s_1 = \lceil k + 5 \rceil$ (k rang effectif), $s_2 = \lceil 2/\epsilon \rceil$ où $\epsilon$ erreur tolérée.

* [ ] **Chebyshev évaluation efficace de f(L)**
  * [ ] Approximer $f(L)$ via polynôme de Chebyshev d’ordre $m$ avec récurrence $T_k$.
  * [ ] Normaliser spectre $L’ = (2L - (\lambda_{\max}+\lambda_{\min})I)/(\lambda_{\max}-\lambda_{\min})$ pour que $x\in[-1,1]$.

* [ ] **Tests de variance vs ground truth**
  * [ ] Générer mini-graphes où diag(f(L)) est calculable dense (SciPy eigendecomp).
  * [ ] Mesurer MAPE & variance sur 100 runs ; échouer si MAPE > 1 % ou var > 1e-3.

**Maths récap**
$$
\begin{aligned}
Y &= A G,\quad Q = \text{orth}(Y),\\
B &= Q^\top A Q,\\
\widehat{\mathrm{diag}}(A) &= \mathrm{diag}(Q B Q^\top) + \frac{1}{s_2}\sum_{i=1}^{s_2} Z_i \odot (\tilde A Z_i),\quad \tilde A = A - Q B Q^\top.
\end{aligned}
$$

**Variables**

| Symbole      | Description                                             |
| ------------ | ------------------------------------------------------- |
| $A$        | Matrice cible $f(L)$ (p.ex. filtre spectre GraphWave) |
| $L$        | Laplacien du graphe                                     |
| $G$        | Matrice aléatoire $(n\times s_1)$ pour range finding |
| $Q$        | Base orthonormée approx. sous-espace dominant           |
| $B$        | Petite matrice projetée $(s_1\times s_1)$           |
| $Z_i$     | Vecteurs Rademacher pour estim. Hutchinson résiduel     |
| $\epsilon$ | Tolérance relative d’erreur cible                       |

**DoD** : Fonction `chebyshev_diag_hutchpp()` renvoie diag avec erreur ≤ 1 % sur bancs tests; bench tput ≤ 1.5× temps Hutchinson simple pour $k/n ≤ 0.05$. Tests paramétriques passants en CI (marqués “heavy” si > 5 s).
([chrismusco.com][1], [SIAM Ebooks][2], [kth.diva-portal.org][3], [arXiv][4])

---

## 2. Bench courbure & sur-contraction pour re-centrage Poincaré (Möbius exp/log)

* [ ] **Tracer overshoot vs rayon**
  * [ ] Échantillonner points $x$ à différentes normes hyperboliques $|x|_ℍ$; appliquer re-centrage actuel vs exp/log Möbius; mesurer $\Delta r = r_{\text{target}}-r_{\text{obt}}$.
  * [ ] Générer courbes pour $κ \in {-1,-0.5,-2}$.

* [ ] **Valider implémentation exp/log**
  * [ ] Vérifier que $\exp_0(\log_0(x)) \approx x$ (erreur <1e-6).
  * [ ] Test gradient auto-diff (backprop) stable en fp16 → tolérance relative 1e-3.

* [ ] **Clamp rayon de sécurité**
  * [ ] Enforcer $|x|_2 < 1 - \delta$ (p.ex. $\delta=1e^{-6}$) après update Möbius addition.

**Maths rappel**
Möbius addition $x \oplus_c y = \frac{(1+2c\langle x,y\rangle + c|y|^2)x + (1-c|x|^2)y}{1+2c\langle x,y\rangle + c^2|x|^2|y|^2}$; exp/log maps dérivées de la géométrie de la boule de Poincaré.

**Variables**

| Symbole    | Description                                          |
| ---------- | ---------------------------------------------------- |
| $c$      | Inverse du rayon de courbure ($c>0$ pour $κ=-c$) |
| $x,c$    | Points dans la boule                                 |
| $κ$      | Courbure négative                                    |
| $\delta$ | Marge anti-bord                                      |

**Objectif** : réduire overshoot median de 15 % vs baseline; aucune NaN en fp16 stress-tests.
([GitHub][5], [Geomstats][6], [GitHub][7], [arXiv][8])

---

## 3. Autotune Node2Vec : ajouter **plafond temps** & mode *early-stop sécuritaire*

* [ ] **Budget mur (wall-clock)** paramétrable (`max_minutes`, défaut 30).
* [ ] **Critère d’arrêt multi-conditions** : arrêter si Var∥Φ∥ < seuil **ou** budget mur dépassé **ou** aucune amélioration EI 5 iters.
* [ ] **Persist last-best** p,q dans artefact JSON + hash dataset.
* [ ] **Tests** simulant budget mur court (patcher timer).

**Maths rappel (EI simplifié)**
$EI(p,q) = \max{0, \mu^* - \mu(p,q) - \xi}$ où $\mu$ score (−loss) attendu, $\xi$ jitter d’exploration.

**Variables**

| Var        | Rôle                                  |
| ---------- | ------------------------------------- |
| $p,q$    | Bias retour / in-out Node2Vec         |
| $\mu^*$ | Meilleure perf observée               |
| $\xi$    | Param exploration                     |
| Var∥Φ∥     | Variance norme embeddings (stabilité) |

**Objectif** : éviter run non borné; garantir au moins 5 explorations; produire p,q < 5 % du meilleur recall sans dépassement temps.
([ScienceDirect][9], [neptune.ai][10], [Deus Ex Machina][11])

---

## 4. Whisper batch GPU : ajouter **fallback CPU automatique** + tests

* [ ] **Détection device** (`torch.cuda.is_available()` ou binding C++) → route GPU sinon CPU.
* [ ] **Graceful degrade** : si GPU indisponible ou OOM, réduire batch & basculer CPU; alerter métrique.
* [ ] **Bench** : comparer temps réel (xRT) GPU vs CPU; log dans Prometheus.
* [ ] **Param quantisation** (8-bit / int8) pour CPU accéléré.

**Maths tput**
$xRT = \frac{T_{\text{traitement}}}{T_{\text{audio}}}$ (viser xRT ≤ 0.5 GPU; ≤ 2 CPU).

**Variables**

| Var   | Description              |
| ----- | ------------------------ |
| $B$ | batch_size segments (s) |
| $M$ | modèle Whisper choisi    |
| xRT   | temps relatif            |

**Objectif** : ne jamais bloquer pipeline tests CI sans GPU; atteindre ≥ 4× accélération GPU vs CPU de base.
([Modal][12], [GitHub][13], [Better Programming][14], [GitHub][15])

---

## 5. Export métrique Prometheus pour causes d’éviction LMDB

* [ ] Étendre exporter : `lmdb_evictions_total{cause="ttl"|"quota"|"manual"}` compteur.
* [ ] Ajouter gauge `lmdb_eviction_last_ts` par cause.
* [ ] Documenter labels cardinalité faible (<5).
* [ ] Dashboard Grafana panneau ratio TTL/quota.

**Rationale** : visibilité fine = corrélation perf & contraintes stockage; pratiques recommandées log structuré + métadonnées cause.
([Medium][16], [Sematext][17], [brennonloveless.medium.com][18])

---

## 6. Param exposé & tuning initial pour PID de TTL Redis

* [ ] **Expose config** `target_hit_ratio`, `Kp`, `Ki`, (Kd=0 initial) dans `config.yaml`.
* [ ] **Limiter intégrale** (anti-windup) : clamp somme err ∈ \[−Imax,+Imax].
* [ ] **Pas d’échantillonnage** Δt constant; calcul discret :
  $$
  e_k = h_{\text{target}} - h_k;\quad I_k = I_{k-1} + e_k \Delta t;\quad u_k = K_p e_k + K_i I_k.
  $$
  TTL_{new} = clamp(TTL_{old} + u_k, \[TTL_{min}, TTL_{max}]).
* [ ] **Test en boucle simulée** (script) sur traces hit_ratio; valider convergence monotone.

**Variables**

| Var           | Description                |
| ------------- | -------------------------- |
| $h_k$      | hit ratio mesuré fenêtre k |
| $e_k$      | erreur                     |
| $K_p,K_i$ | gains PID                  |
| $u_k$      | correction TTL             |

**Objectif** : stabiliser hit_ratio ±5 % autour cible sous charge burst; éviter oscillations soutenues.
([ctms.engin.umich.edu][19], [apmonitor.com][20], [DigiKey][21], [Wikipédia][22])

---

## 7. Renforcer règles d’alerte Prometheus (redis_hit_ratio / eigsh_timeouts) & anti-bruit

* [ ] Ajouter `for:` clause (p.ex. 5m) pour lissage.
* [ ] Utiliser seuils paramétrables via métriques jointes (recording rules) pour environnement-spécifique.
* [ ] Classer severities (warning vs critical) + inhibition pour éviter tempête.
* [ ] Test `promtool test rules` en CI.

**Objectif** : alertes actionnables, bruit < 5 faux positifs/jour env prod.
([prometheus.io][23], [promlabs.com][24], [Sysdig][25], [samber.github.io][26])

---

## 8. Migration live HAA index (normaliser paires) + script de dédup Neo4j

* [ ] **Cypher dédup** : regrouper relations `MATCH (a)-[r:HAA]->(b) WITH a,b,collect(r) as rs WHERE size(rs)>1 ...` conserver min id; delete autres.
* [ ] **Forcer ordre min,max** lors insert (déjà en code) → valider par contrainte d’unicité composite.
* [ ] **Backup & dry-run** via `neo4j-admin copy` sur snapshot avant migration.
* [ ] **Migration tool** (neo4j-migrations) versionnée.

**Objectif** : 0 duplicat résiduel; downtime < 5 min sur 10 M edges.
([Stack Overflow][27], [michael-simons.github.io][28], [Graph Database & Analytics][29])

---

## 9. Tests FAISS / nprobe : marques conditionnelles & tuning auto

* [ ] Marquer tests lourds `@pytest.mark.faiss_gpu` ; skip si lib manquante.
* [ ] Ajouter test CPU fallback (faiss-cpu).
* [ ] Courbe recall/latence vs nprobe 32→256; sélectionner via BO ou grid adaptatif.
* [ ] Persister profil (pkl) pour production.

**Rappel trade-off** : plus grand nprobe ↑ recall mais ↑ latence; calibrer pour recall≥0.92 P95 < 20 ms.
([GitHub][30], [Couchbase][31], [Milvus][32], [Medium][33])

---

## 10. TTL Manager : protocole d’arrêt propre + tests cancellation

* [ ] Ajouter `asyncio.Event` stop; join tâches.
* [ ] Garantir flush dernier cycle PID avant arrêt.
* [ ] Test de fuite tâche (pytest warns).

**Rôle** : éviter threads zombies & blocage shutdown service.
([ctms.engin.umich.edu][19], [apmonitor.com][20], [DigiKey][21])

---

## 11. Application stricte des budgets ε (DP) côté Gateway

* [ ] Middleware : intercepter requêtes DP; appeler `can_consume_epsilon(tenant, ε_req)`.
* [ ] Décrémenter budget atomiquement (txn DB).
* [ ] Rejeter (HTTP 403) si budget insuffisant; inclure en-tête `X-Epsilon-Remaining`.
* [ ] Tâche reset périodique (mensuel) ou consommation cumulative selon politique.
* [ ] UI monitoring budgets.

**Maths base DP composition**
Budget restant $ε_{\text{rem}} = ε_{\max} - \sum_i ε_i$ (composition naïve); pour composition avancée (Moment Accountant) adapter.

**Objectif** : aucune requête au-delà budget; audit log complet.
([AWS Documentation][34], [USENIX][35], [utrechtuniversity.github.io][36], [IAB Tech Lab][37])

---

## 12. Back-pressure ingestion (MAJEUR : Manquant)

* [ ] **File bornée côté producer**
  * [ ] `asyncio.Queue(maxsize=10_000)` ou canal streaming équivalent.
  * [ ] Métrique `ingest_queue_fill_ratio = qsize/maxsize`.
* [ ] **Circuit breaker sur write Neo4j**
  * [ ] Utiliser lib (pyhystrix / pybreaker) avec seuil erreur & timeout.
  * [ ] États CLOSE→OPEN→HALF_OPEN; fallback *shed* messages (HTTP 429).
* [ ] **Stratégie drop / retry**
  * [ ] Politique progressive backoff exp.
  * [ ] Stockage tampon disque si drop non acceptable (configurable).
* [ ] **Tests charge burst** (synthetic Kafka) : injecter 2× tput nominal; valider que latence ne diverge pas et que taux drop < 1 %.

**Maths simple file M/M/1/K** (approx)
Taux rejet $\approx \frac{(1-ρ)ρ^K}{1-ρ^{K+1}}$ où $ρ = λ/μ$; choisir $K$ pour P_drop<1e-3 à ρ=0.9.

**Variables**

| Var   | Description      |
| ----- | ---------------- |
| $λ$ | débit entrants   |
| $μ$ | débit traitement |
| $K$ | taille file      |
| $ρ$ | utilisation      |

**Objectif** : absorber bursts courte durée sans crash; rétro-pression propre en aval.
([GitHub][38], [thebackenddevelopers.substack.com][39], [codereliant.io][40])

---

## 13. Finaliser **GPU GraphWave** (accélération Chebyshev kernels)

* [ ] Portage Chebyshev MVM sur cuSPARSE/CSR; fusion itérations pour réduire mémoire.
* [ ] Batched diffusion pour multi-t valeurs; stream CUDA.
* [ ] Comparer contre impl CPU; viser ≥ 3× speed sur 10 M nœuds.
* [ ] Ajouter test précision (L2 diff < 1e-5).

**Contexte** : GraphWave calcul diffusion spectrale; Chebyshev approx + GPU sparse kernels = accélération significative.
([Computer Science][41], [ACM Digital Library][42], [arXiv][43])

---

## 14. Implémenter **TPL (Topological Pipeline) incrémental** pour sous-graphes modifiés

* [ ] Détection changements (hash edges / timestamps).
* [ ] Extraire sous-complexe affecté; recalcul persistance locale seulement.
* [ ] Fusion résultats partiels dans diagramme global.
* [ ] Bench vs recompute total (≥ 2× plus rapide quand < 20 % edges changent).

**Références algorithmiques** : calcul persistance incrémentale / dynamique; méthodes de réduction partielle; scheduling rapide.
([graphics.stanford.edu][44], [arXiv][45], [SpringerOpen][46])

---

## 15. Endpoint Explainability (3-hop + heatmap attention via Observable Plot)

* [ ] API REST `/explain/{node}` : retourne sous-graphe rayon=3, poids attention par arête.
* [ ] Sérialiser JSON → front JS; visualiser heatmap (cell, raster) avec Observable Plot.
* [ ] Mode export SVG/PNG pour rapports.
* [ ] Tests snapshot (golden).

**Objectif** : introspection rapide pour support client & debugging embeddings.
([observablehq.com][47], [observablehq.com][48], [observablehq.com][49])

---

## 16. Hybrid ANN (HNSW pré-filtre → PQ rerank / exact refine)

* [ ] Étape 1 : HNSW top-K rapides en RAM.
* [ ] Étape 2 : rerank candidats via IVFPQ/Exact distance (GPU ou CPU).
* [ ] Param $K$ intermédiaire & nprobe fine; calibrer latence vs recall.
* [ ] A/B test vs pure IVFPQ & pure HNSW.

**Objectif** : réduire mémoire GPU ~50 % pour recall équivalent; latence P95 ≤ 20 ms.
([zilliz.com][50], [Analytics Vidhya][51], [ApX Machine Learning][52])

---

## 17. Plugin pgvector d’export embeddings

* [ ] Schéma table `embedding(node_id UUID, space TEXT, vec vector(dim))`.
* [ ] Batch COPY en colonnes; index ivfflat pgvector (lists param).
* [ ] Tests latence SELECT top-k; valider ≤ 30 ms sur 1 M vecteurs.
* [ ] Comparer recall vs FAISS baseline.

**Rationale** : analytics SQL ad-hoc; compromis latence/recall selon lists/probes.
([GitHub][30], [Couchbase][31], [Milvus][32])

---

## 18. Stabilisation CI multi-environnements (GPU / CPU)

* [ ] Conditionnels install : `faiss-gpu` si CUDA sinon `faiss-cpu`.
* [ ] Séparer jobs *unit* (rapide) vs *heavy* (gpu, bench).
* [ ] Collecter artefacts bench JSON; comparer aux seuils régression.
* [ ] Documenter dépendances (README CI).

**Objectif** : CI verte sur runners sans GPU; detection régression perf contrôlée.
([GitHub][30], [Couchbase][31], [Medium][33])

---

## 19. Observabilité globale LMDB / cache (corrélation avec TTL PID & back-pressure)

* [ ] Dashboard : evictions par cause, hit_ratio, temps d’accès, queue ingestion.
* [ ] Alertes combinées (multi-metrics threshold) pour éviter avalanche.
* [ ] Log structuré JSON → pipeline central (ELK/Cloud).

**Objectif** : MTTR incidents cache < 15 min; moins d’alertes orphelines.
([Medium][16], [Sematext][17], [brennonloveless.medium.com][18])

---

## 20. Bench tput consolidé & seuils CI (gate)

* [ ] Étendre `bench_all.py` pour mesurer : ingestion/s, GraphWave/s, ANN P95, Whisper xRT, PID convergence temps.
* [ ] Sauver résultats commit-taggés; comparer N-1; échouer si régression > 5 % (configurable).
* [ ] Publier graphe tendance (md badge).

**Objectif** : dérives perfs visibles avant release; corrélation commit SHA model-card.
([promlabs.com][24], [Sysdig][25], [samber.github.io][26])

---

### Résumé des objectifs chiffrés clés

| Domaine             | KPI                                      | Cible |
| ------------------- | ---------------------------------------- | ----- |
| Hutch++ diag        | Rel. err ≤ 1 %, var ≤ 1e-3               |       |
| Poincaré recentrage | Overshoot −15 % vs baseline              |       |
| Node2Vec autotune   | Arrêt < 30 min; recall@10 ≥ baseline    |       |
| Whisper fallback    | xRT GPU ≤ 0.5; CPU degrade sans échec CI |       |
| Redis TTL PID       | Hit ratio ±5 % cible en burst            |       |
| Back-pressure       | Drop < 1 % @ 2× charge                   |       |
| FAISS nprobe        | Recall ≥ 0.92 P95<20 ms                  |       |
| GPU GraphWave       | ≥ 3× speed vs CPU                        |       |

(Ces KPI se retrouvent dans les blocs ci-dessus; table fournie pour lecture rapide.)

---

## Checklist
- [ ] 1. Hutch++ diag
- [ ] 2. Poincaré recentrage
- [ ] 3. Node2Vec autotune
- [ ] 4. Whisper batch fallback
- [ ] 5. Export métrique LMDB
- [ ] 6. TTL PID config
- [ ] 7. Alert rules
- [ ] 8. HAA dedup migration
- [ ] 9. FAISS nprobe tuning
- [ ] 10. TTL Manager shutdown
- [ ] 11. Budgets DP gateway
- [ ] 12. Back-pressure ingestion
- [ ] 13. GPU GraphWave
- [ ] 14. TPL incrémental
- [ ] 15. Endpoint Explainability
- [ ] 16. Hybrid ANN
- [ ] 17. Plugin pgvector
- [ ] 18. CI multi-env
- [ ] 19. Observabilité cache
- [ ] 20. Bench throughput

## History
- Reset tasks and ran formatting hooks
