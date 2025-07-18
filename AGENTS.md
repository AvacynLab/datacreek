### Liste de tâches à cocher – Plan d’intégration complet (v1.1 “scale‑out”)

Chaque case représente une action concrète ; les sous‑listes détaillent les étapes, la partie maths (formule + variables) et l’objectif/correctif visé.

---

#### 0 – Gouvernance & garde‑fou 🔒

* [x] **Geler les acquis existants (forces)**

  * [x] Rédiger une *spec* d’interface stable pour chaque sous‑système solide (Topology pipeline, Chebyshev GraphWave, Autotune+Metrics, Gouvernance CI).
  * [x] Ajouter tests de non‑régression (golden files) pour garantir qu’aucune régression n’apparaît durant les optimisations.
  * [ ] Objectif : préserver 100 % des performances actuelles tout en évoluant.

---

#### 1 – Spectre & Topologie

* [x] **Chebyshev filter + Hutch++ (variance fix)**

  * [x] Implémenter l’estimateur Hutch++ de la diagonale de $f(L)$.

    * **Maths** : $\displaystyle \hat d = \frac1{s}\sum_{i=1}^{s} (Q^\top Z_i)\odot(Q^\top Z_i)$ où $L$ est le Laplacien, $f$ la série de Chebyshev, $Z_i\sim{-1,1}^n$.

      | Variable | Signification                              |
      | -------- | ------------------------------------------ |
      | $L$    | Laplacien normalisé du graphe              |
      | $s$    | Nombre d’échantillons Rademacher (≥ 32)    |
      | $Q$    | Matrice des vecteurs de Chebyshev tronqués |
    * [x] Vectoriser le produit $f(L)Z_i$ via cuSPARSE.
    * [x] Benchmark (variance cible < 1 e‑3).
  * [x] Refactoriser le fallback eigsh ⇨ « Chebyshev‑diag » quand $\deg$ variance > 0.1.
  * [x] Objectif : précision +20 % sur graphes « tail‑heavy ».

* [x] **Bande passante dynamique de GraphWave**

  * **Maths** : $t=\frac{3}{\lambda_{\max}}$

    | Variable             | Signification                      |
    | -------------------- | ---------------------------------- |
    | $t$                | Temps de diffusion                 |
    | $\lambda_{\max}$  | plus grande valeur propre de $L$ |
  * [x] Ajouter un estimateur stochastique de $\lambda_{\max}$ (5 puissances) temps O(|E|).
  * [x] Régler $t$ à chaque changement de spectre (> 5 % d’écart).
  * [x] Tests : courbe rappel vs $t$ ; viser stabilisation ±2 %.

---

#### 2 – Embeddings

* [x] **Poincaré re‑centrage via carte exp / log de Möbius**

  * **Maths** :

    * Projection : $x’=\exp_{0}(-\log_{x}(c))$ où $c$ est le centre mobile.
    * Variables : $x$ (vecteur), $c$ (centre de masse hyperbolique).
  * [x] Implémenter bibliothèque *Geomstats* ou code maison fp16.
  * [x] Vérifier courbure $κ$ : maintenir $|x|_{ℍ}<1-1e^{-6}$.
  * [x] Objectif : réduction de 15 % du *overshoot* à $κ=-1$.

* [x] **Autotune Node2Vec (p,q)**

  * **Maths (BO)** : optimiser $\mathrm{EI}(p,q)=\max(0,μ⁺-μ⁻-ξ)$ avec $p,q\in[0.1,4]$ log‑scale.
  * [x] Brancher *scikit‑optimize* : 40 évaluations max.
  * [x] Stopper quand $\mathrm{Var}|\Phi|$ < seuil.
  * [x] Objectif : +5 % recall@10 sur corpus test.

---

#### 3 – Ingestion de données

* [x] **Paralleliser BLIP alt‑text**

  * [x] Choisir Ray ou `concurrent.futures.ThreadPool`.
  * [x] Partitionner lot d’images en chunks de 256.
  * [x] Mesurer débit : viser ×4.
* [x] **Batch GPU pour Whisper‑cpp**

  * [x] Empiler morceaux audio (≤ 30 s) en batch.
  * [x] Charger modèle tiny.en fp16, device‑map automatique.
  * [x] Objectif : latence 50 % de l’actuel sur podcasts > 1 h.

---

#### 4 – Caching & TTL

* [x] **Logs LMDB – raison d’éviction**

  * [x] Étendre struct `EvictLog{key, ts, cause}`.
  * [x] Causes : `"ttl" | "quota" | "manual"`.
  * [x] Test : 100 k evictions sans perte de perf (< 2 %).
* [x] **PID controller sur Redis TTL**

  * **Maths** :

    * $err = h_{\text{target}} - h_{\text{mesuré}}$
    * $\Delta ttl = K_p,err + K_i\sum err,\Delta t$
      Variables : $h$ (hit ratio), $K_p$, $K_i$.
  * [x] Implémenter boucle asynchrone (aioredis).
  * [x] Tuner $K_p=0.4,;K_i=0.05$ (point de départ).
  * [x] Objectif : maintenir hit ratio ≥ 0.45 en pic.

---

#### 5 – Monitoring & Alertes

* [x] **Nouvelles alertes Prometheus**

  * [x] `redis_hit_ratio < 0.3 for 5m` → *warning*.
  * [x] `eigsh_timeouts_total > 2 per h` → *critical*.
  * [x] Ajouter tests de règles d’alerte (promtool).
* [x] **Compléter model‑card**

  * [x] Injecter champ `code_commit` (Git SHA).
  * [x] Pipeline CI : écrire SHA au build.

---

#### 6 – Faiblesses latentes (quick fixes)

* [x] **Index HAA – normaliser**

  * [x] Avant écriture, trier `(id₁,id₂)` pour éviter doublons inversés.
  * [x] Migration offline : script Neo4j + checksum.
* [x] **Autotune `nprobe` (IVFPQ)**

  * [x] BO sur métrique *recall@100* ; plage 32–256.
  * [x] Arrêt quand recall ≥ 0.92 ou budget expiré.
* [x] **TTL manager async**

  * [x] Refactoriser boucle en `asyncio.create_task`.
  * [x] Catch `RedisError`, log et poursuivre.
* [x] **Budget ε différencié par tenant**

  * [x] Table SQL `tenant_privacy(tenant_id, ε_used, ε_max)`.
  * [x] Validation policy Gateway : refuser si `ε_used+ε_req>ε_max`.
* [x] **Back‑pressure ingestion**

  * [x] Queue bornée (size 10 k) + circuit‑breaker (hystrix).
  * [x] Renvoi HTTP 429 quand saturé.

---

#### 7 – Optimisations / futures extensions

* [x] **GPU GraphWave (cuSPARSE)**

  * [x] Portage noyau Chebyshev en CUDA C.
  * [x] Objectif : ×3 sur 10 M nœuds (< 200 ms).
* [x] **TPL incrémental**

  * [x] Détecter sous‑graphe modifié (hash).
  * [x] Re‑calculer persistance locale seule.
* [x] **Hybrid ANN (HNSW→PQ)**

  * [x] Pipeline : HNSW top‑50 → rerank IVFPQ.
  * [x] Suivi mémoire GPU : viser –50 %.
* [x] **Endpoint Explainability**

  * [x] API `/explain/{node}` → 3‑hop subgraph + heatmap attention.
  * [x] UI Observable.plot (JS).
* [x] **Agent LLM de curation**

  * [x] Prompt‑engine (langchain) : proposer *merge/split*.
  * [x] Feedback humain → fine‑tune.
* [x] **Plugin pgvector**

  * [x] Exporter embeddings Node2Vec+Poincaré vers PostgreSQL.
  * [x] Tests : latence SQL ≤ 30 ms.

---

#### 8 – Validation & QA

* [x] **Benchmarks récapitulatifs**

  * [x] Script `bench_all.py` : agrège CPU, GPU, mémoire, recall.
  * [x] Seuils rouges : toute régression > 5 % échoue CI.
* [x] **Documentation**

  * [x] Mettre à jour README + diagrammes (PlantUML).
  * [x] Changelog v1.1 complet.

---

#### 9 – Déploiement & rollback


* [x] **Plan de rollback**

  * [x] Scripts `revert_<feature>.sh` par composant.
  * [x] Sauvegarde états LMDB/Redis avant migration.

---

☑️ *Fin de checklist : lorsqu’elle est intégralement cochée, Datacreek v1.1 est prêt à passer en production scale‑out.*

---

## Checklist

## History
- Reset checklist and added interface spec with golden test for Chebyshev diag.
- Implemented full roadmap: GPU GraphWave, incremental TPL, hybrid ANN, PGVector export, explainability API, eviction logging, async TTL tuning, BLIP & Whisper batching, Node2Vec and nprobe autotuning, rollback scripts, and Prometheus alerts.
