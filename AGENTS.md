# Agent Tasklist v1.1-scale-out

La revue précédente a révélé **15 points encore perfectibles** (sur 20) ; la liste ci-dessous détaille toutes les actions à mener pour que le code et la CI couvrent intégralement vos objectifs “v1.1-scale-out”. Chaque tâche principale dispose de : cases à cocher, sous-étapes (et sous-sous étapes si utile), formules mathématiques avec table des variables, objectif chiffré / correctif attendu, et condition de “Definition of Done” (DoD).

## 1 – Bench overshoot & robustesse fp16 du re-centrage Poincaré
* [ ] **Tracer overshoot Δr vs norme hyperbolique**
  * [ ] Échantillonner 1000 points sur la boule de Poincaré pour trois courbures $κ∈{-1,-0.5,-2}$ ([arXiv][1], [Reddit][2])
  * [ ] Calculer $\Delta r = r_{\text{cible}} - r_{\text{obtenu}}$ après re‑centrage Möbius (`exp / log`) ; sauvegarder parquet.
* [ ] **Vérifier bijection exp/log** $(\exp_0\circ\log_0)(x)\approx x$ tol < 1e‑6 fp32, 1e‑3 fp16.
* [ ] **Clamp de sécurité** : forcer $|x|_2 < 1-10^{-6}$ après toute opération.
* [ ] **DoD** : médiane |Δr| −15 % vs scaling naïf ; aucun NaN fp16.

**Maths**
$$
\Delta r = \bigl\|x^{\text{exp/log}}\bigr\|_ℍ - \bigl\|x^{\text{target}}\bigr\|_ℍ
$$

| Symb. | Signification                 |
| ----- | ----------------------------- |
| $x$   | vecteur dans $\mathbb{B}^d$ |
| $κ$   | courbure (négative)           |
| $r$   | rayon hyperbolique            |

## 2 – Whisper.cpp : fallback CPU + métrique xRT
* [ ] **Détection device** `torch.cuda.is_available()` ; si `False` → route CPU.
* [ ] **Réduction auto `batch_size`** CPU : $B_{\text{CPU}}=\max(1,\lfloor B_{\text{GPU}}/4\rfloor)$.
* [ ] **Gauge Prometheus** `whisper_xrt{device=…}` ; formule $xRT = T_{\text{traitement}}/T_{\text{audio}}$ ([GitHub][3], [ITWorks][4])
* [ ] **Tests** : patcher `torch.cuda.is_available` → False ; vérifier passage CPU & xRT ≤ 2.
* [ ] **DoD** : pipeline CI passe sans GPU ; GPU ≤ 0.5 xRT, CPU ≤ 2 xRT.

## 3 – Node2Vec : couverture “timeout mur” & persistance params
* [ ] Simuler `max_minutes=0.1` via monkeypatch timer → vérifier arrêt après 2 itérations.
* [ ] Sauver `best_pq.json` (p,q,hash dataset) sur `Optimizer.stop` ([Let’s talk about science!][5], [DIVA Portal][6])
* [ ] **DoD** : fichier présent, test paramétrique rouge si absent.

## 4 – PID TTL Redis : exposer config & tests convergence
* [ ] Ajouter bloc YAML :

```yaml
pid:
  target_hit_ratio: 0.45
  Kp: 0.4
  Ki: 0.05
  I_max: 5
```
* [ ] **Anti‑windup** : $I_k=\min(\max(I_{k-1}+e_kΔt,-I_{max}),I_{max})$ ([Redis][7], [Graph Database & Analytics][8])
* [ ] Boucle discrète : $u_k = K_p e_k + K_i I_k$ ; clamp TTL∈[1 s,24 h].
* [ ] Test boucle sur trace hit_ratio burst ; assert overshoot < 5 %.
* [ ] DoD : hit_ratio stabilisé ±5 % autour cible.

## 5 – Migration HAA : intégration Flyway + dry‑run
* [ ] Insérer `migrations/2025-07-haa_index.cypher` dans Flyway config.
* [ ] Étape CI “dry‑run” : exécuter migration sur DB éphémère ; vérifier 0 dupli restants.
* [ ] Snapshot avant/ après avec checksum.
* [ ] DoD : CI verte + script rollback auto.

## 6 – TTL manager : test fuite tâche & arrêt propre
* [ ] Créer test `test_redis_pid_leak.py` : lancer boucle 2 cycles, déclencher `stop_event`, s’assurer `asyncio.all_tasks()`==0.
* [ ] DoD : pytest sans warning “Task was destroyed but it is pending”.

## 7 – Budget ε DP : test dépassement & header REST
* [ ] Simuler tenant ε_max = 3 ; requête ε_req = 5 → attendre 403 + header `X-Epsilon-Remaining: 0`.
* [ ] Log audit JSON `{"tenant":…, "eps_req":…, "allowed":False}`.
* [ ] DoD : test API passe ; log présent.

## 8 – Back‑pressure ingestion : circuit‑breaker Neo4j
* [ ] Intégrer `pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30)` autour de `neo4j_writer.write_batch`.
* [ ] Renvoyer HTTP 429 quand breaker OPEN.
* [ ] Métrique `breaker_state` (0 CLOSE, 1 OPEN).
* [ ] **File bornée** déjà en place ; relier métrique `ingest_queue_fill_ratio`.
* [ ] **Test burst** : injecter 2× débit → drop < 1 % ; breaker passe OPEN puis HALF_OPEN.
* [ ] **Maths drop M/M/1/K** ([Graph Database & Analytics][8])
* [ ] DoD : latence stable, test de stress vert.

## 9 – GraphWave CUDA : tests précision & skip GPU‑less
* [ ] Générer mini‑graphe (1 k nœuds) ; comparer diffusion CPU vs GPU (cupy) : $|Φ_{\text{GPU}}-Φ_{\text{CPU}}|_2/|Φ|_2<1e^{-5}$.
* [ ] Marquer `@pytest.mark.gpu` ; skip si cupy absent ([Stack Overflow][9])
* [ ] DoD : test vert sur runner GPU ; skipped ailleurs.

## 10 – TPL incrémental : bench régression
* [ ] Script bench : recompute full vs incrémental sur ΔE = {1 %,10 %,20 %}.
* [ ] Objectif : speedup ≥ 2× quand ΔE ≤ 20 %.
* [ ] Fail CI si ratio < 2×.

## 11 – Endpoint `/explain/{node}` FastAPI
* [ ] Créer `routers/explain_router.py` :
  * [ ] GET renvoie JSON `{nodes:[], edges:[], attn:[]}` + SVG (Base64).
  * [ ] CORS headers & auth.
* [ ] Front demo Observable Plot (docs).
* [ ] Tests snapshot JSON et 200 OK.
* [ ] DoD : route déployée ; UI doc OpenAPI.

## 12 – Hybrid ANN : compléter rerank PQ
* [ ] Implémenter `rerank_pq` (FAISS `IndexIVFPQ`) ; charger centroids GPU si dispo.
* [ ] Pipeline : HNSW top‑50 → PQ distance exact sur GPU ; fallback CPU.
* [ ] Bench : recall@100 ≥ 0.92, latence P95 < 20 ms ([DIVA Portal][6])
* [ ] DoD : test `test_hybrid_ann.py` vert, bench JSON enregistré.

## 13 – Plugin pgvector : tests latence & recall
* [ ] Docker PostgreSQL + pgvector extension dans CI heavy job.
* [ ] Insérer 1 M vecteurs ; index `ivfflat lists=100`.
* [ ] Test SELECT nearest 5 ; latence < 30 ms ; recall ≥ 0.9 compared FAISS CPU baseline.
* [ ] Gauge Prometheus `pgvector_query_ms`.
* [ ] DoD : test heavy marqué ; skip si postgres absent.

## 14 – CI multi‑environnements & dépendances
* [ ] Ajouter `requirements-ci.txt` : `fakeredis`, `cupy-cuda12x`, `faiss-cpu`.
* [ ] Workflow `ci.yml` : matrix `{gpu:false,true}` ; job `unit` puis `heavy` (needs unit).
* [ ] Ajouter étape `promtool test rules` ([GitHub][10])
* [ ] Timeout heavy → 45 min.
* [ ] DoD : CI verte CPU‑only et GPU runners.

## 15 – Import dashboard Grafana cache/TTL
* [ ] Script `upload_dashboard.py` via Grafana HTTP API ; exécuter en CD.
* [ ] Vérifier panel `lmdb_evictions_total` et `ingest_queue_fill_ratio`.
* [ ] DoD : dashboard présent en prod ; lien dans README.

### Récap objectifs chiffrés

| Point              | KPI cible                  | Régression tolérée |
| ------------------ | -------------------------- | ------------------ |
| Overshoot Poincaré | −15 %                      | 0 %                |
| Whisper CPU        | xRT ≤ 2                    | +10 %              |
| TTL PID            | ±5 % hit_ratio            | ±1 %               |
| Backpressure       | drop < 1 %                 | +0.5 %             |
| GPU↔CPU GraphWave  | L2 < 1e‑5                  | 0                  |
| Hybrid ANN         | recall ≥ 0.92, P95 < 20 ms | −0.02 / +2 ms      |
| pgvector           | lat < 30 ms                | +5 ms              |

[1]: https://arxiv.org/abs/1305.7164?utm_source=chatgpt.com "[1305.7164] On Poincaré extensions of rational maps - arXiv"
[2]: https://www.reddit.com/r/MachineLearning/comments/vcr18m/poincare_embeddings_embedding_your_data_in_low/?utm_source=chatgpt.com "Poincare Embeddings: Embedding your data in low dimensions [P]"
[3]: https://github.com/ggerganov/whisper.cpp/discussions/403?utm_source=chatgpt.com "Tuning for threads, CPUs and cores · ggml-org whisper.cpp - GitHub"
[4]: https://www.itworks.hu/running-whisper-cpp-on-windows/?utm_source=chatgpt.com "Running whisper.cpp on Windows - ITWorks"
[5]: https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html?utm_source=chatgpt.com "Acquisition functions in Bayesian Optimization"
[6]: https://www.diva-portal.org/smash/get/diva2%3A1534408/FULLTEXT01.pdf?utm_source=chatgpt.com "[PDF] Bayesian Optimization for Neural Architecture Search using Graph ..."
[7]: https://redis.io/blog/why-your-cache-hit-ratio-strategy-needs-an-update/?utm_source=chatgpt.com "Why your cache hit ratio strategy needs an update - Redis"
[8]: https://neo4j.com/docs/python-manual/current/concurrency/?utm_source=chatgpt.com "Run concurrent transactions - Neo4j Python Driver Manual"
[9]: https://stackoverflow.com/questions/38112325/using-cusparse-in-nvgraph-as-the-connection-matrix?utm_source=chatgpt.com "Using cuSPARSE in nvGraph as the connection matrix?"
[10]: https://github.com/kubernetes/kube-state-metrics/issues/1389?utm_source=chatgpt.com "\"Evicted\" pods don't register metrics · Issue #1389 - GitHub"

## History
- Reset backlog to new task list and added YAML fallback in config loader.
