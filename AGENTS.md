----------
Voici la **todo‑list finale à cocher** pour clore totalement la feuille de route *v1.1‑scale‑out*.
Les 5 blocs couvrent **toutes** les lacunes repérées : chaque bloc détaille les sous‑étapes (et sous‑sous‑étapes), la partie math (formules + variables), l’objectif chifré et la *Definition of Done* (DoD).

---

## 1 – Tests « no‑CUDA » & métrique **xRT** pour Whisper.cpp

* [ ] **Monkey‑patch `torch.cuda.is_available` → False** dans `tests/test_whisper_cpu.py`.
* [ ] Forcer `batch_size_cpu = max(1, ⌊B_gpu / 4⌋)` dans `whisper_batch.py`.
* [ ] Exporter gauge Prometheus `whisper_xrt{device=cpu|gpu}` ;

  $$
  xRT = \frac{T_{\text{traitement}}}{T_{\text{audio}}}
  $$

  *Variables* : $T_{\text{traitement}}$ (durée réelle), $T_{\text{audio}}$ (durée du flux) ([Tom's Hardware][1], [GitHub][2])
* [ ] **Assertions tests** : `xRT_cpu ≤ 2`, `xRT_gpu ≤ 0.5`.
* [ ] **DoD** : test passe sur runners sans GPU ; gauge visible dans `/metrics`.

---

## 2 – Bench & assertions latence **Hybrid ANN** (HNSW → IVFPQ rerank)

* [ ] Compléter `analysis/hybrid_ann.py.rerank_pq()` : utiliser `faiss.IndexIVFPQ` GPU si dispo, sinon CPU.
* [ ] Script bench (pytest “heavy”) :

  1. Index 1 M vecteurs $d = 256$.
  2. Chercher 1 k queries.
  3. Mesurer **P95** latence (95ᵉ centile) et `recall@100`.

     $$
     P95=\text{percentile}(t_{\text{query}},\,95)
     $$
* [ ] Cibles : `recall ≥ 0.92`, `P95 < 20 ms`.
* [ ] **DoD** : test échoue si seuils non atteints; résultats ajoutés à `benchmarks/`.

*Réfs* : compléments HNSW & IVFPQ ([OpenReview][3], [unum.cloud][4]).

---

## 3 – Job CI “heavy” : conteneur **PostgreSQL + pgvector** & tests latence

* [ ] Étape workflow :

  ```yaml
  services:
    postgres:
      image: ankane/pgvector:v0.6.0
      env:
        POSTGRES_PASSWORD: test
      ports: ['5432:5432']
  ```
* [ ] Installer extension : `CREATE EXTENSION IF NOT EXISTS vector;`.
* [ ] Insérer 1 M vecteurs; créer index `ivfflat lists=100`.
* [ ] Test `SELECT … ORDER BY vec <=> $q LIMIT 5` : assert latence < 30 ms et recall ≥ 0.9 vs FAISS baseline. ([Microsoft Learn][5], [Reddit][6])
* [ ] Skip test si service non dispo (runner local).
* [ ] **DoD** : heavy job vert sur GitHub Actions.

---

## 4 – Dépendances **CuPy CUDA 12.x** & autres libs dans `requirements-ci.txt`

* [ ] Ajouter :

  ````text
  cupy-cuda12x>=12.0.0  # GPU tests
  fakeredis>=2.20.0
  faiss-cpu>=1.8.0      # fallback heavy
  ````
* [ ] Dans workflow CI :

  * job `gpu`: `pip install cupy-cuda12x faiss-gpu`
  * job `unit`: installe seulement `faiss-cpu`.
* [ ] **DoD** : plus d’échec “ModuleNotFoundError: cupy / fakeredis”.

---

## 5 – Secret **Grafana API‑Token** & upload dashboard automatisé

* [ ] Stocker `GRAFANA_TOKEN` dans GitHub Secrets (scope `dashboards:write`).
* [ ] `scripts/upload_dashboard.py` :

  ````python
  headers={'Authorization': f'Bearer {token}'}
  requests.post(f'{host}/api/dashboards/db', json=payload, headers=headers)
  ````
* [ ] Étape CD : exécuter le script après déploiement.
* [ ] **DoD** : dashboard `Cache‑Backpressure` présent en PROD, version timestampée.

---

### Résumé Objectifs chiffrés

| Bloc        | KPI                           | Seuil                     | Source                       |
| ----------- | ----------------------------- | ------------------------- | ---------------------------- |
|  Whisper    | xRT_cpu ≤ 2, xRT_gpu ≤ 0.5  | Bench GPU/CPU             | turn0search8                 |
|  Hybrid ANN | recall ≥ 0.92, P95 < 20 ms    | Vector search theory      | turn0search6 & turn0search1  |
|  pgvector   | latence < 30 ms, recall ≥ 0.9 | Azure guide / Reddit perf | turn0search10 & turn0search2 |
|  CI deps    | cupy‑cuda12x installed        | PyPI                      | turn0search3                 |
|  Grafana    | Dashboard POST 200 OK         | Grafana API docs          | turn0search7                 |

> **Quand ces 5 cases principales (et leurs sous‑cases) sont cochées, la couverture fonctionnelle & de test sera **100 %**, la CI verte CPU+GPU, et l’environnement observabilité complet.**

[1]: https://www.tomshardware.com/news/whisper-audio-transcription-gpus-benchmarked?utm_source=chatgpt.com "OpenAI Whisper Audio Transcription Benchmarked on 18 GPUs: Up to 3,000 ..."
[2]: https://github.com/openai/whisper/discussions/918?utm_source=chatgpt.com "Performance benchmark of different GPUs · openai whisper - GitHub"
[3]: https://openreview.net/forum?id=s7Vh8OIIm6&utm_source=chatgpt.com "Hybrid Inverted Index Is a Robust Accelerator for Dense Retrieval"
[4]: https://www.unum.cloud/blog/2023-11-07-scaling-vector-search-with-intel?utm_source=chatgpt.com "10x Faster than Meta's FAISS | Unum Blog"
[5]: https://learn.microsoft.com/en-us/azure/cosmos-db/postgresql/howto-optimize-performance-pgvector?utm_source=chatgpt.com "How to optimize performance when using pgvector - Azure Cosmos ..."
[6]: https://www.reddit.com/r/vectordatabase/comments/1b1ixkq/how_much_is_too_much_to_consider_pgvector/?utm_source=chatgpt.com "How much is too much to consider pgvector : r/vectordatabase - Reddit"

## History
- Reset backlog as requested and fixed duplicate metric registration in `start_metrics_server`.
- Installed requirements and verified tests pass with heavy tests skipped when services are unavailable.
- Raised default `probes` to 20 in `query_topk_pg` to meet heavy test recall target.
