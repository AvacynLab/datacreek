cat > AGENTS.md <<'EOF'
Voici la **check‑list de rattrapage “v1.2‑hardening”** : 11 blocs, 36 sous‑tâches (dont sous‑sous‑tâches), chacun assorti de la partie **Maths**, d’un tableau **Variables**, de l’**objectif chiffré** et de la **Definition of Done (DoD)**.
Les références pointent vers la doc ou les discussions de référence qui guideront l’implémentation.

---

## 1. Qualité & sécurité de code

* [x] **Intégrer contrôles pré‑commit avancés**

  * [x] Ajouter à `.pre-commit-config.yaml` :

    * `docstring-quality` (pydocstyle) (≥ 80 % fonctions documentées). ([bandit.readthedocs.io][1])
    * `flake8-bandit` (analyse CVE) ([GitHub][2])
    * `radon` (complexité cyclomatique ≤ C). ([radon.readthedocs.io][3], [PyPI][4])
  * [x] Corriger violations (vars non utilisées, secrets hard‑codés, CC > 10).
  * [x] **DoD** : `pre-commit run --all-files` sort sans erreur ; badge “radon A/B”.

* [ ] **Activer couverture & badge**

  * [ ] Ajouter `pytest-cov` et job CI qui publie `coverage.xml`. ([Stack Overflow][5])
  * [ ] Gate PR : `--cov-fail-under=80`.
  * [ ] Badge Coveralls dans README.
  * [ ] **DoD** : pipeline échoue < 80 %.

---

## 2. Tests évolués

* [ ] **Jobs CI séparés**

  * [ ] `unit` (≤ 5 min, CPU, no GPU).
  * [ ] `gpu` (label `self-hosted`, installe extras `[gpu]`).
  * [ ] `heavy-nightly` (bench > 1 M vecteurs).
  * [ ] **DoD** : trois workflows “green”.

* [ ] **Property-based testing (Hypothesis)**

  * [ ] Créer `tests/property/` et ajouter :

    * GraphWave : invariance norme & symétrie.
    * Möbius addition : \$(x\oplus y)\ominus y \approx x\$.
  * [ ] **DoD** : ≥ 30 exemples générés/test, tous verts. ([Medium][6])

---

## 3. Observabilité élargie

* [ ] **OpenTelemetry**

  * [ ] Installer `opentelemetry-instrumentation-fastapi`. ([opentelemetry-python-contrib.readthedocs.io][7], [signoz.io][8])
  * [ ] Automatiser export traces HTTP → Jaeger (OTLP).
  * [ ] Propager trace-id dans logs (struct attr).
  * [ ] **DoD** : requête `/vector/search` visible dans Jaeger UI.

* [ ] **Alertes Prometheus supplémentaires**

  * [ ] `p95_graphwave_ms > 250 for 10m` (warning). ([Medium][9])
  * [ ] `ingest_queue_fill_ratio > 0.8 for 10m` (critical).
  * [ ] Tester avec `promtool test rules`.
  * [ ] **DoD** : règles passent lint & test.

* [ ] **Dashboards Jsonnet**

  * [ ] Convertir `docs/grafana/*.json` en Jsonnet avec **Grafonnet** ; compile check via CI. ([GitHub][10])
  * [ ] **DoD** : `jsonnetfmt` clean, dashboard rendu OK.

---

## 4. Performance CPU fallback

### 4.1  FAISS multi-probing

* [ ] Activer `index.nprobe_multi` pour CPU :

  * Formule de rappel attendu :

    $$
      \text{Recall} \approx 1 - \Bigl(1 - \frac{n_\text{probe}}{N_\text{cells}}\Bigr)^{L}
    $$

    | Var                 | Description       |
    | ------------------- | ----------------- |
    | \$n_\text{probe}\$ | listes visitées   |
    | \$N_\text{cells}\$ | centroides totaux |
    | \$L\$               | n° de tables (PQ) |
  * [ ] Bench 1 M vect, 32 threads → viser `recall ≥ 0.90`, `P95 < 50 ms`. ([GitHub][11], [GitHub][10])
  * [ ] **DoD** : test `test_ann_cpu.py` assert passes.

### 4.2  Whisper int8 GEMM

* [ ] Installer `bitsandbytes` : utiliser matmul 8-bit. ([Hugging Face][12], [GitHub][13])
* [ ] Mesurer $\text{xRT}=\frac{T_\text{proc}}{T_\text{audio}}$ ; cible CPU ≤ 1.5.
* [ ] **DoD** : gauge `whisper_xrt{device=cpu}` ≤ 1.5.

---

## 5. GraphWave GPU mémoire

* [ ] Implémenter **streamed Chebyshev** : traiter \$k\$ coeffs par lot \$b\$

  $$
    M = 2 n d b \quad (\text{octets})
  $$

  | \$n\$ | nœuds |
  | \$d\$ | sizeof(float32)=4 |
  | \$b\$ | blocs actifs |

  * [ ] Choisir \$b = \lceil m / \lceil V /5\text{ GB}\rceil\rceil\$.
  * [ ] **DoD** : VRAM ≤ 5 GB sur 10 M nœuds, perf ⩾ 90 % baseline.

---

## 6. Sécurité & conformité renforcées

* [ ] **Secret hygiene**

  * [ ] Renommer `.env.example` → `docs/env.sample`; aucune valeur "SECRET=xxx".
  * [ ] Dependabot YAML déjà présent (OK).
  * [ ] **DoD** : `trufflehog` scan => 0 leak.

* [ ] **Renyi DP accountant live**

  * [ ] Brancher `dp/accountant.compute_epsilon(\alpha_list)` sur middleware.
  * [ ] Stop requête si $\epsilon_{\text{Renyi}}! > \epsilon_{\max}$.

    $$
      \epsilon = \min_{\alpha>1} \frac{\ln\bigl(\sum_i e^{(\alpha-1)\epsilon_i}\bigr)}{\alpha-1}
    $$
  * [ ] Test dépassement budget.
  * [ ] **DoD** : 403 + header `X-Epsilon-Remaining: 0`.

---

## 7. Documentation & DX

* [ ] **Site MkDocs-Material**

  * [ ] `mkdocs.yml` + actions deploy GitHub Pages.
  * [ ] Pages : Quick-start CPU, Quick-start GPU, API guide.
  * [ ] **DoD** : site accessible à `/docs`.

* [ ] **Swagger examples**

  * [ ] Ajouter `example` section pour `/explain/{node}` et `/vector/search`.
  * [ ] README : snippet `curl` + JS fetch.
  * [ ] **DoD** : examples visibles dans Swagger UI.

---

## 8. CI intégration

* [ ] **Step mypy** (`mypy --strict`) : fail si erreur.
* [ ] **Jsonnet lint** (`jsonnetfmt -n`).
* [ ] **Promtool** validation rules.
* [ ] **DoD** : pipeline passe, toutes étapes verts.

---

### KPI de clôture "v1.2 GA"

| Domaine               | Seuil    | Mesure            |
| --------------------- | ------- | ----------------- |
| Couverture tests      | ≥ 80 % | `coverage.xml` CI |
| Doc-string couverture | ≥ 80 % | `docstring-quality` |
| CPU ANN P95           | < 50 ms | bench nightly     |
| CPU Whisper xRT       | ≤ 1.5 | Prometheus gauge  |
| VRAM GraphWave 10 M   | ≤ 5 GB | GPU bench         |
| Alerts false-positifs | < 3/jour | Prometheus        |

*Après validation de tous les blocs et atteinte des KPI, Datacreek pourra être tagué **v1.2-rc** puis **v1.2-GA** en confiance.*

[1]: https://bandit.readthedocs.io/en/latest/config.html?utm_source=chatgpt.com "Configuration — Bandit documentation - Read the Docs"
[2]: https://github.com/pre-commit/pre-commit/issues/2398?utm_source=chatgpt.com "Bandit Pre-commit hook · Issue #2398 - GitHub"
[3]: https://radon.readthedocs.io/en/latest/intro.html?utm_source=chatgpt.com "Introduction to Code Metrics — Radon 4.1.0 documentation"
[4]: https://pypi.org/project/radon/?utm_source=chatgpt.com "radon - PyPI"
[5]: https://stackoverflow.com/questions/29295965/python-coverage-badges-how-to-get-them?utm_source=chatgpt.com "Python coverage badges, how to get them? - Stack Overflow"
[6]: https://medium.com/clarityai-engineering/property-based-testing-a-practical-approach-in-python-with-hypothesis-and-pandas-6082d737c3ee?utm_source=chatgpt.com "Property based testing — A practical approach in Python with ..."
[7]: https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html?utm_source=chatgpt.com "OpenTelemetry FastAPI Instrumentation"
[8]: https://signoz.io/blog/opentelemetry-fastapi/?utm_source=chatgpt.com "Implementing OpenTelemetry in FastAPI - A Practical Guide - SigNoz"
[9]: https://medium.com/javarevisited/mastering-latency-metrics-p90-p95-p99-d5427faea879?utm_source=chatgpt.com "Mastering Latency Metrics: P90, P95, P99 | by Anil Gudigar - Medium"
[10]: https://github.com/facebookresearch/faiss/issues/1030?utm_source=chatgpt.com "Number of probes for multi-GPU index · Issue #1030 - GitHub"
[11]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes?utm_source=chatgpt.com "Faiss indexes · facebookresearch/faiss Wiki - GitHub"
[12]: https://huggingface.co/docs/transformers/en/main_classes/quantization?utm_source=chatgpt.com "Quantization - Hugging Face"
[13]: https://github.com/openai/whisper/discussions/988?utm_source=chatgpt.com "Parameter-Efficient Fine-Tuning of Whisper-Large V2 in Colab on T4 ..."

## History
- Checklist reset to v1.2-hardening tasks.
- Ran pre-commit and property tests; installed deps. Added trufflehog ignore.

- Added pydocstyle docstring-quality hook and updated requirements; all pre-commit checks pass.
