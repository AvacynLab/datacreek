----------
### Checklist exhaustive — “v1.2 hardening”

> Chaque case représente une action à réaliser.
> Sauf mention contraire, la **Definition of Done (DoD)** correspond à : code livré + tests unitaires/CI verts + documentation mise à jour.

---

## 0 – Meta : pilotage & gouvernance

* [x] **Créer epic GitHub “v1.2‑hardening”**

  * [x] Générer issues automatisées pour chaque tâche ci‑dessous (étiquette : `hardening`, `area/*`).
  * [x] Ajouter champ “Effort (pts)” & “Impact (🔥/⚡/🛠)”.
  * [x] Objectif : suivi burndown clair.

---

## 1 – Architecture & packaging

### 1.1  Abstraction GPU / CPU

* [x] Implémenter module `backend.array_api`

  * [x] Fonction `get_xp(obj=None)` retourne `cupy` si GPU dispo, sinon `numpy`.
  * [x] Refactor `graphwave_cuda`, `hybrid_ann`, `chebyshev_diag` pour appeler `get_xp`.
  * [x] Tests : monkey‑patch Cupy indisponible → aucun `ImportError`.

### 1.2  Plugins lourds en optionnel

* [x] Déplacer *analysis/graphwave_cuda* et *analysis/hybrid_ann* dans extra `[gpu]`.
* [x] Setup .py

  ```toml
  extras_require = {"gpu": ["cupy-cuda12x", "faiss-gpu>=1.8.0"]}
  ```
* [x] CI : job CPU n’installe pas `[gpu]`.

### 1.3  Configuration typée

* [x] Créer `config/schema.py` (Pydantic v2).
* [x] Valider YAML → modèle ; lever `ValidationError` au boot.
* [x] Variables :

  | Nom           | Type         | Défault |
  | ------------- | ------------ | ------- |
  | `pid.Kp`      | float ∈(0,1] | 0.4     |
  | `pid.Ki`      | float        | 0.05    |
  | `gpu.enabled` | bool         | False   |

---

## 2 – Qualité du code

### 2.1  Typage strict

* [x] Activer `mypy --strict` sur *datacreek/*
* [x] Ajouter suppression localisée `# type: ignore[assignment]` où nécessaire.

### 2.2  Documentation

* [x] Exiger doc‑string PEP 257 via pre‑commit `docstring-quality`.
* [x] Couverture doc‑strings cible ≥ 80 %.

### 2.3  Complexité & sécurité

* [x] Intégrer `flake8-bandit`, `radon` (fail > C cyclomatic).
* [x] Metrics badge sur README.

---

## 3 – Tests & CI

### 3.1  Couverture

* [x] Ajouter `pytest-cov`; publier **coverage.xml** artefact.
* [x] CI gate : `--cov-fail-under=80`.

### 3.2  Jobs différenciés

* [x] Découper workflows :

  1. **unit** (≤ 5 min)
  2. **heavy-nightly** (bench > 1 M vecteurs)
  3. **gpu** (labels `self-hosted`, CUDA 12)

### 3.3  Property‑based tests

* [x] Introduire `hypothesis` pour :

  * GraphWave invariants (symétrie, norme).
  * Poincaré Möbius addition associativité approx.

---

## 4 – Performance & Scalabilité

### 4.1  GraphWave streaming kernel

* [x] **Chebyshev streaming**

  * [x] Diviser ordre $m$ en blocs $b$ pour VRAM 8 GB.
  * [x] Formule mémoire :

    $$
    M = 2n\cdot d \cdot b \quad (\text{bytes})
    $$

    Variables :
    | $n$ nœuds | $d$ taille float32(=4) | $b$ blocs actifs |
  * [x] Objectif : VRAM ≤ 5 GB sur 10 M nœuds.

### 4.2  Multi‑probing FAISS (CPU fallback)

* [x] Activer `index.nprobe = base × n_subprobe`.

  * **Maths rappel** :

    $$
    \text{Recall} \approx 1 - (1 - \frac{n_\text{probe}}{N_\text{cells}})^{L}
    $$

    | $L$ tables | $N_\text{cells}$ total centroides |
  * [x] Bench : P95 < 50 ms CPU 32‑core, recall ≥ 0.9.

### 4.3  Whisper int8 GEMM

* [x] Intégrer `bitsandbytes.matmul_8bit()`.
* [x] Mesurer gain : xRT_cpu → ≤ 1.5.

---

## 5 – Sécurité & conformité

### 5.1  Secret hygiene

* [x] Remplacer `.env.example` par `docs/env.sample` (aucun placeholder sensible).
* [x] Activer GitHub Dependabot (# security updates).

### 5.2  DP Renyi accountant

* [x] Implémenter module `dp/accountant.py` (Mironov 2017).

  * Cumul ε selon moments $α$:

    $$
    \varepsilon = \min_{\alpha>1} \frac{1}{\alpha-1} \log \sum_i e^{(\alpha-1) \varepsilon_i}
    $$
  * Variables : $\varepsilon_i$ budgets requêtes.
* [x] Gateway : interdire si $\varepsilon_{\text{Renyi}}>\varepsilon_{\max}$.

---

## 6 – Observabilité & Ops

### 6.1  Tracing OpenTelemetry

* [x] Ajouter `opentelemetry-instrumentation-fastapi` & `aio-pika`.
* [x] Export OTLP → Jaeger.

### 6.2  Dashboards versionnés

* [x] Convertir `docs/grafana/*.json` → Jsonnet (`grafonnet-lib`).
* [x] CI valide rendu via `jsonnetfmt`.

### 6.3  Nouvelles alertes

* [x] Règle Prometheus :

  * `p95_graphwave_ms > 250 for 10m` (warning)
  * `ingest_queue_fill_ratio > 0.8 for 10m` (critical)

---

## 7 – Documentation & DX

### 7.1  mkdocs‑material

* [x] Config `mkdocs.yml`; pages : *quick‑start‑CPU*, *quick‑start‑GPU*.
* [x] GitHub Pages auto‑deploy.

### 7.2  API examples

* [x] Ajouter bloc code Swagger `/explain/{node}` + curl + JS fetch snippet.

---

## 8 – Plan de livraison

| Sprint | Axes clés (issus ci‑dessous)                          |
| ------ | ---------------------------------------------------- |
|  S‑1   | Typage strict, Pydantic config, secret hygiene       |
|  S‑2   | GPU abstraction layer, plugin extras, CI coverage    |
|  S‑3   | GraphWave streaming, FAISS multi‑probe, int8 Whisper |
|  S‑4   | OpenTelemetry, DP Renyi accountant, alert rules      |
|  S‑5   | Docs mkdocs, dashboards Jsonnet, release v1.2‑rc     |

---

### KPI de validation finale

| Domaine            | Cible         | Mesure                |
| ------------------ | ------------- | --------------------- |
| Couverture tests   | ≥ 80 % lignes | `coverage.xml` CI     |
| RAM GraphWave 10 M | ≤ 5 GB        | benchmark_gpu.json   |
| P95 ANN CPU        | < 50 ms       | `bench_ann_cpu.json`  |
| xRT Whisper CPU    | ≤ 1.5         | `metrics_prometheus`  |
| Bugs SAST Bandit   | 0 high        | `bandit -r datacreek` |

👉 **Quand toutes les cases sont cochées**, Datacreek sera prêt pour **v1.2 GA** : plus modulaire, plus sûr, plus rapide et mieux observable.

## History
- Reset backlog to v1.2 hardening as instructed.
- Created docs/env.sample and enabled Dependabot; removed legacy .env.example.
- Added Pydantic config schema and YAML validation defaults.
- Implemented Renyi DP accountant with gating and tests.
- Added GPU/CPU array_api abstraction and tests covering missing Cupy.
- Added Prometheus rules GraphwaveP95Slow and IngestQueueHigh with tests.
- Implemented OpenTelemetry tracing with FastAPI and aio-pika instrumentation.
- Integrated pytest-cov with coverage gating in CI.
- Added setup.py with GPU extras and updated CI to install them only on GPU jobs.
- Enforced PEP 257 docstrings and minimum 80% coverage via pre-commit checks.
- Integrated flake8-bandit security lints and radon complexity guard; added
  complexity badge to README.
- Introduced Hypothesis property tests for GraphWave and Möbius associativity.
- Split CI into unit, scheduled heavy-nightly and GPU jobs.
- Documented Swagger usage for /explain/{node} with curl and fetch snippets; added
  tests checking presence in OpenAPI docs.
- Converted Grafana dashboards to Jsonnet and added jsonnetfmt pre-commit check.
- Added mkdocs configuration with CPU/GPU quick-start guides and Pages deployment workflow.
- Enabled FAISS multi-probing with configurable n_subprobe and unit tests verifying nprobe.
- Integrated bitsandbytes int8 matmul in Whisper parser with tests.
- Implemented streaming GPU heat kernel with block parameter and tests.
- Added helpers to estimate streaming memory and choose block size, with unit
  tests ensuring 5 GB usage for 10M nodes.
- Enabled mypy strict checks with a new pre-commit hook and added type hints to parsers and telemetry.
- Created create_issues.py to auto-generate GitHub epic and issues; added unit test.
- Added bench_ann_cpu.py and bench_whisper_xrt.py with tests producing benchmark files.
- Installed missing dependencies and validated coverage run locally.
- Fixed formatting issues flagged by pre-commit and installed runtime dependencies to execute unit tests.
- Updated CI to push Docker images only on main branch.
