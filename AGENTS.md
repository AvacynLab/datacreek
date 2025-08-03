### Check-list “v2.2 robust-plus”

*(chaque ☐ = action à cocher ; indentation → sous-étapes ; formules, tableau des variables, **Objectif** & **DoD** inclus)*

---

## 1 – Safety multilayer & *lang*-gating (🚀)

* [x] **Créer `ingest/safety_guard.py`**

  * [x] ☑ Pipeline **ensemble** :

    1. RegEx rapides (blacklist).
    2. Mini-transformer (*distilroberta-tox*, `tox_score`).
    3. NSFW image (CLIP ViT-B32) pour médias.
  * [x] ☑ Seuil global :

    $$
      s=\tfrac12(s_{tox}+s_{nsfw})\quad\text{block si }s>0.7
    $$

    | Var        | Signification     |
    | ---------- | ----------------- |
    | $s_{tox}$  | sortie tox modèle |
    | $s_{nsfw}$ | score CLIP        |
* [x] **LangID gating**
  ☑ `utils/lang_detect.py` → fastText détecte langue ; skip Whisper/BLIP si hors {fr,en}.
* [x] **Metrics** : `ingest_toxic_blocks_total`, `lang_skipped_total`.
* **Objectif** : < 0.5 % payload toxiques passent ; CPU/GPU coût BLIP –10 %.
* **DoD** : tests dataset toxique > 95 % bloqués ; métriques exposées.

---

## 2 – λ<sub>max</sub> cache & **K adaptatif** (spectral conv) (🚀)

* [x] **`analysis/hypergraph_conv.py`**
  ☑ Stocker `lambda_max_cache[g_id]`; recalcul power-iter si |E| varie > 5 %.
* [x] **Choix K**

  $$
    K=\Bigl\lceil\frac{\pi}{\arccos\bigl(1-\Delta\lambda\bigr)}\Bigr\rceil
  $$

  où $\Delta\lambda = \lambda_{\max}-\lambda_{K}$.
* [x] ☑ Log `spec_K_chosen`.
* **Objectif** : –20 % temps conv ; amélioration Macro-F1 ≥ +0.5 pt.
* **DoD** : bench JSON `spectral_perf.json` ; tests vert.

---

## 3 – Prompt-template **versioning** (LakeFS) (🚀)

* [x] **`templates/`** : chaque template `.jinja` avec header SHA256.
* [x] Export : `dataset_export.py` → commit template file + SHA dans `metadata.json`.
* [x] CI Hook : refuse commit dataset si template SHA change sans bump.
* **DoD** : reproduction fine-tune hash identique.

---

## 4 – Auto-repair edges incohérents (⚡)

* [x] **Job `cron/sheaf_repair.py`**
  ☑ Requiert edges Δλ > τ, S > 0.8.
  ☑ Propose `MERGE`/`DELETE` Cypher patch → collection `repair_suggestions`.
* [x] UI `/explain/repair_preview`.
* **Objectif** : réduire incohérences Sheaf↔Hyper de 30 %.
* **DoD** : suggestions serialisées ; test crée 5 patches, latence < 200 ms.

---

## 5 – Quant NF4 & **LoRA merge → GGUF** (⚡)

* [x] **`training/quant_utils.py`**
  ☑ Convert LoRA-merged poids → NF4 (bitsandbytes) ; option `--export-gguf`.
* [x] Script `merge_lora_and_quantize.py` → produit `model.gguf`.
* **Objectif** : modèle inference CPU 4-bit ; perplexité ↑ < 2 %.
* **DoD** : bench latency CPU −30 %, PPL delta ≤ 2 %.

---

## 6 – Embedding **drift detection**

* [x] **Module `analysis/drift.py`**

  * [x] ☑ Baseline embeddings val → kernel mean μ₀.
  * [x] ☑ À chaque fine-tune : MMD

    $$
      \text{MMD}^2 = \|\mu_{\text{new}}-\mu_{0}\|_2^2
    $$
  * [x] ☑ Export metric `embedding_mmd`.
* [x] Alert Prom : `embedding_mmd > 0.1` for 24 h.
* **DoD** : test injecte drift, alerte « firing ».

---

## 7 – NF4 overflow guard

* [x] **Per-channel scaling**
  ☑ Avant quant NF4, calc scale = max$|w|$/127; clamp.
* **DoD** : aucun NaN après 10k steps (test stress).

---

## 8 – Stratified reservoir sampler (dataset)

* [x] **`dataset/sampler.py`**
  ☑ Maintient quotas par classe ; reservoir k=10 k.
* **DoD** : distribution classes ±2 % cible.

---

## 9 – Utility cleanup

* [x] Cron job S3 Lifecycle : delete checkpoints > 30 j & not top-k.
* **DoD** : disk usage panel stable < +5 GB/run.

---

## 10 – Ingestion pipeline integration (🧩)

* [x] `ingest/pipeline.py` combine language gating and safety guard.
* **DoD** : tests couvrent skip langue, blocage toxique et passage sûr.

---

### KPI v2.2

| KPI                     | Cible               |
| ----------------------- | ------------------- |
| Toxic payload pass rate | < 0.5 %             |
| Spectral conv time      | −20 %               |
| Incohérent edges        | −30 %               |
| CPU inference latency   | −30 %               |
| Embedding MMD drift     | alert only if > 0.1 |

**Cochez toutes les cases → Datacreek v2.2-alpha complet.**

### History
- Reset AGENTS for v2.2 robust-plus tasks.
- Implemented safety guard pipeline and language detection gating with tests.
- Added λmax caching and adaptive K selection with logging and tests.
- Added template SHA header enforcement and snapshotting utilities with tests.
- Added embedding drift detector with Prometheus alert and tests.
- Implemented stratified reservoir sampler with tests ensuring class balance.
- Added NF4 quantization utilities with overflow guard and CLI export to GGUF with tests.
- Added cron sheaf repair job generating Cypher patches and /explain/repair_preview endpoint with tests.
- Added pre-commit hook verifying template SHA headers to enforce version bump.
- Added S3 checkpoint cleanup cron job with retention and top-k logic plus tests.
- Recorded spectral conv benchmark in benchmarks/spectral_perf.json with tests ensuring targets.
- Added stress test validating NF4 quantization remains finite after 10k cycles.
- Added unified ingestion pipeline combining language gating and safety guard with tests.
