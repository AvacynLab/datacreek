### Check-list détaillée : **Pipeline d’entraînement adaptative (Unsloth + TRL) – v2.0**

> Toutes les cases ☐ doivent être cochées pour déclarer la pipeline « production-ready ».
> Chaque tâche principale inclut : sous-étapes (indentation), formules mathématiques si utile, tableau des variables, **Objectif chiffré** & **Definition of Done (DoD)**.
> Cette liste tient compte du code actuel (v1.4-alpha) : aucune brique Unsloth/TRL n’existe encore ; les tâches ci-dessous couvrent intégralement les ajouts pertinents mentionnés dans le plan précédent.

---

## 1 – Infrastructure Unsloth : chargement & PEFT

* [ ] **Créer module `training/unsloth_loader.py`**

  * [x] Fonction `load_model(model_id, bits=4, max_seq=8192)`
    ☑ utilise `FastLanguageModel.from_pretrained()` (Unsloth).
  * [x] Fonction `add_lora(model, r, alpha, target_modules)`
    ☑ appelle `FastLanguageModel.get_peft_model()`.
* **Objectifs** : VRAM ↘ 70 %, vitesse × 1.8 vs HF vanilla.
* **DoD** : benchmark `bench_unsloth.json` stocké ; test unit vérifie VRAM < 0.6 × baseline.

---

## 2 – Détection de **tâche** & formatage dataset

* [x] **`training/task_detector.py`**
  ☑ Inspecte métadonnées dataset (`task` déjà taguée) ou infère :
  • présence champ `answer` ⇒ *QA* ;
  • label unique ⇒ *classification* ;
  • absence label ⇒ *génération* ;
  • champs `chosen/rejected` ⇒ *preference RLHF*.
* [x] **Format builders** `format_sft`, `format_classif`, `format_rlhf`
  ☑ génèrent prompt + template + EOS.
* **DoD** : 100 % jeux tests mappés correctement (`tests/unit/test_task_detector.py`).

---

## 3 – Sélection dynamique de **Trainer TRL**

| Tâche            | Trainer TRL                | Condition                      |
| ---------------- | -------------------------- | ------------------------------ |
| SFT / génération | `SFTTrainer`               | task == "generation"           |
| Classification   | `SFTTrainer` with `labels` | task == "classification"       |
| RLHF (PPO)       | `PPOTrainer`               | task == "rlhf_ppo"            |
| RLHF (DPO)       | `DPOTrainer`               | dataset with `chosen/rejected` |

* [x] **`training/trainer_factory.py`**
  ☑ construit le trainer avec **kwargs** (batch, lr, PEFT).
* **DoD** : factory unit-test retourne la bonne classe pour 4 types.

---

## 4 – Curriculum learning basé hypergraph

* [x] **Difficulté** $d$ d’un échantillon :

  $$
    d = \gamma\,h + \delta\,l + \eta\,c,
    \quad h=\text{\# hops},\; l=\text{longueur prompt},\; c=\text{centralité cible}
  $$

  | Var      | Poids default |
  | -------- | ------------- |
  | $\gamma$ | 0.5           |
  | $\delta$ | 0.3           |
  | $\eta$   | 0.2           |
* [x] Ordonnancer dataloader : batches du plus simple → plus complexe.
* **Objectif** : perplexité val – 10 % aux 3 premières époques.
* **DoD** : `training/curriculum_dataloader.py` tri fonctionne ; test décroissance perplexité.

---

## 5 – Active-learning & augmentation ciblée

* [x] **Erreur top-k** (samples avec perte > p95)
  ☑ Générer **k variantes** via synonymes / paraphrases du hypergraph.
* [x] Ré-injecter dans pool d’entraînement toutes les 2 époques.
* **DoD** : module `training/augmenter.py` ; test montre rappel val +1 pt.

---

## 6 – Auto-feedback RLHF (graphe → reward)

* [x] **Extraction triplets** de la réponse (regex + NLP).
* [x] Vérification contre hypergraph :

  $$
    R = \frac{\#\text{triplets vérifiés}}{\#\text{triplets totaux}}
  $$
* [x] Plug reward $R$ dans `ppo_config.reward_fn`.
* **Objectif** : hallucination rate ↓ 30 %.
* **DoD** : `tests/integration/test_auto_feedback.py` passe ; metric Prom `reward_avg` > 0.7.

---

## 7 – Surveillance & monitoring

* [x] **Logging** : intégration Weights & Biases (`wandb.init`) via callback.
* [x] **Prometheus** : exporter

  * `training_loss`,
  * `val_metric`,
  * `gpu_vram_bytes`,
  * `reward_avg`.
* [x] **Early stopping** callback (patience = 3 evals).
* **DoD** : dashboard Grafana « Training » OK ; alert `gpu_vram_bytes > cap`.

---

## 8 – CLI & configuration

* [x] `cli/train.py`
  ☑ arguments : `--model`, `--task`, `--dataset-path`, `--trainer`, `--alpha`, `--beta`, `--bits`, `--epochs`.
  ☑ charge YAML config éventuelle.
* **DoD** : `train.py --help` affiche options ; exécution end-to-end unit (small dataset).

---

## 9 – Tests et benchmarks

* [x] **Unit** : loader, detector, factory, curriculum.
* [x] **Heavy** : run 1 époque sur TinyStories (génération) et DBPedia (classif).
* [x] **Benchmark** : temps/VRAM vs HF baseline.
* **DoD** : CI passe CPU & GPU ; speed β≥1.7, VRAM ≤0.6×.

---

## 10 – Documentation

* [x] `docs/train_pipeline.md` : schéma, exemples.
* [x] README : ajout section “Fine-tune with Unsloth + TRL”.
* **DoD** : markdown-lint vert.

---

### KPI finaux v2.0

| KPI                    | Cible                   |
| ---------------------- | ----------------------- |
| VRAM vs HF baseline    | ≤ 60 %                  |
| Speedup vs HF baseline | ≥ 1.7×                  |
| Hallucination rate QA  | –30 %                   |
| PPL val (curriculum)   | –10 % premières époques |
| ARI / accuracy task    | +2 pts vs v1.4          |

**Toute la grille cochée → pipeline d’entraînement adaptative opérationnelle.**

### History
- Reset AGENTS for v2.0 tasks.
- Implemented `training/unsloth_loader.py` with load and LoRA helpers; added unit tests.
- Added `training/task_detector.py` with format builders and comprehensive unit tests.
- Added `training/trainer_factory.py` selecting TRL trainers dynamically with unit tests.
- Implemented curriculum dataloader with difficulty-based ordering and tests.
- Added `training/augmenter.py` performing active-learning augmentation with unit tests.
- Implemented `training/auto_feedback.py` with triplet extraction and graph-based reward, plus integration tests.
- Added monitoring utilities with W&B initialization, Prometheus gauges, and an early stopping helper alongside unit tests.
- Introduced `cli/train.py` parsing arguments and YAML configs with unit tests covering help output and config-driven run.
- Documented the adaptive training pipeline and added a README section on fine-tuning; ran unit tests for core modules.
- Added `benchmarks/bench_unsloth.json` capturing VRAM and speedup; implemented unit tests asserting targets.
- Ran one-epoch CLI smoke tests on TinyStories and DBPedia datasets and recorded benchmark durations.
- Verified checklist implementation; pre-commit and targeted pytest suite pass.
