### Check-list finale « v2.1 complétion »

*(chaque case ☐ = action à cocher ; sous-étapes imbriquées ; formules + tableau des variables ; **Objectif** + **DoD** en fin de bloc)*

---

## 1 – SNR **dynamique** (soft-gating adaptatif)

* [x] **Implémenter `quality_metrics.dynamic_snr_threshold()`**

  * [x] Collecter SNR des *N* derniers fichiers audio (deque métrique).
  * [x] Calculer moyenne $\overline{SNR}$ et écart-type $\sigma_{SNR}$.

    $$
      \sigma_{SNR} = \sqrt{\tfrac1N \sum_{i=1}^{N}(SNR_i-\overline{SNR})^2}
    $$

    | Var     | Description                       |
    | ------- | --------------------------------- |
    | $N$     | taille fenêtre (par défaut = 500) |
    | $SNR_i$ | SNR du fichier *i*                |
  * [x] Fixer seuil : `thr = 6 dB + 0.5·σ_{SNR}`.
* [x] **Brancher dans `audio_validator.py`**
  [x] Remplacer valeur constante par `dynamic_snr_threshold()`.
* [x] **Exporter métrique Prometheus** `snr_dynamic_thr`.
* [x] **Tests**
  [x] Jeu 100 clips silence/bruit → faux-positifs < 2 %.
* **Objectif** : réduire rejets audio valides de 8 % → < 2 %.
* **DoD** : test passe, métrique visible dans Grafana.

---

## 2 – Route **/explain/sheaf_diff** (arêtes incohérentes)

* [x] **Fonction `top_k_incoherent(k, τ)`** dans `analysis/sheaf_hyper_bridge.py`

  $$
    \Delta\lambda_i = |\lambda_i^{sheaf} - \lambda_i^{hyper}|
  $$

  | Var                 | Signification             |
  | ------------------- | ------------------------- |
  | $\lambda_i^{sheaf}$ | iᵉ valeur propre faisceau |
  | $\lambda_i^{hyper}$ | iᵉ valeur propre hypergr. |
* [x] Retourner liste des *k* arêtes où $\Delta\lambda_i > τ$.
* [x] **API FastAPI** `/explain/sheaf_diff?top=50`

  * [x] Prend params `top`, `tau` (τ).
  * [x] Renvoie JSON `{edge_id, delta_lambda}`.
  * [x] Latence testée < 100 ms (k = 50).
* **DoD** : route documentée dans Swagger, tests unit ok.

---

## 3 – Mapper **auto-tuning overlap**

* [x] **Ajouter `tune_overlap(points, overlaps)`** dans `analysis/mapper.py`

  * [x] Boucle sur overlaps ∈ {0.2, 0.3, 0.4, 0.5}.
  * [x] Calcule silhouette score $S(o)$ pour chaque.
  * [x] Sélectionne $o^* = \arg\max S(o)$.
* [x] Log métrique `mapper_overlap_opt`.
* [x] **Tests**
  [x] Silhouette avec tuning ≥ silhouette(0.3)+0.03.
* **DoD** : optimisation appelée par défaut ; metric exposée.

---

## 4 – Snapshot **tokenizer.json** dans LakeFS

* [x] **Modifier `utils/dataset_export.py`**

  * [x] `tokenizer.save_pretrained(tmp_dir)` ; copier `tokenizer.json`.
  * [x] `lakefs_client.upload_object` dans même commit que dataset.
  * [x] Écrire SHA du tokenizer dans `metadata.json`.
* [x] **Test de reproducibilité**
  [x] Recharger dataset + tokenizer commit X → même hash.
* **DoD** : fichier présent sur LakeFS UI, hash stable CI.

---

## 5 – Métrique Prometheus **training_eta_seconds**

* [x] **Dans `training/callbacks.py`**
  [x] Ajouter `Gauge("training_eta_seconds", "ETA to finish")`.

  $$
    ETA = \frac{steps_{tot} - steps_{done}}{steps/sec}
  $$

  | Var            | Description          |
  | -------------- | -------------------- |
  | $steps_{tot}$  | max\_steps           |
  | $steps_{done}$ | global\_step courant |
  | $steps/sec$    | moyenne glissante    |
* [x] Update gauge chaque `on_log`.
* [x] Dashboard Grafana barre de progression.
* **DoD** : widget affiche ETA, actualisé < 30 s.

---

### KPI de clôture v2.1

| Domaine                     | Cible          |
| --------------------------- | -------------- |
| Audio faux-positifs         | < 2 %          |
| Route /sheaf_diff latence   | < 100 ms       |
| Silhouette gain Mapper      | ≥ +0.03        |
| Reproduce split (tokenizer) | Hash identique |
| ETA gauge rafraîchissement  | ≤ 30 s         |

*Une fois ces cinq blocs cochés, la feuille de route v2.1 sera entièrement réalisée.*

### History
- Reset AGENTS for v2.1 completion tasks.
- Revalidated dynamic SNR threshold and Prometheus metric with tests.

- Completed sheaf_diff endpoint, Mapper overlap tuning, tokenizer snapshot, and training ETA gauge with targeted tests.
- Re-ran pre-commit and targeted unit tests after installing missing dependencies
  to verify all v2.1 tasks remain satisfied.
