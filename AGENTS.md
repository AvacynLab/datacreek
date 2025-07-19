# Agent Tasks - v1.1-scale-out

## Backlog

1. Bench overshoot & robustesse fp16 du re-centre Poincaré
   - [x] Tracer overshoot Δr vs norme hyperbolique
   - [x] Vérifier bijection exp/log tolérance fp32/fp16
   - [x] Clamp de sécurité après opérations
   - [x] **DoD**: médiane |Δr| -15% vs scaling

2. Whisper.cpp fallback CPU + métrique xRT
   - [x] Détection device
   - [x] Réduction auto batch_size CPU
   - [x] Gauge Prometheus `whisper_xrt{device}`
   - [x] Tests CPU & GPU xRT
   - [x] **DoD**: GPU <=0.5 xRT, CPU <=2

3. Node2Vec timeout & persistance params
   - [x] Simuler max_minutes via timer patch
   - [x] Sauver best_pq.json avec hash dataset
   - [x] **DoD**: fichier présent

4. PID TTL Redis configuration
   - [x] Bloc YAML avec paramètres PID
   - [x] Anti-windup clamp intégral
   - [x] Boucle discrète TTL clamp [1s,24h]
   - [x] Test overshoot <5%
   - [x] **DoD**: hit_ratio stable ±5%

5. Migration HAA Flyway
   - [x] Insérer 2025-07-haa_index.cypher dans config
   - [x] CI dry-run sur DB éphémère
   - [x] Snapshot checksum avant/après
   - [x] **DoD**: rollback auto

6. TTL manager fuite tâche
   - [x] Test redis_pid_leak.py
   - [x] **DoD**: pas de tâche pendante

7. Budget ε DP
   - [x] Simuler dépassement, attendre 403 header restant=0
   - [x] Log audit JSON
   - [x] **DoD**: log présent

8. Back-pressure ingestion breaker
   - [x] CircuitBreaker autour write_batch
   - [x] HTTP 429 quand OPEN
   - [x] Métrique breaker_state
   - [x] Test burst drop <1% + transition OPEN->HALF_OPEN

9. GraphWave CUDA précision
   - [x] Comparer diffusion CPU vs GPU L2 <1e-5
   - [x] @pytest.mark.gpu et skip si cupy absent

10. TPL incrémental bench
   - [x] Bench ΔE={1%,10%,20%} speedup>=2x
   - [x] Fail CI si ratio <2x

11. Endpoint /explain/{node}
   - [x] Router FastAPI JSON+SVG, auth, CORS
   - [x] Demo Observable Plot
   - [x] Tests snapshot

12. Hybrid ANN rerank PQ
   - [x] Implémenter rerank_pq avec FAISS GPU fallback
   - [x] Bench recall@100 >=0.92 latence P95<20ms
   - [x] **DoD**: bench JSON enregistré

13. Plugin pgvector
   - [x] Docker PostgreSQL+pgvector in CI
   - [x] Index ivfflat, latence<30ms recall>=0.9 vs FAISS
   - [x] Gauge pgvector_query_ms

14. CI multi-environnements
   - [x] requirements-ci.txt with fakeredis cupy-cuda12x faiss-cpu
   - [x] Matrix gpu true/false, heavy job 45min
   - [x] Promtool test rules

15. Import dashboard Grafana cache/TTL
   - [x] Script upload_dashboard.py via HTTP API
   - [x] Vérifier panels lmdb_evictions_total & ingest_queue_fill_ratio

## History
- Initial backlog imported and dependencies installed for heavy tests.
- Added fallback stub for missing pydantic to ensure heavy tests import modules.
- Ran black formatting on pipelines and config after pre-commit failure.
- Ran black on utils after pre-commit failures.
- Made heavy imports optional (requests, redis, networkx, numpy, rich) and added
  missing deps to requirements-ci.txt.
- Installed heavy dependencies and verified heavy tests start running.
