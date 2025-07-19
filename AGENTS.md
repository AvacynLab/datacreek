# Agent Tasks - v1.1-scale-out

## Backlog

1. Bench overshoot & robustesse fp16 du re-centre Poincaré
   - [ ] Tracer overshoot Δr vs norme hyperbolique
   - [ ] Vérifier bijection exp/log tolérance fp32/fp16
   - [ ] Clamp de sécurité après opérations
   - [ ] **DoD**: médiane |Δr| -15% vs scaling

2. Whisper.cpp fallback CPU + métrique xRT
   - [ ] Détection device
   - [ ] Réduction auto batch_size CPU
   - [ ] Gauge Prometheus `whisper_xrt{device}`
   - [ ] Tests CPU & GPU xRT
   - [ ] **DoD**: GPU <=0.5 xRT, CPU <=2

3. Node2Vec timeout & persistance params
   - [ ] Simuler max_minutes via timer patch
   - [ ] Sauver best_pq.json avec hash dataset
   - [ ] **DoD**: fichier présent

4. PID TTL Redis configuration
   - [ ] Bloc YAML avec paramètres PID
   - [ ] Anti-windup clamp intégral
   - [ ] Boucle discrète TTL clamp [1s,24h]
   - [ ] Test overshoot <5%
   - [ ] **DoD**: hit_ratio stable ±5%

5. Migration HAA Flyway
   - [ ] Insérer 2025-07-haa_index.cypher dans config
   - [ ] CI dry-run sur DB éphémère
   - [ ] Snapshot checksum avant/après
   - [ ] **DoD**: rollback auto

6. TTL manager fuite tâche
   - [ ] Test redis_pid_leak.py
   - [ ] **DoD**: pas de tâche pendante

7. Budget ε DP
   - [ ] Simuler dépassement, attendre 403 header restant=0
   - [ ] Log audit JSON
   - [ ] **DoD**: log présent

8. Back-pressure ingestion breaker
   - [ ] CircuitBreaker autour write_batch
   - [ ] HTTP 429 quand OPEN
   - [ ] Métrique breaker_state
   - [ ] Test burst drop <1% + transition OPEN->HALF_OPEN

9. GraphWave CUDA précision
   - [ ] Comparer diffusion CPU vs GPU L2 <1e-5
   - [ ] @pytest.mark.gpu et skip si cupy absent

10. TPL incrémental bench
   - [ ] Bench ΔE={1%,10%,20%} speedup>=2x
   - [ ] Fail CI si ratio <2x

11. Endpoint /explain/{node}
   - [ ] Router FastAPI JSON+SVG, auth, CORS
   - [ ] Demo Observable Plot
   - [ ] Tests snapshot

12. Hybrid ANN rerank PQ
   - [ ] Implémenter rerank_pq avec FAISS GPU fallback
   - [ ] Bench recall@100 >=0.92 latence P95<20ms
   - [ ] **DoD**: bench JSON enregistré

13. Plugin pgvector
   - [ ] Docker PostgreSQL+pgvector in CI
   - [ ] Index ivfflat, latence<30ms recall>=0.9 vs FAISS
   - [ ] Gauge pgvector_query_ms

14. CI multi-environnements
   - [ ] requirements-ci.txt with fakeredis cupy-cuda12x faiss-cpu
   - [ ] Matrix gpu true/false, heavy job 45min
   - [ ] Promtool test rules

15. Import dashboard Grafana cache/TTL
   - [ ] Script upload_dashboard.py via HTTP API
   - [ ] Vérifier panels lmdb_evictions_total & ingest_queue_fill_ratio

## History
- Initial backlog imported and dependencies installed for heavy tests.
