# Agent Tasks Checklist

- [ ] Add `cleanup.k_min` to configs and code; ensure `gds_quality_check` uses this setting.
- [ ] Update `cleanup.hub_deg` default to 1000 in configuration.
- [ ] Write `fractal_dim` and `fractal_sigma` to Neo4j after running `bootstrap_db`.

## Notes
- Tests should cover new behaviour for saving fractal metrics with a driver.
- Remember to run `pytest -q` before committing.
