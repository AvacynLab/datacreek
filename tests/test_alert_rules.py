import yaml


def test_jitter_alert_rule():
    with open("configs/alerts.yaml") as fh:
        data = yaml.safe_load(fh)
    rules = {r["alert"]: r for r in data["groups"][0]["rules"]}
    jitter = rules.get("SVGPJitterStorm")
    assert jitter is not None
    assert jitter["expr"] == "rate(gp_jitter_restarts_total[10m]) > 0.01"
    assert jitter["for"] == "10m"
    assert jitter["labels"]["severity"] == "warning"

    redis_rule = rules.get("RedisLowHitRatio")
    assert redis_rule is not None
    assert redis_rule["expr"] == "redis_hit_ratio < 0.3"
    assert redis_rule["for"] == "5m"
    assert redis_rule["labels"]["severity"] == "warning"

    eigsh_rule = rules.get("EigshTimeoutCritical")
    assert eigsh_rule is not None
    assert eigsh_rule["expr"] == "increase(eigsh_timeouts_total[1h]) > 2"
    assert eigsh_rule["for"] == "1m"
    assert eigsh_rule["labels"]["severity"] == "critical"
