import yaml


def test_jitter_alert_rule():
    with open("configs/alerts.yaml") as fh:
        data = yaml.safe_load(fh)
    rules = {r["alert"]: r for r in data["groups"][0]["rules"]}
    jitter = rules.get("GPJitterRateHigh")
    assert jitter is not None
    assert jitter["expr"] == "rate(gp_jitter_restarts_total[10m]) > 0.01"
    assert jitter["for"] == "10m"
    assert jitter["labels"]["severity"] == "warning"
