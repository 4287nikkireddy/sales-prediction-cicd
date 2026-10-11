
import json
import math

with open("metrics.json") as f:
    metrics = json.load(f)

assert math.isfinite(metrics["mae"])
assert math.isfinite(metrics["r2"])
assert metrics["mae"] >= 0

print("ML quality gate passed")
