from __future__ import annotations

import json
from pathlib import Path

from nominal_price_invariance.prompts import CONDITIONS
from nominal_price_invariance.scenarios import canonical_sha256, generate_scenarios, write_scenarios_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "design" / "scenarios.csv"
MANIFEST = ROOT / "data" / "design" / "manifest.json"

scenarios = generate_scenarios()
write_scenarios_csv(OUT, scenarios)
manifest = {
    "seed": 20261007,
    "n_scenarios": len(scenarios),
    "n_symmetric": sum(s.scenario_type == "symmetric" for s in scenarios),
    "n_asymmetric": sum(s.scenario_type == "asymmetric" for s in scenarios),
    "conditions": [c.condition_id for c in CONDITIONS],
    "n_decisions_per_model": len(scenarios) * len(CONDITIONS),
    "scenario_sha256": canonical_sha256(scenarios),
}
MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
