"""Official entry point for the CPU repair experiments."""
from functools import lru_cache
import json
from pathlib import Path

from repair.ranked_agent import RankedAgent


@lru_cache(maxsize=16)
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create_agent(checkpoint=None):
    base = Path(__file__).resolve().parent
    config = read(Path(checkpoint).resolve() if checkpoint else base / "config.json")
    return RankedAgent(config, *(read(base / name) for name in (
        "skills.json", "policy.json", "obligation_weights.json", "plan_priors.json", "followup_policy.json")))
