"""A train-only linear preference over interpretable same-task query differences."""
from functools import lru_cache
import json
from pathlib import Path
import re

from sqlglot import exp

from .obligations import Request, signals
from .semantic import claims
from .structure import header_agreement, local, parse, parts, sources


def features(sql, question, schema, example, ordered):
    request = Request.read(question)
    tree = parse(sql)
    if tree is None:
        return {}
    selected = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if selected is None:
        return {}
    values = {"header": header_agreement(selected, example["columns"]),
              "joins": min(5, len(list(tree.find_all(exp.Join)))) / 5,
              "subqueries": min(3, len(list(tree.find_all(exp.Subquery)))) / 3,
              "distinct": int(bool(selected.args.get("distinct"))),
              "group": int(bool(selected.args.get("group"))),
              "limit": int(bool(selected.args.get("limit"))),
              "set": int(isinstance(tree, exp.SetOperation)),
              "ordered_output": int(bool(ordered and selected.args.get("order")))}
    for tag, claim in claims(sql, request, schema).items():
        group = tag.split(":", 1)[0]
        values[f"claim:{group}"] = values.get(f"claim:{group}", 0) + claim.state
    values.update({f"signal:{name}": value for name, value in signals(sql, request, example, schema).items()})
    text = question.casefold()
    literal_values = [literal.this for literal in tree.find_all(exp.Literal) if literal.is_string]
    values["unsupported_literals"] = min(3, sum(len(value) >= 3 and value.strip("%").casefold() not in text
                                               for value in literal_values)) / 3
    values["literal_exact_case"] = sum(value in question for value in literal_values) / max(1, len(literal_values))
    values["literal_lower_case"] = sum(value == value.casefold() for value in literal_values) / max(1, len(literal_values))
    group = selected.args.get("group")
    values["group_width"] = min(4, len(group.expressions)) / 4 if group else 0
    sorts = selected.args.get("order")
    count_sort = bool(sorts and sorts.expressions[0].find(exp.Count))
    rank = bool(re.search(r"\b(?:most|least|fewest|largest|smallest|highest|lowest)\b", text))
    values["count_rank"] = int(count_sort and rank)
    values["group_without_aggregate"] = int(bool(group and not tree.find(exp.AggFunc) and not rank))
    values["limit_unrequested"] = int(bool(selected.args.get("limit") and not rank and not re.search(r"\b(?:first|last|top|bottom)\b", text)))
    aliases = {table.alias_or_name.casefold(): table.name.casefold() for table in sources(selected)}
    used = {field.table.casefold() for field in local(selected, exp.Column) if field.find_ancestor(exp.Join) is None}
    values["unused_relations"] = min(3, sum(alias not in used for alias in aliases)) / 3
    values["foreign_key_defects"] = min(3, -signals(sql, request, example, schema).get("relationship_key", 0)) / 3
    return values


@lru_cache(maxsize=1)
def model():
    path = Path(__file__).resolve().parent.parent / "pairwise_policy.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"weights": {}, "margin": 1.}


def score(vector, weights):
    return sum(weights.get(name, 0) * value for name, value in vector.items())


def admissible(proposed, previous):
    explicit = {"signal:aligned_projection", "signal:quoted_filter", "signal:numeric_boundary",
                "signal:explicit_order", "signal:explicit_limit", "signal:explicit_distinct"}
    if any(proposed.get(name, 0) < value for name, value in previous.items() if name in explicit):
        return False
    obligations = {"claim:entity_name", "claim:independent_membership", "claim:entity_absence"}
    return all(proposed.get(name, 0) >= value for name, value in previous.items() if name in obligations and value > 0)


def choose(agent, old):
    config = model()
    weights = config["weights"]
    incumbent = features(old["sql"], agent.question_text, agent.schema, agent.example, agent.ordered)
    baseline_score = score(incumbent, weights)
    matching = [branch for branch in agent.branches if branch.feedback.get("matches_example")]
    if len(matching) < 2:
        return old
    candidates = [(branch, features(branch.sql, agent.question_text, agent.schema, agent.example, agent.ordered))
                  for branch in matching]
    candidates = [(branch, vector) for branch, vector in candidates if admissible(vector, incumbent)]
    top, new = max(candidates, key=lambda item: score(item[1], weights))
    threshold = agent.config.get("pairwise_margin", config["margin"])
    if score(new, weights) - baseline_score > threshold:
        return {"tool": "submit", "sql": top.sql}
    return old
