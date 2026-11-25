"""Prioritize distinct, grounded interpretations within the existing move menu."""
from collections import Counter
from functools import lru_cache
import re

from sqlglot import exp

from .plans import cues
from .obligations import signals
from .structure import key, parse


PROBES = {"probe", "value_probe"}
CORE = {"joins", "relations", "group", "having", "set_operation", "operands", "where", "expressions", "distinct"}
STRUCTURAL = CORE - {"where", "expressions", "distinct"}
REPAIRS = {"binding", "inner_binding", "relationship", "projection", "value_repair", "restore_filter", "probed_count"}


def plain_conjunction(expression):
    if isinstance(expression, exp.Paren):
        return plain_conjunction(expression.this)
    if isinstance(expression, exp.And):
        return plain_conjunction(expression.this) + plain_conjunction(expression.expression)
    return [expression]


@lru_cache(maxsize=4096)
def equivalent_key(sql):
    """Normalize conjunction order for deterministic column-to-literal predicates."""
    canonical = key(sql)
    tree = parse(canonical)
    if not isinstance(tree, exp.Select) or not tree.args.get("where"):
        return canonical
    leaves = plain_conjunction(tree.args["where"].this)
    if len(leaves) < 2 or not all(isinstance(leaf, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE))
                                  and isinstance(leaf.this, exp.Column) and isinstance(leaf.expression, exp.Literal)
                                  for leaf in leaves):
        return canonical
    copy = tree.copy()
    ordered = sorted(leaves, key=lambda leaf: leaf.sql(normalize=True, comments=False))
    copy.where(exp.and_(*ordered), append=False, copy=False)
    return key(copy.sql(dialect="sqlite", comments=False))


def different(left, right):
    return {name for name in CORE if left.get(name) != right.get(name)}


def requested_count_order(agent, sql):
    text = agent.intent.text
    if not re.search(r"\b(?:most|least|largest|smallest|greatest|highest|lowest|fewest)\s+(?:number\s+of|count\s+of)\b", text):
        return False
    query = parse(sql)
    if not isinstance(query, exp.Select) or not query.args.get("group") or not query.args.get("order"):
        return False
    first = query.args["order"].expressions[0]
    ascending = bool(re.search(r"\b(?:least|smallest|lowest|fewest)\s+(?:number|count)\b", text))
    return isinstance(first.this, exp.Count) and bool(first.args.get("desc")) != ascending


def select(agent, options, observation):
    rated = [(agent.priority(move, observation), move) for move in options]
    baseline_score, baseline = max(rated, key=lambda entry: entry[0])
    if not agent.config.get("diverse_schedule"):
        return baseline
    matched = [branch for branch in agent.branches if branch.feedback.get("matches_example")]
    if matched and agent.example["row_count"] or baseline.kind == "probed_count":
        return baseline

    already = {equivalent_key(branch.sql) for branch in agent.branches}
    unique = {}
    for score, move in rated:
        if move.kind in PROBES:
            candidate = ("probe", key(move.sql))
        else:
            candidate = ("check", equivalent_key(move.sql))
            if candidate[1] in already:
                continue
        if candidate not in unique or score > unique[candidate][0]:
            unique[candidate] = (score, move)
    if not unique:
        return None

    strength = agent.config.get("diversity_strength", .30)
    latest = agent.branches[-1]
    recent = agent.branches[-3:]
    shapes = [agent.shape(branch.sql) for branch in recent]
    roots = Counter(agent.lineage.get(branch.sql, branch.sql) for branch in recent if branch.path)
    families = Counter(branch.origin for branch in recent if branch.path)
    tags = cues(agent.intent.text)
    expected_rank = requested_count_order(agent, baseline.sql)
    baseline_evidence = signals(baseline.sql, agent.request, agent.example, agent.schema)

    def score(entry):
        value, move = entry
        if move.kind in PROBES or move.faults:
            return value
        proof = 0.
        if agent.config.get("explicit_schedule") and not matched and observation["remaining"] <= 4:
            candidate_evidence = signals(move.sql, agent.request, agent.example, agent.schema)
            protected = {"numeric_boundary", "quoted_filter", "explicit_limit", "explicit_order", "aligned_projection"}
            compared = protected & baseline_evidence.keys() & candidate_evidence.keys()
            deltas = [candidate_evidence[name] - baseline_evidence[name] for name in compared]
            if deltas and min(deltas) >= 0 and max(deltas) > 0:
                proof = .28
            if not expected_rank and requested_count_order(agent, move.sql):
                proof = max(proof, .62)
        value += proof
        if value < baseline_score - .42:
            return value
        if move.kind in REPAIRS and (move.diagnosis.error in {"column", "table"} or move.projection_gain >= .3):
            return value + strength * .38
        candidate = agent.shape(move.sql)
        contrast = min((different(shape, candidate) for shape in shapes),
                       key=lambda diff: (len(diff & STRUCTURAL), len(diff)))
        branch = agent.root(move)
        grounded = (move.kind == "hint" or move.confidence >= .78 or move.projection_gain > 0)
        changes_structure = bool(contrast & STRUCTURAL) or len(contrast) >= 2
        untried = not roots[branch]
        stalled = len(recent) >= 2 and all(b.feedback.get("matches_example", False) == latest.feedback.get("matches_example", False)
                                           and agent.diagnose(b).rows == agent.diagnose(latest).rows for b in recent[-2:])
        bonus = (.70 if grounded and changes_structure and untried else .28 if grounded and changes_structure else 0.)
        if changes_structure and (("group" in tags or "count" in tags) and contrast & {"group", "having"}
                                  or tags & {"both", "absence"} and contrast & {"set_operation", "operands"}):
            bonus += .28
        if stalled and not changes_structure:
            bonus -= .30 * min(2, roots[branch]) + .20 * min(2, families[move.kind])
        if not matched and move.parent is latest and latest.feedback.get("ok") and len(recent) >= 2:
            progressed = agent.evidence(latest) > agent.evidence(recent[-2]) + .18
            if progressed and move.confidence >= .75:
                bonus += .34
        if matched:
            # Empty outputs provide little feedback. Keep diversion narrow and prioritize an untried explanation.
            bonus = bonus * .55 if grounded and changes_structure else min(0., bonus)
        return value + strength * bonus

    return max(unique.values(), key=score)[1]
