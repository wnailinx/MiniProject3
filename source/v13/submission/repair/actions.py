"""Features and JSON inference for the train-only action utility model."""
from sqlglot import exp

from .controller import tree_value
from .obligations import signals
from .plans import cues
from .progress import fraction_agree
from .structure import header_agreement, parse, sources


def policy_score(model, features):
    return model["bias"] + model["rate"] * sum(tree_value(tree, features) for tree in model["trees"])


def action_features(agent, move, observation):
    score = agent.features(move, observation)
    parent, proposed = parse(move.parent.sql), parse(move.sql)
    q = agent.request
    example = agent.example
    if isinstance(proposed, exp.Select):
        score["proposed_header"] = round(header_agreement(proposed, example["columns"]), 2)
        score["proposed_joins"] = min(3, len(sources(proposed)) - 1)
        if isinstance(parent, exp.Select):
            old = {t.name.casefold() for t in sources(parent)}
            new = {t.name.casefold() for t in sources(proposed)}
            score["changes_tables"] = int(old != new)
            score["removes_tables"] = min(2, len(old - new))
    score["has_subquery"] = int(bool(proposed and proposed.find(exp.Subquery)))
    score["has_set_relation"] = int(isinstance(proposed, exp.SetOperation))
    score["output_column_count"] = min(5, len(example["columns"]))
    score["candidate_agreement"] = round(fraction_agree(move.parent.feedback, example), 2)
    score["sample_is_null_scalar"] = int(example.get("rows") == [[None]])
    score["hint_branches_checked"] = sum(branch.origin == "hint" for branch in agent.branches)
    score["checked_public_matches"] = sum(bool(b.feedback.get("matches_example")) for b in agent.branches)
    score["continuation"] = int(bool(move.parent.path) and move.kind != "hint")
    score["parent_is_recent"] = int(move.parent is agent.branches[-1])
    for tag in cues(agent.intent.text):
        score[f"cue:{tag}"] = 1
    if q and proposed is not None:
        symptom = signals(move.sql, q, example, agent.schema)
        score.update({f"planned:{name}": value for name, value in symptom.items()})
    return score
