"""Observe v4's decisions and optionally change its next-check policy."""
import json
import os
from pathlib import Path
from threading import Lock

from sqlglot import exp

from .controller import Move
from .coverage_edits import propose
from .planning import PlanningAgent
from .semantic_agent import SemanticAgent
from .scheduling import select
from .structure import header_agreement, invalid_bindings, key, parse, sources, sql_text


AUDIT_LOCK = Lock()
NEW_EDITS = {"grounded_clause", "relation_scope"}


def describe(sql):
    tree = parse(key(sql))
    if tree is None:
        return {"syntax": "unparsed"}
    result = {"set_operation": tree.key, "relations": tuple(t.name.casefold() for t in sources(tree))}
    for name in ("joins", "expressions", "where", "group", "having", "distinct", "order", "limit", "offset", "with_"):
        items = tree.args.get(name)
        if isinstance(items, list) and items:
            result[name] = tuple(item.sql(normalize=True, comments=False) for item in items)
        elif isinstance(items, exp.Expression):
            result[name] = items.sql(normalize=True, comments=False)
        elif items:
            result[name] = items
    if tree.key in {"intersect", "except", "union"}:
        result["operands"] = tree.sql(normalize=True, comments=False)
    return result


def difference(before, after):
    return sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))


class CoverageAgent(SemanticAgent):
    def __init__(self, *args):
        super().__init__(*args)
        audit = os.environ.get("MP3_V7_AUDIT")
        self.audit = Path(audit) if audit else None
        self.steps = []
        self.lineage = {}
        self.shapes = {}

    def shape(self, sql):
        if sql not in self.shapes:
            self.shapes[sql] = describe(sql)
        return self.shapes[sql]

    def root(self, move):
        return move.sql if move.kind == "hint" else self.lineage.get(move.parent.sql, move.parent.sql)

    def offer(self, observation):
        options = super().options(observation)
        if (not self.config.get("coverage_generation") or self.schema is None
                or len(self.branches) < self.config.get("generation_min_checks", 2)):
            return options
        matches = any(branch.feedback.get("matches_example") for branch in self.branches)
        if matches and (self.example["row_count"] or len(self.branches) < 3
                        or not self.config.get("generation_on_empty_match", True)):
            return options
        used = {branch.origin for branch in self.branches} & NEW_EDITS
        if used == NEW_EDITS:
            return options
        active = [self.best(), self.branches[-1]]
        if matches:
            selected = PlanningAgent.submit(self)["sql"]
            active.insert(0, next(branch for branch in self.branches if branch.sql == selected))
        known = self.checked | {key(move.sql) for move in options}
        proposed = {}
        for branch in active:
            tree = parse(branch.sql)
            if len(branch.path) >= 4:
                continue
            for edit in propose(tree, self.schema, self.intent):
                if edit.kind in used or edit.kind in proposed:
                    continue
                sql = sql_text(edit.tree)
                canonical = key(sql)
                if canonical in known or parse(sql) is None or invalid_bindings(edit.tree, self.schema):
                    continue
                gain = 0.
                if isinstance(tree, exp.Select) and isinstance(edit.tree, exp.Select):
                    headers = self.example["columns"]
                    gain = header_agreement(edit.tree, headers) - header_agreement(tree, headers)
                if gain < -.10:
                    continue
                proposed[edit.kind] = Move(edit.kind, sql, branch, edit.confidence, edit.reason,
                                           diagnosis=self.diagnose(branch), projection_gain=gain)
                known.add(canonical)
        return options + list(proposed.values())

    def priority(self, move, observation):
        if move.kind in NEW_EDITS:
            matched = any(branch.feedback.get("matches_example") for branch in self.branches)
            if move.confidence >= .94:
                return 1.04 + .20 * move.projection_gain
            return (.76 if matched else .92) + .40 * (move.confidence - .86) + .20 * move.projection_gain
        return super().priority(move, observation)

    def choose(self, options, observation):
        return select(self, options, observation)

    def record(self, observation, action, options, chosen):
        if not self.audit:
            return
        ranked = []
        for move in options:
            ranked.append({"sql": move.sql, "source": "official_hint" if move.kind == "hint" else "rewrite",
                           "family": move.kind, "parent": move.parent.sql, "root": self.root(move),
                           "difference": difference(self.shape(move.parent.sql), self.shape(move.sql)),
                           "priority": self.priority(move, observation), "selected": move is chosen,
                           "tool": "execute" if move.kind in {"probe", "value_probe"} else "check"})
        index = len(observation["history"])
        self.steps.append({"at": index, "remaining": observation["remaining"], "action": action, "options": ranked})
        if action["tool"] == "submit":
            updated = []
            for step in self.steps:
                i = step["at"]
                feedback = observation["history"][i]["observation"] if i < index else None
                updated.append(step | {"feedback": feedback})
            record = {"id": observation["task"]["id"], "steps": updated}
            with AUDIT_LOCK, self.audit.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def act(self, observation):
        action = self.start(observation)
        options, chosen = [], None
        if action is None:
            options = self.offer(observation)
            chosen = self.choose(options, observation) if options else None
            if chosen is not None:
                action = self.apply(chosen)
                if action["tool"] == "check":
                    self.lineage[chosen.sql] = self.root(chosen)
            else:
                action = self.submit()
        self.record(observation, action, options, chosen)
        return action
