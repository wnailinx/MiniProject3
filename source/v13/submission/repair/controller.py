"""Incremental repair with bounded active branches and replayable official actions."""
from collections import Counter
from dataclasses import dataclass
import math
import re

from sqlglot import exp

from .skills import diagnostic_query, edits, literal_diagnostic, literal_from_probe
from .structure import Diagnosis, Intent, Schema, header_agreement, invalid_bindings, key, local, parse, parts, sources, sql_text


@dataclass
class Branch:
    sql: str
    origin: str
    feedback: dict
    path: tuple
    confidence: float
    support: int = 0


@dataclass
class Move:
    kind: str
    sql: str
    parent: Branch
    confidence: float
    reason: str
    support: int = 0
    diagnosis: Diagnosis | None = None
    faults: int = 0
    projection_gain: float = 0.
    probe_literal: str | None = None


def tree_value(tree, features):
    while isinstance(tree, list):
        name, threshold, lower, upper = tree
        tree = lower if features.get(name, 0.0) <= threshold else upper
    return tree


class Agent:
    def __init__(self, config, memory, policy):
        self.config = config
        self.memory = memory if config.get("memory", True) else {}
        self.policy = policy if config.get("learned", True) else {}
        self.branches = []
        self.hints = {}
        self.schema = None
        self.seen = 0
        self.checked = set()
        self.pending = None
        self.probe = None
        self.probed = False
        self.probe_values = None
        self.intent = None
        self.example = None
        self.ordered = False

    def ingest(self, observation):
        if self.intent is None:
            self.intent = Intent.read(observation["task"]["question"])
            self.example = observation["example"]
            self.ordered = observation["ordered"]
        for event in observation["history"][self.seen:]:
            action, feedback = event["tool"], event["observation"]
            if action == "check":
                move = self.pending
                branch = Branch(event["sql"], move.kind if move else "original", feedback,
                                move.parent.path + (move.kind,) if move else (),
                                move.confidence if move else 1., move.support if move else 0)
                self.branches.append(branch)
                self.checked.add(key(branch.sql))
                self.pending = None
            elif action == "repair_hint":
                for position, query in enumerate(feedback):
                    if not isinstance(query, str) or parse(query) is None:
                        continue
                    canonical = key(query)
                    if canonical not in self.hints:
                        self.hints[canonical] = (query, position, 0)
                    first, index, support = self.hints[canonical]
                    self.hints[canonical] = (first, index, support + 1)
            elif action == "schema":
                self.schema = Schema(feedback)
            elif action == "execute":
                self.probed = True
                rows = feedback.get("rows", [])
                literal = self.pending.probe_literal
                if literal is not None:
                    values = [row[0] for row in rows if len(row) == 1 and isinstance(row[0], str)
                              and row[0] != literal and len(row[0]) <= 100]
                    self.probe_values = (self.pending.parent, literal, tuple(values[:2]))
                elif len(rows) == 1 and len(rows[0]) == 2 and all(isinstance(v, int) for v in rows[0]):
                    self.probe = (self.pending.parent.sql, tuple(rows[0]))
                self.pending = None
        self.seen = len(observation["history"])

    def diagnose(self, branch):
        return Diagnosis.inspect(parse(branch.sql), branch.feedback, self.example)

    def reliability(self, kind, diagnosis):
        skills = self.memory.get("skills", {})
        general = skills.get(kind, {}).get("outcomes", {})
        context = skills.get(kind, {}).get("contexts", {}).get(
            diagnosis.error if diagnosis.error != "none" else diagnosis.projection, {})
        positive, total = general.get("improved", 0), general.get("attempts", 0)
        prior = (positive + 2) / (total + 12)
        return (context.get("improved", 0) + 12 * prior) / (context.get("attempts", 0) + 12)

    def risk(self, kind, diagnosis):
        summary = self.memory.get("skills", {}).get(kind, {}).get("matched_parent", {})
        subset = summary.get("empty" if diagnosis.empty else "nonempty", {})
        return (subset.get("regressed", 0) - subset.get("improved", 0)) / (subset.get("attempts", 0) + 20)

    def evidence(self, branch):
        tree = parse(branch.sql)
        intent = self.intent
        diagnosis = self.diagnose(branch)
        score = self.config.get("original_bias", .63) if branch.origin == "original" else .12 if branch.origin == "hint" else 0.
        score += .13 * min(4, branch.support)
        if not branch.feedback.get("matches_example"):
            expected = self.example["row_count"]
            actual = branch.feedback.get("row_count", 0)
            score -= 1.3 * abs(actual - expected) / max(expected, actual, 1)
            if branch.feedback.get("ok"):
                score += .50
        if tree is None:
            return score - 3
        if not isinstance(tree, exp.Select):
            if isinstance(tree, exp.Intersect) and re.search(r"\bboth\b", intent.text):
                score += .9
            if isinstance(tree, exp.Except) and intent.negative:
                score += .8
            tree = tree.find(exp.Select)
            if tree is None:
                return score
        score += self.config.get("header_weight", 3.6) * header_agreement(tree, self.example["columns"])
        projection_is_count = any(node.find(exp.Count) for node in tree.expressions)
        if intent.distinct and (tree.args.get("distinct") or tree.find(exp.Distinct)):
            score += .6
        elif tree.args.get("distinct"):
            score -= .20
        if intent.grouped and tree.args.get("group"):
            score += .45
        if intent.frequent and tree.args.get("group") and tree.args.get("order"):
            if tree.args["order"].find(exp.Count):
                score += 1.2
        if intent.count and projection_is_count:
            score += .25
        if intent.negative and (tree.find(exp.Not) or tree.find(exp.NEQ)):
            score += .45
        if intent.negative and re.search(r"\b(never|without|do not|does not|did not|don't)\b", intent.text):
            if tree.find(exp.Not) and any(isinstance(n.this, (exp.Exists, exp.In)) for n in tree.find_all(exp.Not)):
                score += .62
        if self.ordered and tree.args.get("order"):
            score += .55
        if intent.descending is not None and tree.args.get("order") and not intent.frequent:
            score += .30 if bool(tree.args["order"].expressions[0].args.get("desc")) == intent.descending else -.30
        if branch.origin in ("boundary", "contrast", "group", "literal", "predicate", "relationship",
                             "value_repair", "reshape") and branch.confidence >= .75:
            score += .36
        if self.schema is not None:
            scope = sources(tree)
            for table in scope:
                score += .06 * min(2, len(parts(table.name) & intent.words))
            # Question-irrelevant literals are often hallucinated filters, especially on empty examples.
            for value in local(tree, exp.Literal):
                if value.is_string and len(value.this) >= 3 and value.this.strip("%").casefold() not in intent.text:
                    score -= .25
        score -= .14 * len(tree.args.get("joins", [])) + .12 * max(0, len(branch.path) - 2)
        if self.memory and branch.origin not in {"original", "hint"}:
            score += .30 * self.reliability(branch.origin, diagnosis)
            if diagnosis.matched:
                score -= self.config.get("tie_risk_weight", 1.6) * self.risk(branch.origin, diagnosis)
        if branch.origin == "skeleton" and self.example["row_count"] == 0:
            score -= .5
        return score

    def best(self):
        return max(self.branches, key=lambda b: (bool(b.feedback.get("matches_example")), self.evidence(b)))

    def start(self, observation):
        self.ingest(observation)
        if not self.branches:
            return {"tool": "check", "sql": observation["issue_sql"]}
        if observation["remaining"] <= 0:
            return self.submit()
        if not any(event["tool"] == "repair_hint" for event in observation["history"]):
            return {"tool": "repair_hint"}
        if self.schema is None and observation["remaining"] > 1:
            return {"tool": "schema"}
        return None

    def submit(self):
        return {"tool": "submit", "sql": self.best().sql}

    def options(self, observation):
        by_identity = {}

        def add(move):
            canonical = key(move.sql)
            if canonical not in self.checked and canonical not in by_identity and parse(move.sql) is not None:
                move.diagnosis = self.diagnose(move.parent)
                tree = parse(move.sql)
                parent_tree = parse(move.parent.sql)
                if self.schema is not None:
                    move.faults = invalid_bindings(tree, self.schema)
                if isinstance(tree, exp.Select) and isinstance(parent_tree, exp.Select):
                    headers = self.example["columns"]
                    move.projection_gain = header_agreement(tree, headers) - header_agreement(parent_tree, headers)
                by_identity[canonical] = move

        original = self.branches[0]
        for identity, (sql, position, support) in self.hints.items():
            if identity not in self.checked:
                add(Move("hint", sql, original, .65 - position * .04, "switch to a distinct hint branch", support))
        if self.schema is None or not self.config.get("incremental", True):
            return list(by_identity.values())
        recent = self.branches[-1]
        best = self.best()
        # Active branches are chosen from checked history. A fresh modification
        # can only descend from a checked branch, so tool feedback gates composition.
        active = [recent]
        if best is not recent:
            active.append(best)
        if not any(branch.feedback.get("matches_example") for branch in self.branches) and original not in active:
            active.append(original)
        peer_trees = [parse(query) for query, _, _ in self.hints.values()]
        counts = Counter()
        for branch in active:
            query = parse(branch.sql)
            if query is None or len(branch.path) > 4:
                continue
            diagnosis = self.diagnose(branch)
            probe = self.probe[1] if self.probe and self.probe[0] == branch.sql else None
            for edit in edits(query, self.schema, self.intent, self.example, branch.feedback, diagnosis,
                              self.ordered, peer_trees, probe):
                identifier = (id(branch), edit.kind)
                if counts[identifier] >= 2:
                    continue
                rendered = sql_text(edit.tree)
                prior = len(by_identity)
                add(Move(edit.kind, rendered, branch, edit.confidence, edit.reason))
                if len(by_identity) > prior:
                    counts[identifier] += 1
        if self.probe_values is not None:
            parent, literal, values = self.probe_values
            for edit in literal_from_probe(parse(parent.sql), literal, values):
                add(Move(edit.kind, sql_text(edit.tree), parent, edit.confidence, edit.reason))
        if (self.config.get("probe", True) and not self.probed and observation["remaining"] >= 3
                and self.intent.count and self.diagnose(recent).join and recent.feedback.get("ok")
                and not recent.feedback.get("matches_example")):
            query = diagnostic_query(parse(recent.sql), self.schema)
            if query is not None and key(query) not in self.checked:
                move = Move("probe", query, recent, .55, "measure duplicate entity rows")
                move.diagnosis = self.diagnose(recent)
                by_identity[key(query)] = move
        if (self.config.get("probe", True) and not self.probed and self.example["row_count"] > 0
                and observation["remaining"] >= 2 and not best.feedback.get("matches_example")):
            for branch in active:
                diagnosis = self.diagnose(branch)
                if not branch.feedback.get("ok") or diagnosis.rows != "fewer":
                    continue
                diagnostic = literal_diagnostic(parse(branch.sql), self.schema, self.intent)
                if diagnostic is not None and key(diagnostic[0]) not in self.checked:
                    move = Move("value_probe", diagnostic[0], branch, .64,
                                "inspect a plausible text value encoding", diagnosis=diagnosis,
                                probe_literal=diagnostic[1])
                    by_identity[key(move.sql)] = move
                    break
        return list(by_identity.values())

    def features(self, move, observation):
        diagnosis = move.diagnosis
        state = self.intent
        return {f"action:{move.kind}": 1, f"error:{diagnosis.error}": 1,
                f"rows:{diagnosis.rows}": 1, f"projection:{diagnosis.projection}": 1,
                f"origin:{move.parent.origin}": 1, "confidence": move.confidence,
                "hint_agreement": move.support, "remaining": observation["remaining"],
                "prior_checks": len(self.branches), "branch_depth": len(move.parent.path),
                "prior_success": int(any(branch.feedback.get("matches_example") for branch in self.branches)),
                "parent_match": int(diagnosis.matched), "empty_example": int(diagnosis.empty),
                "aggregate": int(diagnosis.aggregate), "join": int(diagnosis.join),
                "predicate": int(diagnosis.predicate), "grouped": int(diagnosis.grouped),
                "asked_group": int(state.grouped), "asked_negative": int(state.negative),
                "asked_distinct": int(state.distinct), "asked_count": int(state.count),
                "asked_frequency": int(state.frequent), "ordered": int(self.ordered),
                "static_faults": move.faults, "projection_gain": move.projection_gain,
                "previous_action_attempts": sum(branch.origin == move.kind for branch in self.branches),
                "probe_duplicate": int(self.probe is not None and self.probe[1][0] > self.probe[1][1])}

    def priority(self, move, observation):
        diagnosis = move.diagnosis
        kind = move.kind
        base = {"hint": .55, "projection": .31, "binding": .32, "group": .27,
                "literal": .18, "order": .15, "boundary": .23, "distinct": .14,
                "contrast": .28, "compact": .10, "skeleton": .06, "semi": .17,
                "transfer": .14, "probe": -.08, "predicate": .26,
                "multiplicity": .22, "relationship": .27, "set_logic": .31,
                "single_relation": .20, "reshape": .30, "value_probe": .27,
                "value_repair": .72}.get(kind, 0.)
        base += .38 * move.confidence + .045 * min(4, move.support)
        base -= self.config.get("static_fault_weight", .17) * min(2, move.faults)
        base += self.config.get("projection_gain_weight", .17) * max(-1., min(1., move.projection_gain))
        if kind == "hint" and diagnosis.matched and diagnosis.projection == "aligned":
            base -= .16
        if kind in ("projection", "binding") and diagnosis.projection == "arity":
            base += .10
        if kind == "projection" and not diagnosis.matched and diagnosis.projection != "aligned":
            base += .13
        if kind == "compact" and diagnosis.matched and diagnosis.empty:
            base += .15
        if kind == "probe" and self.intent.count and not diagnosis.empty:
            base += self.config.get("probe_priority", .33)
        if self.memory:
            base += self.config.get("memory_weight", .8) * self.reliability(kind, diagnosis)
            if diagnosis.matched:
                base -= .4 * self.risk(kind, diagnosis)
        if self.policy:
            base += self.config.get("policy_weight", 1.25) * tree_value(self.policy["tree"], self.features(move, observation))
        return base

    def apply(self, move):
        self.pending = move
        return {"tool": "execute" if move.kind in {"probe", "value_probe"} else "check", "sql": move.sql}

    def act(self, observation):
        forced = self.start(observation)
        if forced:
            return forced
        options = self.options(observation)
        if not options:
            return self.submit()
        move = max(options, key=lambda edit: self.priority(edit, observation))
        return self.apply(move)
