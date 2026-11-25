"""Independent terminal selection and one-shot ambiguity checks."""
from functools import lru_cache
import json
import os
from pathlib import Path
from threading import Lock

from sqlglot import exp

from .controller import Move
from .followups import cardinality_query, entity_name_filter
from .obligations import compare, signals
from .planning import PlanningAgent
from .plans import plans
from .semantic import (CONFLICT, SATISFIED, claims, equivalent_after_removing_restriction, family,
                       improvements, restriction_repairs, separate_memberships)
from .skills import Edit, relationships
from .structure import invalid_bindings, key, parse, sql_text


LOG_LOCK = Lock()


@lru_cache(maxsize=1)
def policy():
    path = Path(__file__).resolve().parent.parent / "semantic_policy.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"accepted_families": []}


def log(event):
    path = os.environ.get("MP3_V6_TELEMETRY")
    if path:
        with LOG_LOCK, Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def protected_losses(previous, proposed, agent):
    old = signals(previous, agent.request, agent.example, agent.schema)
    new = signals(proposed, agent.request, agent.example, agent.schema)
    dimensions = {"aligned_projection", "quoted_filter", "numeric_boundary", "explicit_order",
                  "explicit_limit", "explicit_distinct", "alphabetical_field"}
    return {tag for tag, direction in compare(new, old) if tag in dimensions and direction < 0}


class SemanticAgent(PlanningAgent):
    def __init__(self, *args):
        super().__init__(*args)
        self.semantic_cache = {}
        self.ambiguity_checked = False
        self.ambiguity_seen = set()

    def claims(self, sql):
        if sql not in self.semantic_cache:
            self.semantic_cache[sql] = claims(sql, self.request, self.schema)
        return self.semantic_cache[sql]

    def upgrades(self, previous, proposed):
        old, new = self.claims(previous), self.claims(proposed)
        better = improvements(old, new)
        if "count_restriction" in old and equivalent_after_removing_restriction(
                previous, proposed, self.request, self.schema):
            better.add("count_restriction")
        if not better or improvements(new, old) or protected_losses(previous, proposed, self):
            return set()
        # Preserve positive obligations unless a replacement can also establish them.
        if any(claim.state == SATISFIED and tag not in new for tag, claim in old.items()
               if tag != "join_key" or "count_restriction" not in better):
            return set()
        return better

    def submit(self):
        prior = super().submit()
        if not self.config.get("semantic_selection") or self.schema is None:
            return prior
        current = next(b for b in self.branches if b.sql == prior["sql"])
        if not current.feedback.get("matches_example"):
            return prior
        accepted = set(policy()["accepted_families"])
        candidates = []
        for branch in self.branches:
            if branch is current or not branch.feedback.get("matches_example"):
                continue
            better = self.upgrades(current.sql, branch.sql)
            if better and {family(tag) for tag in better} <= accepted:
                candidates.append((len(better), self.evidence(branch), branch))
        if not candidates:
            return prior
        choice = max(candidates, key=lambda item: item[:2])[2]
        return {"tool": "submit", "sql": choice.sql}

    def rebuilds(self, branch):
        tree = parse(branch.sql)
        if not isinstance(tree, exp.Select) or len(branch.path) >= 4:
            return
        yield from plans(tree, self.schema, self.intent, self.request, self.example, self.plan_priors)
        conflict = self.claims(branch.sql)
        if conflict.get("independent_membership") and conflict["independent_membership"].state == CONFLICT:
            rebuilt = separate_memberships(tree, self.request, self.schema)
            if rebuilt is not None:
                yield Edit("plan_set", rebuilt, "require the two named memberships separately", .92)
        if conflict.get("join_key") and conflict["join_key"].state == CONFLICT:
            yield from relationships(tree, self.schema)
        for rebuilt in restriction_repairs(tree, self.request, self.schema):
            yield Edit("plan_join", rebuilt, "remove the unrequested restriction on the distinct count", .88)
        if conflict.get("count_grain") and conflict["count_grain"].state == CONFLICT:
            count = cardinality_query(tree, self.schema, self.intent)
            if count:
                yield Edit("plan_group", count[1], "count the named entity key across the existing join", .9)
        if any(family(tag) == "entity_name" and c.state == CONFLICT for tag, c in conflict.items()):
            yield from entity_name_filter(tree, self.schema, self.intent, self.diagnose(branch))

    def ambiguity(self, observation, options):
        if (not self.config.get("ambiguity_rebuild") or self.ambiguity_checked or self.schema is None
                or not 0 < observation["remaining"] <= self.config.get("ambiguity_remaining", 2)
                or len(self.branches) < self.config.get("plan_after_checks", 3)):
            return None
        matching = [b for b in self.branches if b.feedback.get("matches_example")]
        if not matching:
            return None
        previous = super().submit()["sql"]
        parent = next(b for b in matching if b.sql == previous)
        conflicts = {tag for tag, claim in self.claims(previous).items() if claim.state == CONFLICT}
        if not conflicts:
            return None
        empty = not self.example["row_count"]
        if not empty:
            # A nonempty match is ambiguous if matched branches disagree on a named obligation.
            disagreement = any(self.upgrades(previous, b.sql) for b in matching if b is not parent)
            if not disagreement:
                return None
        emitted = {}
        for edit in self.rebuilds(parent):
            sql = sql_text(edit.tree)
            identity = key(sql)
            if identity in self.checked or identity in emitted or invalid_bindings(edit.tree, self.schema):
                continue
            improved = self.upgrades(previous, sql)
            if improved:
                emitted[identity] = (improved, edit, sql)
        info = {"task": observation["task"]["id"], "remaining": observation["remaining"],
                "empty": empty, "conflicts": sorted({family(tag) for tag in conflicts}),
                "generated": len(emitted)}
        status = (previous, tuple(sorted(conflicts)))
        if not emitted:
            if status not in self.ambiguity_seen:
                self.ambiguity_seen.add(status)
                log(info | {"event": "conflict_without_rebuild"})
            return None
        improved, edit, sql = max(emitted.values(), key=lambda item: (len(item[0]), item[1].confidence))
        self.ambiguity_checked = True
        log(info | {"event": "ambiguity_check", "sql": sql, "reason": edit.reason,
                    "families": sorted({family(tag) for tag in improved}),
                    "already_offered": any(key(move.sql) == key(sql) for move in options)})
        return self.apply(Move(edit.kind, sql, parent, edit.confidence, edit.reason, diagnosis=self.diagnose(parent)))

    def act(self, observation):
        forced = self.start(observation)
        if forced is not None:
            return forced
        options = self.options(observation)
        action = self.ambiguity(observation, options)
        if action is not None:
            return action
        if not options:
            return self.submit()
        return self.apply(max(options, key=lambda move: self.priority(move, observation)))
