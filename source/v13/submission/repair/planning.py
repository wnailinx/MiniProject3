"""A bounded structural planner on the verified v3 controller."""
from collections import Counter

from sqlglot import exp

from .actions import action_features, policy_score
from .controller import Move
from .plans import plans
from .structure import header_agreement, invalid_bindings, key, parse, sql_text
from .verified import VerifiedAgent


PLAN_KINDS = {"plan_join", "plan_group", "plan_set", "plan_absence"}


class PlanningAgent(VerifiedAgent):
    def __init__(self, config, memory, policy, obligations, plan_priors, action_policy):
        super().__init__(config, memory, policy, obligations)
        self.plan_priors = plan_priors["families"]
        self.action_policy = action_policy

    def options(self, observation):
        options = super().options(observation)
        if self.schema is None or not self.config.get("plans", True):
            return options
        matches = [branch for branch in self.branches if branch.feedback.get("matches_example")]
        if matches and not self.config.get("plans_on_ambiguous", False):
            return options
        if matches and self.example["row_count"] > 0:
            return options
        if len(self.branches) < self.config.get("plan_after_checks", 2):
            return options

        # Each new plan is anchored to a checked branch. Only one per family
        # and two per decision are admitted to the runtime menu.
        active = [self.best(), self.branches[-1]]
        checked = {key(branch.sql) for branch in self.branches}
        proposed = {key(option.sql) for option in options}
        generated = {}
        for branch in active:
            tree = parse(branch.sql)
            if not isinstance(tree, exp.Select) or len(branch.path) >= 4:
                continue
            diagnosis = self.diagnose(branch)
            for edit in plans(tree, self.schema, self.intent, self.request, self.example, self.plan_priors):
                statement = sql_text(edit.tree)
                identity = key(statement)
                if identity in checked or identity in proposed or edit.kind in generated:
                    continue
                if invalid_bindings(edit.tree, self.schema):
                    continue
                if matches and not tree.args.get("joins") and edit.kind == "plan_join":
                    continue
                move = Move(edit.kind, statement, branch, edit.confidence, edit.reason, diagnosis=diagnosis)
                if isinstance(edit.tree, exp.Select):
                    move.projection_gain = (header_agreement(edit.tree, self.example["columns"])
                                            - header_agreement(tree, self.example["columns"]))
                generated[edit.kind] = move
                proposed.add(identity)
            if len(generated) >= 2:
                break
        additions = sorted(generated.values(), key=lambda move: self.priority(move, observation), reverse=True)
        return options + additions[:2]

    def base_priority(self, move, observation):
        if move.kind not in PLAN_KINDS:
            return super().priority(move, observation)
        kind = "set" if move.kind in {"plan_set", "plan_absence"} else move.kind.removeprefix("plan_")
        profile = self.plan_priors[kind]
        rate = profile["draft_different"] / max(1, profile["examples"])
        more = .17 if not any(b.feedback.get("matches_example") for b in self.branches) else 0.
        rate_bonus = .26 * rate
        attempts = Counter(branch.origin for branch in self.branches)
        return (.66 + more + rate_bonus + .28 * move.projection_gain + self.config.get("plan_priority", 0.)
                - .17 * attempts[move.kind])

    def priority(self, move, observation):
        base = self.base_priority(move, observation)
        if not self.config.get("action_rollout", False) or not self.action_policy:
            return base
        features = action_features(self, move, observation)
        if self.config.get("action_late_only", False) and (
                observation["remaining"] > 2 or features["checked_public_matches"]):
            return base
        if (features["checked_public_matches"] and self.example["row_count"]
                and not self.config.get("action_when_matched", False)):
            return base
        return base + self.config.get("action_weight", 1.6) * policy_score(self.action_policy, features)

    def evidence(self, branch):
        score = super().evidence(branch)
        if branch.origin in PLAN_KINDS and branch.feedback.get("matches_example"):
            score += self.config.get("plan_terminal", .08)
        return score
