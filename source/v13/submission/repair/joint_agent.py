"""Admit complete composed queries from the first schema-informed decision."""
from sqlglot import exp

from .controller import Move
from .coverage_agent import CoverageAgent
from .joint import constructions
from .structure import header_agreement, key, parse, sql_text


class JointAgent(CoverageAgent):
    def __init__(self, *args):
        super().__init__(*args)
        self.construction_info = {}
        self.question_text = ""

    def ingest(self, observation):
        self.question_text = observation["task"]["question"]
        return super().ingest(observation)

    def submit(self):
        selected = super().submit()
        if self.config.get("pairwise_selection") and self.schema is not None:
            from .pairwise import choose
            return choose(self, selected)
        return selected

    def offer(self, observation):
        options = super().offer(observation)
        if self.schema is None or not (self.config.get("joint_filter") or self.config.get("joint_stats")):
            return options
        if self.config.get("joint_after_match", "all") == "empty" and self.example["row_count"] and any(
                branch.feedback.get("matches_example") for branch in self.branches):
            return options
        known = self.checked | {key(move.sql) for move in options}
        parent = self.best()
        remaining = []
        for proposal in constructions(self):
            statement = sql_text(proposal.tree)
            canonical = key(statement)
            if canonical in known:
                continue
            known.add(canonical)
            self.construction_info[statement] = proposal
            old = parse(parent.sql)
            before = header_agreement(old, self.example["columns"]) if isinstance(old, exp.Select) else 0.
            gain = header_agreement(proposal.tree, self.example["columns"]) - before
            remaining.append(Move(proposal.family, statement, parent, min(.99, proposal.evidence),
                                  " + ".join(proposal.operations), diagnosis=self.diagnose(parent), projection_gain=gain))
        return options + remaining[:self.config.get("joint_per_round", 3)]

    def priority(self, move, observation):
        if move.kind not in {"joint_filter", "joint_stats"}:
            return super().priority(move, observation)
        matched = any(branch.feedback.get("matches_example") for branch in self.branches)
        attempts = sum(branch.origin == move.kind for branch in self.branches[-2:])
        evidence = self.construction_info[move.sql].evidence
        base = self.config.get("joint_priority", 1.12) + .32 * (evidence - .8) + .25 * move.projection_gain
        if move.diagnosis.error in {"column", "table", "aggregate"}:
            base += .40
        if matched and self.example["row_count"]:
            base -= .45
        return base - .16 * attempts

    def record(self, observation, action, options, chosen):
        super().record(observation, action, options, chosen)
        if not self.audit:
            return
        # The inherited record gets serialized at submit, so annotate before that step.
        if action["tool"] != "submit":
            for item in self.steps[-1]["options"]:
                info = self.construction_info.get(item["sql"])
                if info:
                    item.update(source_sql=info.source, operations=info.operations, dependencies=info.dependencies,
                                dependencies_satisfied=True, depth=len(info.operations))
