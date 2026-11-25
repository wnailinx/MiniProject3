"""V3 controller with terminal verification and focused follow-up actions."""
from sqlglot import exp

from .controller import Agent, Move
from .followups import cardinality_query, followups
from .obligations import Request, relationship_refinement, signals, structural_margin
from .progress import fraction_agree
from .skills import Edit, edits
from .structure import header_agreement, invalid_bindings, key, parse, sql_text


class VerifiedAgent(Agent):
    def __init__(self, config, memory, policy, obligations):
        super().__init__(config, memory, policy)
        self.obligations = obligations.get("weights", {})
        self.request = None
        self.cardinality_edit = None
        self.probe_query = None

    def ingest(self, observation):
        if self.request is None:
            self.request = Request.read(observation["task"]["question"])
        for event in observation["history"][self.seen:]:
            if event["tool"] == "execute":
                self.probe_query = event["sql"]
        super().ingest(observation)

    def submit(self):
        incumbent = self.best()
        if not self.config.get("verify", True) or not incumbent.feedback.get("matches_example"):
            return {"tool": "submit", "sql": incumbent.sql}
        empty = not self.example["row_count"]
        base_signals = signals(incumbent.sql, self.request, self.example, self.schema)
        baseline = self.evidence(incumbent)
        gain = self.config.get("verify_margin", .18)
        choice = incumbent
        for branch in self.branches:
            if branch is incumbent or not branch.feedback.get("matches_example"):
                continue
            verified = signals(branch.sql, self.request, self.example, self.schema)
            proof = structural_margin(verified, base_signals, self.obligations, empty)
            if verified.get("tautological_join", 0) > base_signals.get("tautological_join", 0):
                proof += .18
            if "alphabetical_field" in verified and "alphabetical_field" in base_signals:
                proof += .52 * (verified["alphabetical_field"] - base_signals["alphabetical_field"])
            if "absence" in verified and "absence" in base_signals:
                proof += .22 * (verified["absence"] - base_signals["absence"])
            if self.config.get("verify_bridge", True):
                proof += .9 * relationship_refinement(incumbent.sql, branch.sql, self.schema)
            score = self.config.get("verify_weight", 1.8) * proof + .55 * (self.evidence(branch) - baseline)
            if proof > 0 and score > gain:
                gain, choice = score, branch
        return {"tool": "submit", "sql": choice.sql}

    def evidence(self, branch):
        baseline = super().evidence(branch)
        if self.config.get("row_progress", False) and not branch.feedback.get("matches_example"):
            baseline += 1.45 * fraction_agree(branch.feedback, self.example)
        return baseline

    def options(self, observation):
        options = super().options(observation)
        if self.schema is None or not self.config.get("followups", True):
            return options
        checked = set(self.checked)
        proposed = {key(move.sql) for move in options}

        def add(edit, branch, force=False):
            query = sql_text(edit.tree)
            identity = key(query)
            if identity in checked or parse(query) is None:
                return
            if identity in proposed:
                if not force:
                    return
                options[:] = [move for move in options if key(move.sql) != identity]
            proposed.add(identity)
            move = Move(edit.kind, query, branch, edit.confidence, edit.reason, diagnosis=self.diagnose(branch))
            move.faults = invalid_bindings(edit.tree, self.schema)
            if isinstance(edit.tree, exp.Select) and isinstance(parse(branch.sql), exp.Select):
                headers = self.example["columns"]
                move.projection_gain = header_agreement(edit.tree, headers) - header_agreement(parse(branch.sql), headers)
            options.append(move)

        matching = [b for b in self.branches if b.feedback.get("matches_example")]
        live = [self.branches[-1], self.best()]
        if not matching:
            recent_good = next((b for b in reversed(self.branches[:-1]) if b.feedback.get("ok")), None)
            if recent_good is not None:
                live.append(recent_good)

        seen = set()
        for branch in live:
            if branch.sql in seen or len(branch.path) > 4:
                continue
            seen.add(branch.sql)
            siblings = [parse(b.sql) for b in self.branches if b is not branch and b.feedback.get("ok")]
            previous_moves = set()
            for edit in followups(parse(branch.sql), branch, self.schema, self.intent, self.request,
                                  self.example, siblings[-3:], self.diagnose(branch), self.ordered):
                if edit.kind in previous_moves:
                    continue
                previous_moves.add(edit.kind)
                add(edit, branch)

        if self.config.get("row_progress", False) and not matching and observation["remaining"] >= 1:
            archived = [b for b in self.branches if b.sql not in seen and len(b.path) <= 4
                        and b.feedback.get("ok") and isinstance(parse(b.sql), exp.Select)]
            if archived:
                anchor = max(archived, key=lambda b: fraction_agree(b.feedback, self.example))
                if fraction_agree(anchor.feedback, self.example) >= .45:
                    diagnosis = self.diagnose(anchor)
                    count = len(options)
                    for edit in edits(parse(anchor.sql), self.schema, self.intent, self.example, anchor.feedback,
                                      diagnosis, self.ordered, [], None):
                        if edit.kind in {"projection", "group", "reshape", "relationship", "compact", "distinct"}:
                            add(edit, anchor)
                        if len(options) >= count + 2:
                            break

        if not self.config.get("targeted_probe", True):
            return options
        if self.cardinality_edit and self.probe and self.probe_query == self.cardinality_edit[2]:
            parent, tree, _ = self.cardinality_edit
            expected = self.example.get("rows", [])
            if expected and len(expected[0]) == 1 and self.probe[1][0] > self.probe[1][1] == expected[0][0]:
                add(Edit("probed_count", tree, "observed distinct entity count explains the mismatch", .98),
                    parent, force=True)
        remaining = observation["remaining"]
        if matching or self.probed or remaining < 2:
            return options
        for branch in live:
            if not branch.feedback.get("ok") or self.diagnose(branch).rows != "more":
                continue
            tree = parse(branch.sql)
            if not isinstance(tree, exp.Select):
                continue
            query_and_repair = cardinality_query(tree, self.schema, self.intent)
            if not query_and_repair:
                continue
            query, repair = query_and_repair
            rendered = sql_text(query)
            if key(rendered) in proposed or key(rendered) in checked:
                continue
            self.cardinality_edit = (branch, repair, rendered)
            options.append(Move("probe", rendered, branch, .92, "test the requested entity grain",
                                diagnosis=self.diagnose(branch)))
            break
        return options

    def priority(self, move, observation):
        stronger = {"inner_binding": .95, "restore_filter": .87, "correlated_average": 1.,
                    "absence_chain": .93, "probed_count": 2., "entity_literal": 1.15,
                    "unrestricted_list": 1.02, "alphabetical": 1.12}
        if move.kind in stronger:
            return stronger[move.kind] + .2 * move.projection_gain - .18 * move.faults
        if move.reason == "test the requested entity grain":
            return .93
        return super().priority(move, observation)
