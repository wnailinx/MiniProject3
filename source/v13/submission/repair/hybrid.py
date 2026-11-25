"""Keep the proven rule loop; learned generation rescues its unresolved tail."""
from .controller import Move
from .joint_agent import JointAgent
from .structure import key, parse
from .rewriter import payload, generate


class HybridAgent(JointAgent):
    def __init__(self, *args):
        super().__init__(*args)
        self.generations = 0

    def rescue(self, observation, schema, checks):
        self.generations += 1
        data = payload(observation["task"]["question"], observation["issue_sql"],
                       observation["example"], observation["ordered"], schema, checks)
        sql = generate(data, observation["task"]["id"], self.config)
        if not sql or key(sql) in self.checked or parse(sql) is None:
            return None
        if observation["remaining"]:
            parent = self.best()
            move = Move("neural_rewrite", sql, parent, .9, "feedback-conditioned whole-query rewrite",
                        diagnosis=self.diagnose(parent))
            return self.apply(move), [move], move
        # A new final SQL is legal after the official feedback loop, even with no checks left.
        return {"tool":"submit", "sql":sql}, [], None

    def act(self, observation):
        action = self.start(observation)
        checks = [e for e in observation["history"] if e["tool"] == "check"]
        schema = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
        eligible = (self.config.get("rewriter", "off") != "off" and checks and schema
                    and not any(e["observation"].get("matches_example") for e in checks)
                    and self.generations < self.config.get("max_generations", 1))
        options, chosen = [], None
        at_tail = observation["remaining"] <= self.config.get("rescue_remaining", 1)
        neural = self.rescue(observation, schema, checks) if eligible and at_tail else None
        if neural is None:
            if action is None:
                options = self.offer(observation)
                chosen = self.choose(options, observation) if options else None
                if chosen is not None:
                    action = self.apply(chosen)
                    if action["tool"] == "check":
                        self.lineage[chosen.sql] = self.root(chosen)
                else:
                    action = self.submit()
            # Also rescue a rule loop that exhausts its menu before exhausting its budget.
            if action["tool"] == "submit" and eligible and not at_tail:
                neural = self.rescue(observation, schema, checks)
        if neural is not None:
            action, options, chosen = neural
        self.record(observation, action, options, chosen)
        return action
