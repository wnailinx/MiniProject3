"""Preserve the feedback controller; rerank only at the actual terminal action."""
from .hybrid import HybridAgent
from .terminal_rank import select


class RankedAgent(HybridAgent):
    def act(self, observation):
        action = super().act(observation)
        if action['tool']=='submit' and self.config.get('terminal_rerank'):
            # A generated, unchecked rescue must not be mistaken for a checked candidate.
            if any(e['tool']=='check' and e['sql']==action['sql'] and
                   e['observation'].get('matches_example') for e in observation['history']):
                action = {'tool':'submit','sql':select(observation,action['sql'],self.config)}
        return action
