import random
# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept['agent_pos']
        return random.choice(self.actions_pool)

class SimpleReflexAgent:
    """Reacts only to the current percept — no memory of past percepts or actions."""

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'Right'          # keep moving; food auto-collects on arrival
        if percept['wall_ahead']:
            return 'Up'             # always the SAME reaction to "wall ahead"
        return 'Right'              # default: keep moving forward

class ModelBasedAgent:
    """Maintains internal state so it can escape situations that look identical
    on the surface but have a different history behind them."""

    def __init__(self):
        self.percept_history = []        # Sensor Model: what we've perceived
        self.last_action = None          # Transition Model: what we last did
        self.consecutive_wall_hits = 0
        self.turn_order = ['Up', 'Right', 'Down', 'Left']

    def sense_and_act(self, percept: dict) -> str:
        # Update state first, as required
        self.percept_history.append(percept)
        self.consecutive_wall_hits = self.consecutive_wall_hits + 1 if percept['wall_ahead'] else 0

        # Rules query the memory
        if percept['wall_ahead']:
            action = self.turn_order[self.consecutive_wall_hits % 4]
        else:
            action = 'Right'

        self.last_action = action
        return action

class SearchAgent:
    def bfs_search(self, start, goal, walls, grid_size):
        raise NotImplementedError("Implement in Practical 3")