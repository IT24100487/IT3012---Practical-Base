# agent.py
from collections import deque
import heapq


class SearchAgent:
    """A Goal-Based/Planning Agent that uses uninformed search (BFS/DFS/UCS)
    to plan a full path to the nearest food before acting."""

    # (dx, dy, action_name) for each move — matches execute_action in the env
    MOVES = [(0, 1, 'Up'), (0, -1, 'Down'), (-1, 0, 'Left'), (1, 0, 'Right')]

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'  # switch to 'DFS' or 'UCS' to compare

    def get_successors(self, state, grid_size, walls):
        """Return valid (next_state, action, step_cost) triples from `state`."""
        width, height = grid_size
        x, y = state
        successors = []
        for dx, dy, action in self.MOVES:
            nx, ny = x + dx, y + dy
            # Stay inside the grid (mirrors the clamping in execute_action)
            nx = max(0, min(width - 1, nx))
            ny = max(0, min(height - 1, ny))
            if (nx, ny) in walls:
                continue  # walls are impassable for planning purposes
            successors.append(((nx, ny), action, 1))  # every step costs 1
        return successors

    def bfs_search(self, start, goal, grid_size, walls):
        frontier = deque([start])
        reached = {start: None}  # state -> (parent_state, action)

        while frontier:
            state = frontier.popleft()
            if state == goal:
                return self._reconstruct_path(reached, start, goal)

            for next_state, action, _cost in self.get_successors(state, grid_size, walls):
                if next_state not in reached:
                    reached[next_state] = (state, action)
                    frontier.append(next_state)

        return []  # no path found

    def dfs_search(self, start, goal, grid_size, walls):
        frontier = [start]
        reached = {start: None}

        while frontier:
            state = frontier.pop()  # LIFO
            if state == goal:
                return self._reconstruct_path(reached, start, goal)

            for next_state, action, _cost in self.get_successors(state, grid_size, walls):
                if next_state not in reached:
                    reached[next_state] = (state, action)
                    frontier.append(next_state)

        return []

    def ucs_search(self, start, goal, grid_size, walls):
        counter = 0  # tie-breaker so heapq never compares tuples of states
        frontier = [(0, counter, start)]  # (g_cost, tie_breaker, state)
        reached = {start: (None, None, 0)}  # state -> (parent, action, best_cost)

        while frontier:
            cost, _, state = heapq.heappop(frontier)
            if state == goal:
                return self._reconstruct_path(reached, start, goal)

            if cost > reached[state][2]:
                continue  # stale entry, a cheaper path was already found

            for next_state, action, step_cost in self.get_successors(state, grid_size, walls):
                new_cost = cost + step_cost
                if next_state not in reached or new_cost < reached[next_state][2]:
                    reached[next_state] = (state, action, new_cost)
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, next_state))

        return []

    def _reconstruct_path(self, reached, start, goal):
        """Walk parent pointers backward from goal to start, then reverse."""
        actions = []
        state = goal
        while state != start:
            parent, action = reached[state][0], reached[state][1]
            actions.append(action)
            state = parent
        actions.reverse()
        return actions

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            agent_pos = tuple(percept['agent_pos'])
            all_food = [tuple(f) for f in percept['all_food']]
            walls = set(tuple(w) for w in percept['walls'])
            grid_size = percept['grid_size']

            if not all_food:
                return 'Up'  # nothing left to do

            # Closest food by Manhattan distance (a cheap heuristic just to pick a goal)
            goal = min(all_food, key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1]))

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(agent_pos, goal, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(agent_pos, goal, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(agent_pos, goal, grid_size, walls)

            if not self.plan:
                return 'Up'  # goal unreachable — fallback

        return self.plan.pop(0)


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)