import random
import heapq
import math


class SearchAgent:

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'
        self.walls = set()
        self.grid_size = (10, 10)

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        width, height = grid_size

        if start == goal:
            return []

        frontier = [(start, [])]
        reached = {start}

        while frontier:
            current_pos, path_taken = frontier.pop(0)
            x, y = current_pos
            neighbors = [
                ((x, y + 1), 'Up'),
                ((x, y - 1), 'Down'),
                ((x - 1, y), 'Left'),
                ((x + 1, y), 'Right'),
            ]

            for neighbor, action in neighbors:
                nx, ny = neighbor
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbor in walls or neighbor in reached:
                    continue

                new_path = path_taken + [action]
                if neighbor == goal:
                    return new_path

                reached.add(neighbor)
                frontier.append((neighbor, new_path))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        width, height = grid_size

        frontier = [(0, start, [])]
        reached_states = set()

        while frontier:
            g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            x, y = current_pos
            neighbors = [
                ((x, y + 1), 'Up'),
                ((x, y - 1), 'Down'),
                ((x - 1, y), 'Left'),
                ((x + 1, y), 'Right'),
            ]

            for neighbor, action in neighbors:
                nx, ny = neighbor
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbor in walls or neighbor in reached_states:
                    continue

                g_new = g_cost + 1
                heapq.heappush(frontier, (g_new, neighbor, path_taken + [action]))

        return None

    def manhattan_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        width, height = grid_size

        def h(pos):
            if heuristic_type == 'manhattan':
                return self.manhattan_distance(pos, goal)
            return self.euclidean_distance(pos, goal)

        g0 = 0
        f0 = g0 + h(start)
        frontier = [(f0, g0, start, [])]
        reached_states = set()

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            x, y = current_pos
            neighbors = [
                ((x, y + 1), 'Up'),
                ((x, y - 1), 'Down'),
                ((x - 1, y), 'Left'),
                ((x + 1, y), 'Right'),
            ]

            for neighbor, action in neighbors:
                nx, ny = neighbor
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbor in walls or neighbor in reached_states:
                    continue

                g_new = g_cost + 1
                h_new = h(neighbor)
                f_new = g_new + h_new
                heapq.heappush(frontier, (f_new, g_new, neighbor, path_taken + [action]))

        return None

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            agent_pos = tuple(percept['agent_pos'])
            food_positions = percept.get('food_positions')

            if food_positions:
                goal = min(food_positions, key=lambda f: self.manhattan_distance(agent_pos, f))

                if self.active_algo == 'BFS':
                    self.plan = self.bfs_search(agent_pos, goal, self.walls, self.grid_size) or []
                elif self.active_algo == 'UCS':
                    self.plan = self.ucs_search(agent_pos, goal, self.walls, self.grid_size) or []
                elif self.active_algo == 'AStar':
                    self.plan = self.astar_search(
                        agent_pos, goal, self.walls, self.grid_size, heuristic_type='manhattan'
                    ) or []

        if self.plan:
            return self.plan.pop(0)

        return random.choice(['Up', 'Down', 'Left', 'Right'])