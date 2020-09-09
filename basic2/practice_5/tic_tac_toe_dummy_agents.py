import random


class Dummy_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def get_action(self, state):
        available_positions = state.get_available_positions()
        action = random.choice(available_positions)
        return action