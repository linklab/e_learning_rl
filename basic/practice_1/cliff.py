import random
import time


class CliffGridWorld():
    def __init__(
            self,
            height=4, width=12,         # 격자판의 크기
            start_state=(3, 0),         # 시작 상태
            terminal_states=[(3, 11)],  # 종료 상태
            transition_reward=-1.0,     # 일반적인 상태 전이 보상
            terminal_reward=1.0,        # 종료 상태로 이동하는 행동 수행 시 받는 보상
            outward_reward=-1.0,        # 미로 바깥으로 이동하는 행동 수행 시 받는 보상 (이동하지 않고 제자리 유지)
            cliff_states=None           # 절벽 상태
    ):
        self.__version__ = "0.0.1"

        # 그리드월드의 세로 길이
        self.HEIGHT = height

        # 그리드월드의 가로 길이
        self.WIDTH = width

        self.STATES = []
        self.num_states = self.WIDTH * self.HEIGHT

        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                self.STATES.append((i, j))

        for state in terminal_states:     # 터미널 스테이트 제거
            self.STATES.remove(state)

        # 모든 가능한 행동
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3

        self.ACTIONS = [
            self.ACTION_UP,
            self.ACTION_DOWN,
            self.ACTION_LEFT,
            self.ACTION_RIGHT
        ]

        self.ACTION_SYMBOLS = ["\u21E7", "\u21E9", "\u21E6", "\u21E8"] # UP, DOWN, LEFT, RIGHT
        #self.ACTION_SYMBOLS = ["\u2191", "\u2193", "\u2190", "\u2192"] # UP, DOWN, LEFT, RIGHT
        self.NUM_ACTIONS = len(self.ACTIONS)

        # 시작 상태 위치
        self.START_STATE = start_state

        # 종료 상태 위치
        self.TERMINAL_STATES = terminal_states

        # 웜홀 상태 위치
        self.CLIFF_STATES = cliff_states

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.transition_reward = transition_reward

        self.terminal_reward = terminal_reward
        self.outward_reward = outward_reward

        self.current_state = None

    def reset(self):
        self.current_state = self.START_STATE
        return self.current_state

    def moveto(self, state):
        self.current_state = state

    def is_cliff_state(self, state):
        i, j = state

        if self.CLIFF_STATES is not None and len(self.CLIFF_STATES) > 0:
            for cliff_info in self.CLIFF_STATES:
                cliff_state = cliff_info[0]
                if i == cliff_state[0] and j == cliff_state[1]:
                    return True
        return False

    def get_next_state_cliff(self, state):
        i, j = state
        next_state = None

        for cliff_info in self.CLIFF_STATES:
            cliff_state = cliff_info[0]
            cliff_prime_state = cliff_info[1]

            if i == cliff_state[0] and j == cliff_state[1]:
                next_state = cliff_prime_state
                break
        return next_state

    def get_reward_cliff(self, state):
        i, j = state
        reward = None

        for cliff_info in self.CLIFF_STATES:
            cliff_state = cliff_info[0]
            cliff_reward = cliff_info[2]

            if i == cliff_state[0] and j == cliff_state[1]:
                reward = cliff_reward
                break

        return reward

    def get_next_state(self, state, action):
        i, j = state

        if self.is_cliff_state(state):
            next_state = self.get_next_state_cliff(state)
            next_i = next_state[0]
            next_j = next_state[1]
        elif (i, j) in self.TERMINAL_STATES:
            next_i = i
            next_j = j
        else:
            if action == self.ACTION_UP:
                next_i = max(i - 1, 0)
                next_j = j
            elif action == self.ACTION_DOWN:
                next_i = min(i + 1, self.HEIGHT - 1)
                next_j = j
            elif action == self.ACTION_LEFT:
                next_i = i
                next_j = max(j - 1, 0)
            elif action == self.ACTION_RIGHT:
                next_i = i
                next_j = min(j + 1, self.WIDTH - 1)
            else:
                raise ValueError()

        return next_i, next_j

    def get_reward(self, state, next_state):
        i, j = state
        next_i, next_j = next_state

        if self.is_cliff_state(state):
            reward = self.get_reward_cliff(state)
        else:
            if (next_i, next_j) in self.TERMINAL_STATES:
                reward = self.terminal_reward
            else:
                if i == next_i and j == next_j:
                    reward = self.outward_reward
                else:
                    reward = self.transition_reward

        return reward

    def get_state_action_probability(self, state, action):
        next_i, next_j = self.get_next_state(state, action)

        reward = self.get_reward(state, (next_i, next_j))
        transition_prob = 1.0

        return (next_i, next_j), reward, transition_prob

    # take @action in @state
    # @return: (reward, new state)
    def step(self, action):
        next_i, next_j = self.get_next_state(state=self.current_state, action=action)

        reward = self.get_reward(self.current_state, (next_i, next_j))

        self.current_state = (next_i, next_j)

        if self.current_state in self.TERMINAL_STATES:
            done = True
        else:
            done = False

        return (next_i, next_j), reward, done, None

    def render(self, mode='human'):
        print(self.__str__())

    # 임의의 행동을 선택하여 반환
    def get_random_action(self):
        return random.choice(self.ACTIONS)

    def __str__(self):
        gridworld_str = ""
        for i in range(self.HEIGHT):
            gridworld_str += "-" * 99 + "\n"

            for j in range(self.WIDTH):
                if self.current_state[0] == i and self.current_state[1] == j:
                    gridworld_str += "|   {0}   ".format("*")
                elif (i, j) == self.START_STATE:
                    gridworld_str += "|   {0}   ".format("S")
                elif (i, j) in self.TERMINAL_STATES:
                    gridworld_str += "|   {0}   ".format("G") if j < 10 else "|   {0}    ".format("G")
                elif self.CLIFF_STATES and (i, j) in [state[0] for state in self.CLIFF_STATES]:
                    gridworld_str += "|   {0}   ".format("W") if j < 10 else "|   {0}    ".format("W")
                else:
                    gridworld_str += "|       " if j < 10 else "|        "
            gridworld_str += "|\n"

            for j in range(self.WIDTH):
                gridworld_str += "| ({0},{1}) ".format(i, j)

            gridworld_str += "|\n"

        gridworld_str += "-" * 99 + "\n"
        return gridworld_str


def main_cliff_gridworld():
    # 그리드월드 높이와 너비
    GRID_HEIGHT = 4
    GRID_WIDTH = 12

    # 초기 상태와 종료 상태
    START_STATE = (3, 0)
    TERMINAL_STATES = [(3, 11)]
    CLIFF_STATES = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]

    env = CliffGridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=START_STATE,
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0,
        cliff_states=[(s, START_STATE, -100.0) for s in CLIFF_STATES]
    )

    env.reset()
    print("reset")
    env.render()

    done = False
    total_steps = 0
    while not done:
        total_steps += 1
        action = env.get_random_action()
        next_state, reward, done, _ = env.step(action)
        print("action: {0}, reward: {1}, done: {2}, total_steps: {3}".format(
            env.ACTION_SYMBOLS[action],
            reward, done, total_steps
        ))
        env.render()
        time.sleep(3)


if __name__ == "__main__":
    main_cliff_gridworld()
