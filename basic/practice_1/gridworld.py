import time

import gym

# -------------------------------
# |(0,0)|(0,1)|(0,2)|(0,3)|(0,4)|
# |(1,0)|(1,1)|(1,2)|(1,3)|(1,4)|
# |(2,0)|(2,1)|(2,2)|(2,3)|(2,4)|
# |(3,0)|(3,1)|(3,2)|(3,3)|(3,4)|
# |(4,0)|(4,1)|(4,2)|(4,3)|(4,4)|
# -------------------------------


class GridWorld(gym.Env):
    def __init__(
            self,
            height=5, width=5,        # 격자판의 크기
            start_state=(0, 0),       # 시작 상태
            terminal_states=[(4, 4)], # 종료 상태
            transition_reward=0.0,    # 일반적인 상태 전이 보상
            terminal_reward=1.0,      # 종료 상태로 이동하는 행동 수행 시 받는 보상
            outward_reward=-0.0,      # 미로 바깥으로 이동하는 행동 수행 시 받는 보상 (이동하지 않고 제자리 유지)
            warm_hole_states=None     # 윔홀 정의
    ):
        self.__version__ = "0.0.1"

        # 그리드월드의 세로 길이
        self.HEIGHT = height

        # 그리드월드의 가로 길이
        self.WIDTH = width

        self.observation_space = gym.spaces.MultiDiscrete([self.HEIGHT, self.WIDTH])
        self.action_space = gym.spaces.Discrete(4)

        self.observation_space.STATES = []
        self.observation_space.num_states = self.WIDTH * self.HEIGHT

        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                self.observation_space.STATES.append((i, j))

        for state in terminal_states:     # 터미널 스테이트 제거
            self.observation_space.STATES.remove(state)

        # 모든 가능한 행동
        self.action_space.ACTION_UP = 0
        self.action_space.ACTION_DOWN = 1
        self.action_space.ACTION_LEFT = 2
        self.action_space.ACTION_RIGHT = 3

        self.action_space.ACTIONS = [
            self.action_space.ACTION_UP,
            self.action_space.ACTION_DOWN,
            self.action_space.ACTION_LEFT,
            self.action_space.ACTION_RIGHT
        ]

        self.action_space.ACTION_SYMBOLS = ["\u2191", "\u2193", "\u2190", "\u2192"]
        self.action_space.NUM_ACTIONS = len(self.action_space.ACTIONS)

        # 시작 상태 위치
        self.observation_space.START_STATE = start_state

        # 종료 상태 위치
        self.observation_space.TERMINAL_STATES = terminal_states

        # 웜홀 상태 위치
        self.observation_space.WARM_HOLE_STATES = warm_hole_states

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.transition_reward = transition_reward

        self.terminal_reward = terminal_reward
        self.outward_reward = outward_reward


        self.current_state = None

    def reset(self):
        self.current_state = self.observation_space.START_STATE
        return self.current_state

    def moveto(self, state):
        self.current_state = state

    def is_warm_hole_state(self, state):
        i, j = state

        if self.observation_space.WARM_HOLE_STATES is not None and len(self.observation_space.WARM_HOLE_STATES) > 0:
            for warm_hole_info in self.observation_space.WARM_HOLE_STATES:
                warm_hole_state = warm_hole_info[0]
                if i == warm_hole_state[0] and j == warm_hole_state[1]:
                    return True
        return False

    def get_next_state_warm_hole(self, state):
        i, j = state
        next_state = None

        for warm_hole_info in self.observation_space.WARM_HOLE_STATES:
            warm_hole_state = warm_hole_info[0]
            warm_hole_prime_state = warm_hole_info[1]

            if i == warm_hole_state[0] and j == warm_hole_state[1]:
                next_state = warm_hole_prime_state
                break
        return next_state

    def get_reward_warm_hole(self, state):
        i, j = state
        reward = None

        for warm_hole_info in self.observation_space.WARM_HOLE_STATES:
            warm_hole_state = warm_hole_info[0]
            warm_hole_reward = warm_hole_info[2]

            if i == warm_hole_state[0] and j == warm_hole_state[1]:
                reward = warm_hole_reward
                break

        return reward

    def get_next_state(self, state, action):
        i, j = state

        if self.is_warm_hole_state(state):
            next_state = self.get_next_state_warm_hole(state)
            next_i = next_state[0]
            next_j = next_state[1]
        elif (i, j) in self.observation_space.TERMINAL_STATES:
            next_i = i
            next_j = j
        else:
            if action == self.action_space.ACTION_UP:
                next_i = max(i - 1, 0)
                next_j = j
            elif action == self.action_space.ACTION_DOWN:
                next_i = min(i + 1, self.HEIGHT - 1)
                next_j = j
            elif action == self.action_space.ACTION_LEFT:
                next_i = i
                next_j = max(j - 1, 0)
            elif action == self.action_space.ACTION_RIGHT:
                next_i = i
                next_j = min(j + 1, self.WIDTH - 1)
            else:
                raise ValueError()

        return next_i, next_j

    def get_reward(self, state, next_state):
        i, j = state
        next_i, next_j = next_state

        if self.is_warm_hole_state(state):
            reward = self.get_reward_warm_hole(state)
        else:
            if (next_i, next_j) in self.observation_space.TERMINAL_STATES:
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

        if self.current_state in self.observation_space.TERMINAL_STATES:
            done = True
        else:
            done = False

        return (next_i, next_j), reward, done, None

    def render(self, mode='human'):
        print(self.__str__())

    def __str__(self):
        gridworld_str = ""
        for i in range(self.HEIGHT):
            gridworld_str += "-------------------------------\n"

            for j in range(self.WIDTH):
                if self.current_state[0] == i and self.current_state[1] == j:
                    gridworld_str += "|  {0}  ".format("*")
                elif (i, j) == self.observation_space.START_STATE:
                    gridworld_str += "|  {0}  ".format("S")
                elif (i, j) in self.observation_space.TERMINAL_STATES:
                    gridworld_str += "|  {0}  ".format("G")
                elif self.observation_space.WARM_HOLE_STATES and (i, j) in [state[0] for state in self.observation_space.WARM_HOLE_STATES]:
                    gridworld_str += "|  {0}  ".format("W")
                else:
                    gridworld_str += "|     "
            gridworld_str += "|\n"

            for j in range(self.WIDTH):
                gridworld_str += "|({0},{1})".format(i, j)

            gridworld_str += "|\n"

        gridworld_str += "-------------------------------\n"
        return gridworld_str


def main():
    env = GridWorld()
    env.reset()
    print("reset")
    env.render()

    done = False
    total_steps = 0
    while not done:
        total_steps += 1
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print("action: {0}, reward: {1}, done: {2}, total_steps: {3}".format(
            env.action_space.ACTION_SYMBOLS[action],
            reward, done, total_steps
        ))
        env.render()
        time.sleep(3)


def main_warm_hole():
    A_POSITION = (0, 1)  # 임의로 지정한 특별한 상태 A 좌표
    B_POSITION = (0, 3)  # 임의로 지정한 특별한 상태 B 좌표

    A_PRIME_POSITION = (4, 1)  # 상태 A에서 임의의 행동시 도착할 위치 좌표
    B_PRIME_POSITION = (2, 3)  # 상태 B에서 임의의 행동시 도착할 위치 좌표

    env = GridWorld(
        warm_hole_states=[
            (A_POSITION, A_PRIME_POSITION, 10.0),
            (B_POSITION, B_PRIME_POSITION, 5.0)
        ]
    )

    env.reset()
    print("reset")
    env.render()

    done = False
    total_steps = 0
    while not done:
        total_steps += 1
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print("action: {0}, reward: {1}, done: {2}, total_steps: {3}".format(
            env.action_space.ACTION_SYMBOLS[action],
            reward, done, total_steps
        ))
        env.render()
        time.sleep(3)


if __name__ == "__main__":
    main()
    #main_warm_hole()
