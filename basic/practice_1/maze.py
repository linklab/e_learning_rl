# 미로 내 장애물 및 시작 상태, 종료 상태 정보등을 모두 지닌 미로 클래스
import time
import gym


class Maze(gym.Env):
    def __init__(self):
        # 미로의 가로 길이
        self.MAZE_WIDTH = 10

        # 미로의 세로 길이
        self.MAZE_HEIGHT = 6

        self.observation_space = gym.spaces.MultiDiscrete([self.MAZE_HEIGHT, self.MAZE_WIDTH])
        self.action_space = gym.spaces.Discrete(4)

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
        self.observation_space.START_STATE = (2, 1)

        # 종료 상태 위치
        self.observation_space.GOAL_STATES = [(5, 9)]

        # 장애물들의 위치
        self.observation_space.OBSTACLES = [
            (0, 7), (0, 8),
            (1, 4), (1, 5),
            (2, 2), (2, 4), (2, 5), (2, 8),
            (3, 2), (3, 8),
            (4, 2), (4, 7), (4, 8),
            (5, 7), (5, 8)
        ]

        # Q 가치의 크기
        self.q_size = (self.MAZE_HEIGHT, self.MAZE_WIDTH, len(self.action_space.ACTIONS))

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.current_state = None

    def reset(self):
        self.current_state = self.observation_space.START_STATE
        return self.current_state

    # take @action in @state
    # @return: [new state, reward]
    def step(self, action):
        x, y = self.current_state
        if action == self.action_space.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.action_space.ACTION_DOWN:
            x = min(x + 1, self.MAZE_HEIGHT - 1)
        elif action == self.action_space.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.action_space.ACTION_RIGHT:
            y = min(y + 1, self.MAZE_WIDTH - 1)

        if (x, y) in self.observation_space.OBSTACLES:
            x, y = self.current_state

        if (x, y) in self.observation_space.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0

        self.current_state = (x, y)

        if self.current_state in self.observation_space.GOAL_STATES:
            done = True
        else:
            done = False

        return reward, (x, y), done, None

    def render(self):
        print(self.__str__())

    def __str__(self):
        maze_str = ""
        for i in range(self.MAZE_HEIGHT):
            maze_str += "-------------------------------------------------------------\n"
            out = '| '
            for j in range(self.MAZE_WIDTH):
                if (i, j) == self.observation_space.START_STATE:
                    t = "S"
                elif (i, j) in self.observation_space.GOAL_STATES:
                    t = "G"
                elif self.current_state[0] == i and self.current_state[1] == j:
                    t = "*"
                else:
                    t = " " if (i, j) not in self.observation_space.OBSTACLES else "x"
                out += str(" {0} ".format(t)) + ' | '
            maze_str += out + "\n"

            for j in range(self.MAZE_WIDTH):
                maze_str += "|({0},{1})".format(i, j)
            maze_str += "\n"

        maze_str += "-------------------------------------------------------------\n"
        return maze_str


def main():
    env = Maze()
    env.reset()
    print("reset")
    env.render()

    done = False
    total_steps = 0
    while not done:
        total_steps += 1
        action = env.action_space.sample()
        reward, next_state, done, _ = env.step(action)
        print("action: {0}, reward: {1}, done: {2}, total_steps: {3}".format(
            env.action_space.ACTION_SYMBOLS[action],
            reward, done, total_steps
        ))
        env.render()

        time.sleep(3)


if __name__ == "__main__":
    main()