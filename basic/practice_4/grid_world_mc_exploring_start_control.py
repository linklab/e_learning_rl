# 사용 패키지 임포트
import numpy as np
from basic.practice_1.gridworld import GridWorld
from utils.util import softmax, draw_grid_world_policy_image

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]
DISCOUNT_RATE = 1.0
THETA_1 = 0.0001
THETA_2 = 0.0001
MAX_EPISODES = 300


class MonteCarloControl:
    def __init__(self, env):
        self.env = env

        self.max_iteration = MAX_EPISODES

        self.terminal_states = [(0, 0), (4, 4)]

        # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
        self.state_action_values = dict()
        self.returns = dict()
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                for action in range(env.action_space.NUM_ACTIONS):
                    self.state_action_values[((i, j), action)] = 0.0
                    self.returns[((i, j), action)] = list()

        self.policy = self.generate_initial_random_policy()

    # 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
    # 초기에 각 행동의 선택 확률은 모두 같음
    def generate_initial_random_policy(self):
        policy = dict()

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                actions = []
                prob = []
                for action in range(self.env.action_space.NUM_ACTIONS):
                    actions.append(action)
                    prob.append(0.25)
                policy[(i, j)] = (actions, prob)

        return policy

    # 환경에서 현재 정책에 입각하여 에피소드(현재 상태, 행동, 다음 상태, 보상) 생성
    def generate_random_episode(self):
        episode = []
        visited_state_actions = []

        state = self.env.reset() # exploring start

        done = False
        trajectory_size = 0
        while trajectory_size < 10000 and not done:
            trajectory_size += 1
            actions, prob = self.policy[state]
            action = np.random.choice(actions, size=1, p=prob)[0]
            next_state, reward, done, _ = self.env.step(action)

            episode.append(((state, action), reward))
            visited_state_actions.append((state, action))

            state = next_state

        return episode, visited_state_actions

    # 첫 방문 행동 가치 MC 추정 함수
    def first_visit_mc_prediction(self, episode, visited_state_actions):
        G = 0
        for idx, ((state, action), reward) in enumerate(reversed(episode)):
            G = DISCOUNT_RATE * G + reward

            value_prediction_conditions = [
                (state, action) not in visited_state_actions[:len(visited_state_actions) - idx - 1],
                state not in TERMINAL_STATES
            ]

            if all(value_prediction_conditions):
                self.returns[(state, action)].append(G)
                self.state_action_values[(state, action)] = np.mean(self.returns[(state, action)])

    # 탐욕적인 정책을 생성함
    def generate_greedy_policy(self):
        new_policy = dict()

        is_policy_stable = True

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                if (i, j) in TERMINAL_STATES:
                    actions = []
                    action_probs = []
                    for action in range(self.env.action_space.num_actions):
                        actions.append(action)
                        action_probs.append(0.25)
                    new_policy[(i, j)] = (actions, action_probs)
                else:
                    actions = []
                    q_values = []
                    for action in self.env.action_space.ACTIONS:
                        actions.append(action)
                        q_values.append(self.state_action_values[((i, j), action)])

                    new_policy[(i, j)] = (actions, softmax(q_values))

        error = 0.0
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                error += np.sum(
                    np.absolute(
                        np.array(self.policy[(i, j)][1]) - np.array(new_policy[(i, j)][1])
                    )
                )

        if error > THETA_2:
            is_policy_stable = False

        self.policy = new_policy

        return is_policy_stable, error

    # 탐험적 시작 전략 기반의 몬테카를로 방법 함수
    def exploring_start_control(self):
        iter_num = 0

        print("[[[ MC 제어 반복 시작! ]]]")
        while iter_num < self.max_iteration:
            iter_num += 1

            print("*** 에피소드 생성 ***")
            episode, visited_state_actions = self.generate_random_episode()

            print("*** MC 예측 수행 ***")
            self.first_visit_mc_prediction(episode, visited_state_actions)

            _, error = self.generate_greedy_policy()
            print("*** 정책 개선 [에러 값: {0:9.7f}], 총 반복 수: {1} ***".format(error, iter_num))

            print()

        print("[[[ MC 제어 반복 종료! ]]]\n\n")


def main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=None,   # exploring start
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )

    MC = MonteCarloControl(env)
    MC.exploring_start_control()

    draw_grid_world_policy_image(
        MC.policy,
        "images/grid_world_mc_exploring_start_policy.png",
        GRID_HEIGHT, GRID_WIDTH,
        env.action_space.ACTION_SYMBOLS
    )

    # with np.printoptions(precision=2, suppress=True):
    #     for i in range(GRID_HEIGHT):
    #         for j in range(GRID_WIDTH):
    #             print(
    #                 i, j,
    #                 ": UP, DOWN, LEFT, RIGHT",
    #                 MC.policy[(i, j)][1]
    #             )
    #         print()


if __name__ == "__main__":
    main()
