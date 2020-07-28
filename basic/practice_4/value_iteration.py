import numpy as np
from environments.gridworld import GridWorld
from utils.util import softmax

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]

DISCOUNT_RATE = 1.0
THETA_1 = 0.0001
THETA_2 = 0.0001
MAX_EPISODES = 5000


# 정책 반복 클래스
class ValueIteration:
    def __init__(self, env):
        self.env = env

        self.max_iteration = MAX_EPISODES

        self.terminal_states = [(0, 0), (4, 4)]

        self.state_values = None
        self.policy = np.empty([GRID_HEIGHT, GRID_WIDTH, self.env.action_space.num_actions])

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                for action in self.env.action_space.ACTIONS:
                    if (i, j) in TERMINAL_STATES:
                        self.policy[i][j][action] = 0.00
                    else:
                        self.policy[i][j][action] = 0.25

        # 정책 평가
        self.delta = 0.0

        # 정책 평가의 역
        self.theta = 0.001

    # 정책 평가 함수
    def value_evaluation(self):
        # 상태-가치 함수 초기화
        state_values = np.zeros((GRID_HEIGHT, GRID_WIDTH))

        # 가치 함수의 값들이 수렴할 때까지 반복
        iter_num = 0
        while True:
            old_state_values = state_values.copy()

            for i in range(GRID_HEIGHT):
                for j in range(GRID_WIDTH):
                    if (i, j) in TERMINAL_STATES:
                        state_values[i][j] = 0.0
                    else:
                        values = []
                        for action in self.env.action_space.ACTIONS:
                            (next_i, next_j), reward, prob = env.get_state_action_probability(state=(i, j), action=action)

                            # Bellman-Equation, 벨만 방정식 적용
                            values.append(
                                prob * (reward + DISCOUNT_RATE * state_values[next_i, next_j])
                            )

                        state_values[i][j] = np.max(values)

            iter_num += 1

            # 갱신되는 값이 THETA_1(=0.0001)을 기준으로 수렴하는지 판정
            max_delta_value = abs(old_state_values - state_values).max()
            if max_delta_value < THETA_1:
                break

        self.state_values = state_values

        return iter_num

    # 정책 개선 함수
    def policy_setup(self):
        # 행동-가치 함수 생성
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                if (i, j) in TERMINAL_STATES:
                    for action in self.env.action_space.ACTIONS:
                        self.policy[i][j][action] = 0.0
                else:
                    q_func = []
                    for action in self.env.action_space.ACTIONS:
                        (next_i, next_j), reward, prob = env.get_state_action_probability(state=(i, j), action=action)
                        q_func.append(
                            prob * (reward + DISCOUNT_RATE * self.state_values[next_i, next_j])
                        )

                    self.policy[i][j] = softmax(q_func)

    # 정책 반복 함수
    def start_iteration(self):
        iter_num = 0

        iter_num_policy_evaluation = self.value_evaluation()
        print("*** 가치 평가 [수렴까지 누적 반복 횟수: {0}] ***".format(iter_num_policy_evaluation))

        self.policy_setup()
        print("*** 정책 셋업 완료 ***")

        return self.state_values, self.policy


if __name__ == '__main__':
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(0, 0),
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    env.reset()

    VI = ValueIteration(env)
    VI.start_iteration()
    print(VI.state_values, end="\n\n")

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            print(i, j, VI.policy[i][j])  # UP, DOWN, LEFT, RIGHT
        print()
