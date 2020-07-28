import numpy as np
from basic.practice_1.gridworld import GridWorld
from utils.util import softmax, draw_grid_world_action_values_image, draw_grid_world_policy_image

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
        self.policy = np.empty([GRID_HEIGHT, GRID_WIDTH, self.env.action_space.NUM_ACTIONS])

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

    # 최적 정책 산출 함수
    def policy_setup(self):
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

    def calculate_optimal_policy(self):
        optimal_policy = dict()
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                max_ = np.max(self.policy[i, j, :])
                arg_max_ = np.nonzero(self.policy[i, j, :] == max_)
                indices = arg_max_[0]
                optimal_policy[(i, j)] = indices

        return optimal_policy

    def calculate_grid_world_optimal_action_values(self):
        action_value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH, self.env.action_space.NUM_ACTIONS))
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                # 주어진 상태에서 가능한 모든 행동들의 결과로 다음 상태 및 보상 정보 갱신
                for action in self.env.action_space.ACTIONS:
                    (next_i, next_j), reward, prob = self.env.get_state_action_probability(state=(i, j), action=action)

                    action_value_function[i, j, action] = \
                        prob * (reward + DISCOUNT_RATE * self.state_values[next_i, next_j])

        return action_value_function


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

    draw_grid_world_action_values_image(
        VI.calculate_grid_world_optimal_action_values(),
        'images/grid_world_vi_optimal_action_values.png',
        GRID_HEIGHT, GRID_WIDTH,
        env.action_space.NUM_ACTIONS,
        env.action_space.ACTION_SYMBOLS
    )

    draw_grid_world_policy_image(
        VI.calculate_optimal_policy(),
        "images/grid_world_vi_optimal_policy.png",
        GRID_HEIGHT, GRID_WIDTH, env.action_space.ACTION_SYMBOLS
    )
