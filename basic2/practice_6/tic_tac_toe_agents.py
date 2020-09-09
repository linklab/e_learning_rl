import random

# 스텝 사이즈
import numpy as np

from basic2.practice_5.tic_tac_toe import map_idx_to_position

ALPHA = 0.01

# 감가율
GAMMA = 1.0


class Human_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def get_action(self, current_state):
        available_positions = current_state.get_available_positions()
        valid_idx = False
        selected_position = None

        while not valid_idx:
            idx = int(input("9개 셀 중 하나를 선택하세요 (1부터 9까지의 각 셀에 매칭되는 숫자 키패드를 선택하고 엔터를 누르세요)"))
            if idx > 9 or idx < 0:
                print("[입력 오류: {0}] 1부터 9사이의 숫자 값을 입력하세요.".format(idx))
                continue

            selected_position = map_idx_to_position(idx)
            if selected_position not in available_positions:
                print("[입력 오류] 유효한 셀을 선택하세요.")
            else:
                valid_idx = True

        action = selected_position
        return action


class Q_Learning_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env
        self.q_table = self.generate_initial_q_value()
        self.policy = self.generate_initial_random_policy()

    # 비어있는 행동 가치 테이블을 0~1 사이의 임의의 값으로 초기화하며 생성함
    # q_value 테이블의 상태: state_identifier, q_value 테이블의 행동: 해당 상태에서의 available_positions
    def generate_initial_q_value(self):
        q_table = dict()
        for state_identifier in self.env.ALL_STATES:
            q_table[state_identifier] = {}
            available_positions = self.env.ALL_STATES[state_identifier].get_available_positions()
            for available_position in available_positions:
                q_table[state_identifier][available_position] = 0.0

        return q_table

    # 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
    # 초기에 각 행동의 선택 확률은 모두 같음
    def generate_initial_random_policy(self):
        policy = dict()

        for state_identifier in self.env.ALL_STATES:
            actions = []
            action_probs = []
            available_positions = self.env.ALL_STATES[state_identifier].get_available_positions()
            num_available_positions = len(available_positions)
            for available_position in available_positions:
                actions.append(available_position)
                action_probs.append(1.0 / num_available_positions)

            policy[state_identifier] = (actions, action_probs)

        return policy

    # epsilon-탐욕적 정책 갱신
    def update_epsilon_greedy_policy(self, state_identifier, epsilon):
        max_q_value = max(self.q_table[state_identifier].values())
        max_prob_actions = []
        for available_position, q_value in self.q_table[state_identifier].items():
            if max_q_value == q_value:
                max_prob_actions.append(available_position)

        actions = []
        action_probs = []
        available_positions = self.env.ALL_STATES[state_identifier].get_available_positions()
        num_available_positions = len(available_positions)

        for available_position in available_positions:
            actions.append(available_position)
            if available_position in max_prob_actions:
                action_probs.append(
                    (1 - epsilon) / len(max_prob_actions) + epsilon / num_available_positions
                )
            else:
                action_probs.append(
                    epsilon / num_available_positions
                )

        self.policy[state_identifier] = (actions, action_probs)

    def q_learning(self, state, action, next_state, reward, done, epsilon):
        # Q-러닝 갱신
        if done:
            target_value = reward
        else:
            target_value = reward + GAMMA * max(self.q_table[next_state.identifier()].values())

        self.q_table[state.identifier()][action] += ALPHA * (target_value - self.q_table[state.identifier()][action])
        self.update_epsilon_greedy_policy(state.identifier(), epsilon=epsilon)

    def get_action(self, current_state):
        actions, prob = self.policy[current_state.identifier()]

        actions_idx = [idx for idx, _ in enumerate(actions)]
        action_idx = np.random.choice(actions_idx, size=1, p=prob)[0]
        action = actions[action_idx]
        return action

    def make_greedy_policy(self):
        for state_identifier in self.env.ALL_STATES:
            if len(self.q_table[state_identifier].values()) != 0:
                self.update_epsilon_greedy_policy(state_identifier, epsilon=0.0)

    def print_q_table(self):
        for idx, state_identifier in enumerate(self.q_table):
            print("Num: {0}[{1}] {2}".format(
                idx, state_identifier, self.q_table[state_identifier]
            ))

    def print_q_table_one(self, state):
        print(state.identifier(), list(self.q_table[state.identifier()].values()))

    def num_valid_state(self):
        num_valid_state = 0
        for state_identifier in self.env.ALL_STATES:
            if len(self.q_table[state_identifier].values()) != 0.0:
                num_valid_state += 1
        return num_valid_state