# 스텝 사이즈
import numpy as np

ALPHA = 0.1

# 감가율
GAMMA = 0.99


class Human_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def get_action(self, current_state):
        available_actions_ids = current_state.get_available_actions()
        valid_action_id = False
        action_id = None

        while not valid_action_id:
            action_id = int(input("9개 셀 중 하나를 선택하세요 (보드 상단부터 숫자 키패드와 매칭하여 [7,8,9, 4,5,6, 1,2,3] 숫자 중 하나를 선택하고 엔터를 누르세요)"))
            if action_id > 9 or action_id < 0:
                print("[입력 오류: {0}] 1부터 9사이의 숫자 값을 입력하세요.".format(action_id))
                continue

            if action_id not in available_actions_ids:
                print("[입력 오류: {0}] 유효한 셀을 선택하세요.".format(action_id))
            else:
                valid_action_id = True

        return action_id


class Q_Learning_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env
        self.q_table = self.generate_initial_q_value()
        self.policy = self.generate_initial_random_policy()
        self.count_state_updates = {}
        for state_id in env.ALL_STATES:
            self.count_state_updates[state_id] = 0

    # 비어있는 행동 가치 테이블을 0.0 값으로 초기화하며 생성함
    # q_value 테이블의 상태: state_id, q_value 테이블의 행동: 해당 상태에서의 available_action_id
    def generate_initial_q_value(self):
        q_table = dict()
        for state_id in self.env.ALL_STATES:
            q_table[state_id] = {}
            available_action_ids = self.env.ALL_STATES[state_id].get_available_actions()
            for available_action_id in available_action_ids:
                q_table[state_id][available_action_id] = 0.0

        return q_table

    def avg_q_value(self):
        q_value_list = []

        for state_id in self.env.ALL_STATES:
            available_action_ids = self.env.ALL_STATES[state_id].get_available_actions()
            for available_action_id in available_action_ids:
                q_value_list.append(self.q_table[state_id][available_action_id])

        return np.mean(q_value_list)

    # 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성
    # 초기에 각 상태에서 수행할 수 있는 행동의 선택 확률은 모두 같음
    def generate_initial_random_policy(self):
        policy = dict()

        for state_id in self.env.ALL_STATES:
            action_ids = []
            action_probs = []
            available_action_ids = self.env.ALL_STATES[state_id].get_available_actions()
            num_available_positions = len(available_action_ids)
            for available_position in available_action_ids:
                action_ids.append(available_position)
                action_probs.append(1.0 / num_available_positions)

            policy[state_id] = (action_ids, action_probs)

        return policy

    # epsilon-탐욕적 정책 갱신
    def update_epsilon_greedy_policy(self, state_id, epsilon):
        # Q값이 가장 큰 행동 선별
        max_q_value = max(self.q_table[state_id].values())
        max_prob_actions = []
        for available_position, q_value in self.q_table[state_id].items():
            if max_q_value == q_value:
                max_prob_actions.append(available_position)

        action_ids = []
        action_probs = []

        # 현재 상태에서 수행가능한 모든 행동 취합
        available_action_ids = self.env.ALL_STATES[state_id].get_available_actions()
        num_available_actions = len(available_action_ids)

        # 각 수행가능한 행동별로 epsilon-탐욕적 정책 갱신
        for available_position in available_action_ids:
            action_ids.append(available_position)
            if available_position in max_prob_actions:
                action_probs.append(
                    (1 - epsilon) / len(max_prob_actions) + epsilon / num_available_actions
                )
            else:
                action_probs.append(
                    epsilon / num_available_actions
                )

        self.policy[state_id] = (action_ids, action_probs)

    def q_learning(self, state, action_id, next_state, reward, done, epsilon):
        # 각 상태별로 갱신 횟수 관리
        self.count_state_updates[state.identifier()] += 1

        # Q-러닝 갱신
        if done:
            target_value = reward
        else:
            target_value = reward + GAMMA * max(self.q_table[next_state.identifier()].values())

        td_error = target_value - self.q_table[state.identifier()][action_id]
        self.q_table[state.identifier()][action_id] += ALPHA * td_error
        self.update_epsilon_greedy_policy(state.identifier(), epsilon=epsilon)
        return td_error

    def get_action(self, current_state):
        action_ids, action_probs = self.policy[current_state.identifier()]
        action_id = np.random.choice(action_ids, size=1, p=action_probs)[0]
        return action_id

    def make_greedy_policy(self):
        for state_id in self.env.ALL_STATES:
            if len(self.q_table[state_id].values()) != 0:
                self.update_epsilon_greedy_policy(state_id, epsilon=0.0)

    def print_q_table(self):
        for idx, state_id in enumerate(self.q_table):
            print("Num: {0}[{1}] {2}".format(
                idx, state_id, self.q_table[state_id]
            ))

    def get_q_values_for_one_state(self, state):
        state_q_values = self.q_table[state.identifier()]
        state_q_value_list = []
        for action_id, q_value in state_q_values.items():
            state_q_value_list.append("{0}:{1:.3f}".format(action_id, q_value))

        return ", ".join(state_q_value_list)

    def get_policy_for_one_state(self, state):
        action_ids, action_probs = self.policy[state.identifier()]
        state_policy_list = []
        for idx, action_id in enumerate(action_ids):
            state_policy_list.append("{0}:{1:.3f}".format(action_id, action_probs[idx]))

        return ", ".join(state_policy_list)