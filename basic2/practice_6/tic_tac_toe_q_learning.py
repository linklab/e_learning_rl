import random
import time

import numpy as np
from basic2.practice_5.tic_tac_toe import TicTacToe, PLAYER_1, PLAYER_2

# 스텝 사이즈
ALPHA = 0.2

# 감가율
GAMMA = 0.9

# 탐색 확률
INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.00
LAST_SCHEDULED_EPISODES = 10000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 50000


def epsilon_scheduled(current_episode):
    fraction = min(current_episode / LAST_SCHEDULED_EPISODES, 1.0)
    epsilon = min(INITIAL_EPSILON + fraction * (FINAL_EPSILON - INITIAL_EPSILON), INITIAL_EPSILON)
    return epsilon


class DQN_Agent:
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
                q_table[state_identifier][available_position] = random.random()

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
    def update_epsilon_greedy_policy(self, state, epsilon):
        max_q_value = max(self.q_table[state.identifier()].values())
        max_prob_actions = []
        for available_position, q_value in self.q_table[state.identifier()].items():
            if max_q_value == q_value:
                max_prob_actions.append(available_position)

        actions = []
        action_probs = []
        available_positions = self.env.ALL_STATES[state.identifier()].get_available_positions()
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

        self.policy[state.identifier()] = (actions, action_probs)

    def q_learning(self, state, action, next_state, reward, done, epsilon):
        # Q-러닝 갱신
        if done:
            target_value = reward
        else:
            target_value = reward + GAMMA * max(self.q_table[next_state.identifier()].values())

        self.q_table[state.identifier()][action] += ALPHA * (target_value - self.q_table[state.identifier()][action])
        self.update_epsilon_greedy_policy(state, epsilon)

    def print_q_table(self):
        for idx, state_identifier in enumerate(self.q_table):
            print("Num: {0}, {1}[{2}] {3} {4}".format(
                idx, self.env.ALL_STATES[state_identifier],
                state_identifier, self.q_table[state_identifier], list(self.q_table[state_identifier].values()))
            )

    def print_q_table_one(self, state):
        print(state.identifier(), list(self.q_table[state.identifier()].values()))


def q_learning_main(render):
    env = TicTacToe()

    player_1 = DQN_Agent(name="PLAYER_1", env=env)
    player_2 = DQN_Agent(name="PLAYER_2", env=env)

    total_steps = 0
    NUM_PLAYER_1_WIN = 0
    NUM_PLAYER_2_WIN = 0
    NUM_DRAW = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()
        current_player = player_1

        epsilon = epsilon_scheduled(episode)

        print("GAME: {0}, EPSILON: {1:.2f}".format(episode, epsilon))
        if render:
            env.render()

        done = False
        while not done:
            total_steps += 1
            actions, prob = current_player.policy[state.identifier()]

            actions_idx = [idx for idx, _ in enumerate(actions)]
            action_idx = np.random.choice(actions_idx, size=1, p=prob)[0]
            action = actions[action_idx]

            next_state, reward, done, info = env.step(action)
            # print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}, total_steps: {5}".format(
            #     current_player.name, action, reward, done, info, total_steps
            # ))
            if render:
                env.render()

            # batch_list.append([state, action, next_state, reward, done])
            #
            # for _ in range(10):
            #     transition = random.choice(batch_list)
            #     state, action, next_state, reward, done = transition
            #     current_player.q_learning(state, action, next_state, reward, done)

            current_player.q_learning(state, action, next_state, reward, done, epsilon)

            if done:
                if info['winner'] == 1:
                    NUM_PLAYER_1_WIN += 1
                elif info['winner'] == -1:
                    NUM_PLAYER_2_WIN += 1
                else:
                    NUM_DRAW += 1
                print("NUM_PLAYER_1_WIN: {0} ({1:.2f}), NUM_PLAYER_2_WIN: {2} ({3:.2f}), NUM_DRAW: {4}\n".format(
                    NUM_PLAYER_1_WIN, NUM_PLAYER_1_WIN / (episode + 1),
                    NUM_PLAYER_2_WIN, NUM_PLAYER_2_WIN / (episode + 1),
                    NUM_DRAW
                ))
            else:
                state = next_state

            if current_player == player_1:
                current_player = player_2
            else:
                current_player = player_1


if __name__ == '__main__':
    render = False
    q_learning_main(render)
