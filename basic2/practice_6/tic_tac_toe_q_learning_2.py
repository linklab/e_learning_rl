import random
import numpy as np
from basic2.practice_5.tic_tac_toe import TicTacToe, PLAYER_1, PLAYER_2
import matplotlib.pyplot as plt

# 스텝 사이즈
ALPHA = 0.2

# 감가율
GAMMA = 0.9

# 탐색 확률
EPSILON = 0.2


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

        for state_identifier in env.ALL_STATES:
            q_table[state_identifier] = {}
            available_positions = env.ALL_STATES[state_identifier].get_available_positions()
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
            available_positions = env.ALL_STATES[state_identifier].get_available_positions()
            num_available_positions = len(available_positions)
            for available_position in available_positions:
                actions.append(available_position)
                action_probs.append(1.0 / num_available_positions)

            policy[state_identifier] = (actions, action_probs)

        return policy

    # epsilon-탐욕적 정책 갱신
    def update_epsilon_greedy_policy(self, state):
        max_q_value = np.max(self.q_table[state.identifier()].values())
        max_prob_actions = []
        for available_position, q_value in self.q_table[state.identifier()].items():
            if max_q_value == q_value:
                max_prob_actions.append(available_position)

        actions = []
        action_probs = []
        available_positions = env.ALL_STATES[state.identifier()].get_available_positions()
        num_available_positions = len(available_positions)

        for available_position in available_positions:
            actions.append(available_position)
            if available_position in max_prob_actions:
                action_probs.append(
                    (1 - EPSILON) / len(max_prob_actions) + EPSILON / num_available_positions
                )
            else:
                action_probs.append(
                    EPSILON / num_available_positions
                )

        self.policy[state.identifier()] = (actions, action_probs)

    def q_learning(self, state, action, next_state, reward, done):
        # Q-러닝 갱신
        if done:
            target_value = reward
        else:
            target_value = reward + GAMMA * max(list(self.q_table[next_state.identifier()].values()))

        self.q_table[state.identifier()][action] += ALPHA * (target_value - self.q_table[state.identifier()][action])

    def print_q_table(self):
        for idx, state_identifier in enumerate(self.q_table):
            print("Num: {0}, {1}[{2}] {3} {4}".format(
                idx, self.env.ALL_STATES[state_identifier],
                state_identifier, self.q_table[state_identifier], list(self.q_table[state_identifier].values()))
            )

    def print_q_table_one(self, state):
        print(state.identifier(), list(self.q_table[state.identifier()].values()))


def q_learning_main(env):
    episodes = 500000

    player_1 = DQN_Agent(env)
    player_2 = DQN_Agent(env)

    num_test = 100
    test_results = []

    current_player = player_1

    for episode in range(episodes):
        batch_list = []
        state = env.reset()
        done = False
        while not done:
            actions, prob = player_1.policy[state]
            action = np.random.choice(actions, size=1, p=prob)[0]
            next_state, reward, done, info = env.step(action)
            print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}, total_steps: {5}".format(
                current_player.name, action, reward, done, info, total_steps
            ))
            env.render()

            batch_list.append([state, action_1, next_state_2, reward_2, done_2])

            if done_2:
                break

            state = next_state_2

        # for _ in range(10000):
        #     for transition in batch_list:
        #         state, action, next_state, reward, done = transition
        #         agent_1.q_learning(state, action, next_state, reward, done)

        for _ in range(10):
            transition = random.choice(batch_list)
            state, action, next_state, reward, done = transition

            # print(
            #     state,
            #     list(agent_1.q_table[state.identifier()].values()),
            #     action,
            #     next_state,
            #     list(agent_1.q_table[next_state.identifier()].values()),
            #     sum(list(agent_1.q_table[next_state.identifier()].values())),
            #     reward,
            #     done
            # )
            agent_1.q_learning(state, action, next_state, reward, done)


        if episode % 1000 == 0:
            # agent_1.print_q_table()
            num_win = 0
            num_draw = 0
            for _ in range(num_test):
                episode_reward = test(agent_1, agent_2)
                if episode_reward > 0.0:
                    num_win += 1
                elif episode_reward == 0:
                    num_draw += 1
            num_lose = num_test - num_win - num_draw
            print("########################", episode, num_win, num_draw, num_lose)
            print(list(agent_1.q_table[env.INITIAL_STATE.identifier()].values()), np.argmax(list(agent_1.q_table[env.INITIAL_STATE.identifier()].values())), end="\n\n")
            test_results.append(num_win)

    plt.plot(range(len(test_results)), test_results)
    plt.show()


def test(agent_1, agent_2):
    env = TicTacToe()
    state = env.reset()

    done = False
    while not done:
        if env.current_player_int == PLAYER_1:
            action = agent_1.choose_action(state, 0.0)
        else:
            action = agent_2.choose_action(state)
        next_state, reward, done, info = env.step(action)
        #env.render()

        state = next_state

    return reward


if __name__ == '__main__':
    env = TicTacToe()

    print("INITIAL_STATE: {0}".format(env.INITIAL_STATE))
    print("NUMBER OF ALL STATES: {0}".format(len(ALL_STATES)))

    q_learning_main(env)
