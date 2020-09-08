import random
import numpy as np
from environments.tic_tac_toe import TicTacToe, ALL_STATES, PLAYER_1
import matplotlib.pyplot as plt

# 스텝 사이즈
ALPHA = 0.2

# 감가율
GAMMA = 0.9

# 탐색 확률
EPSILON = 0.2


class Dummy_Agent:
    def __init__(self, env):
        self.env = env

    def choose_action(self, state):
        state = ALL_STATES[state.identifier()]
        return random.choice(state.get_available_positions())


class DQN_Agent:
    def __init__(self, env):
        self.env = env

        self.q_table = {}
        for state_identifier in ALL_STATES:
            self.q_table[state_identifier] = {}
            for available_position in ALL_STATES[state_identifier].get_available_positions():
                self.q_table[state_identifier][available_position] = 0.0

    # epsilon-탐욕적 정책에 따른 행동 선택
    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            state = ALL_STATES[state.identifier()]
            return random.choice(state.get_available_positions())
        else:
            values_ = self.q_table[state.identifier()]
            # print([action_ for action_, value_ in values_.items() if value_ == max(list(values_.values()))])
            return random.choice([action_ for action_, value_ in values_.items() if value_ == max(list(values_.values()))])

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
                idx, ALL_STATES[state_identifier], state_identifier, self.q_table[state_identifier], list(self.q_table[state_identifier].values()))
            )

    def print_q_table_one(self, state):
        print(state.identifier(), list(self.q_table[state.identifier()].values()))


def q_learning_main(env):
    episodes = 500000

    agent_1 = DQN_Agent(env)
    agent_2 = Dummy_Agent(env)

    num_test = 100
    test_results = []

    for episode in range(episodes):
        batch_list = []
        state = env.reset()
        while True:
            action_1 = agent_1.choose_action(state, EPSILON)
            next_state_1, reward_1, done_1, _ = env.step(action_1)

            if not done_1:
                action_2 = agent_2.choose_action(next_state_1)
                next_state_2, reward_2, done_2, _ = env.step(action_2)
            else:
                next_state_2, reward_2, done_2 = None, reward_1, done_1

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
