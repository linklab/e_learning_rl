import numpy as np
import os
import matplotlib.pyplot as plt

from basic.practice_1.gridworld import GridWorld

# 그리드월드 높이와 너비
GRID_HEIGHT = 4
GRID_WIDTH = 12

NUM_ACTIONS = 4

# 탐색 확률
EPSILON = 0.1

# 스텝 사이즈
ALPHA = 0.5

# 감가율
GAMMA = 1

# 초기 상태와 종료 상태
START_STATE = (3, 0)
TERMINAL_STATES = [(3, 11)]
CLIFF_STATES = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]


# epsilon-탐욕적 정책에 따른 행동 선택
def choose_action(env, state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(env.action_space.ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


# @q_value: 갱신해야할 행동 가치
# @step_size: 스텝 사이즈
# @expected: 이 인수가 True이면 기대값 기반 SARSA 알고리즘 수행
# @return: 본 에피소드에서의 누적 보상
def sarsa(env, q_value, expected=False, step_size=ALPHA):
    env.reset()

    sum_of_rewards = 0.0
    state = START_STATE
    action = choose_action(env, START_STATE, q_value)
    done = False
    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = choose_action(env, next_state, q_value)
        sum_of_rewards += reward

        if not expected:
            next_q = q_value[next_state[0], next_state[1], next_action]
        else:
            # 새로운 상태에 대한 기대값 계산
            next_q = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in env.action_space.ACTIONS:
                if action_ in best_actions:
                    next_q += ((1.0 - EPSILON) / len(best_actions) + EPSILON / NUM_ACTIONS) * q_value[
                        next_state[0], next_state[1], action_]
                else:
                    next_q += EPSILON / NUM_ACTIONS * q_value[next_state[0], next_state[1], action_]

        q_value[state[0], state[1], action] += step_size * (
                    reward + GAMMA * next_q - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return sum_of_rewards


# @q_value: 갱신해야할 행동 가치
# @step_size: 스텝 사이즈
# @return: 본 에피소드에서의 누적 보상
def q_learning(env, q_value, step_size=ALPHA):
    env.reset()

    sum_of_rewards = 0.0
    state = START_STATE
    done = False
    while not done:
        action = choose_action(env, state, q_value)
        next_state, reward, done, _ = env.step(action)
        sum_of_rewards += reward

        # Q-러닝 갱신
        max_value_for_next_state = np.max(q_value[next_state[0], next_state[1], :])
        q_value[state[0], state[1], action] += step_size * (
                    reward + GAMMA * max_value_for_next_state - q_value[state[0], state[1], action])
        state = next_state
    return sum_of_rewards


# print optimal policy
def print_optimal_policy(env, q_value):
    optimal_policy = []
    for i in range(0, GRID_HEIGHT):
        optimal_policy.append([])
        for j in range(0, GRID_WIDTH):
            if (i, j) in TERMINAL_STATES:
                optimal_policy[-1].append('G')
                continue

            if (i, j) in CLIFF_STATES:
                optimal_policy[-1].append('-')
                continue

            best_action = np.argmax(q_value[i, j, :])
            if best_action == env.action_space.ACTION_UP:
                optimal_policy[-1].append('U')
            elif best_action == env.action_space.ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif best_action == env.action_space.ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif best_action == env.action_space.ACTION_RIGHT:
                optimal_policy[-1].append('R')

    for row in optimal_policy:
        print(row)


def cumulative_rewards_for_episodes(env):
    # 각 수행에서 수행하는 에피소드 개수
    episodes = 500

    # 50번의 수행
    runs = 50

    rewards_expected_sarsa = np.zeros(episodes)
    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)

    for run in range(runs):
        print("runs: {0}".format(run))
        q_table_expected_sarsa = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
        q_table_sarsa = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
        q_table_q_learning = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))

        for i in range(episodes):
            rewards_expected_sarsa[i] += sarsa(env, q_table_expected_sarsa, expected=True)
            rewards_sarsa[i] += sarsa(env, q_table_sarsa)
            rewards_q_learning[i] += q_learning(env, q_table_q_learning)

    # 50번의 수행에 대해 평균 계산
    rewards_expected_sarsa /= runs
    rewards_sarsa /= runs
    rewards_q_learning /= runs

    # 그래프 출력
    plt.plot(rewards_expected_sarsa, linestyle='-.', color='dodgerblue', label='Expected SARSA')
    plt.plot(rewards_sarsa, linestyle='-', color='darkorange', label='SARSA')
    plt.plot(rewards_q_learning, linestyle=':', color='green', label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Episode rewards')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('images/cumulative_rewards_for_episodes.png')
    plt.close()

    # display optimal policy
    print()

    print('기대값 기반 SARSA 최적 정책:')
    print_optimal_policy(env, q_table_expected_sarsa)

    print('SARSA 최적 정책:')
    print_optimal_policy(env, q_table_sarsa)

    print('Q-Learning 최적 정책:')
    print_optimal_policy(env, q_table_q_learning)


def cumulative_rewards_for_step_size(env):
    # 각 수행에서 수행하는 에피소드 개수
    episodes = 1000

    # 5번의 수행
    runs = 15

    step_sizes = np.arange(0.1, 1.1, 0.1)

    ASYMPTOTIC_EXPECTED_SARSA = 0
    ASYMPTOTIC_SARSA = 1
    ASYMPTOTIC_QLEARNING = 2

    INTERIM_EXPECTED_SARSA = 3
    INTERIM_SARSA = 4
    INTERIM_QLEARNING = 5

    methods = range(0, 6)

    performace = np.zeros((6, len(step_sizes)))

    for run in range(runs):
        print("runs: {0}".format(run))
        for idx, step_size in list(zip(range(len(step_sizes)), step_sizes)):
            q_table_expected_sarsa = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
            q_table_sarsa = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
            q_table_q_learning = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))

            for episode in range(episodes):
                expected_sarsa_reward = sarsa(env, q_table_expected_sarsa, expected=True, step_size=step_size)
                sarsa_reward = sarsa(env, q_table_sarsa, expected=False, step_size=step_size)
                q_learning_reward = q_learning(env, q_table_q_learning, step_size=step_size)

                performace[ASYMPTOTIC_EXPECTED_SARSA, idx] += expected_sarsa_reward
                performace[ASYMPTOTIC_SARSA, idx] += sarsa_reward
                performace[ASYMPTOTIC_QLEARNING, idx] += q_learning_reward

                if episode < 100:
                    performace[INTERIM_EXPECTED_SARSA, idx] += expected_sarsa_reward
                    performace[INTERIM_SARSA, idx] += sarsa_reward
                    performace[INTERIM_QLEARNING, idx] += q_learning_reward

    performace[:3, :] /= episodes * runs
    performace[3:, :] /= 100 * runs
    labels = [
        'Expected SARSA (1000 episodes)', 'SARSA (1000 episodes)', 'Q-Learning (1000 episodes)',
        'Expected SARSA (100 episodes)', 'SARSA (100 episodes)', 'Q-Learning (100 episodes)'
    ]

    for method, label in zip(methods, labels):
        if method == 0 or method == 1 or method == 2:
            linestyle = '-'
        else:
            linestyle = ':'

        if method == 0 or method == 3:
            marker = 'o'
            color = 'dodgerblue'
        elif method == 1 or method == 4:
            marker = 'x'
            color = 'darkorange'
        else:
            marker = '+'
            color = 'green'

        plt.plot(step_sizes, performace[method, :], linestyle=linestyle, color=color, marker=marker, label=label)

    plt.xlabel('Step size (alpha)')
    plt.ylabel('Episode rewards')
    plt.legend()

    plt.savefig('images/cumulative_rewards_for_step_size.png')
    plt.close()


if __name__ == '__main__':
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=START_STATE,
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0,
        warm_hole_states=[(s, START_STATE, -100.0) for s in CLIFF_STATES]
    )
    cumulative_rewards_for_episodes(env)
    cumulative_rewards_for_step_size(env)

