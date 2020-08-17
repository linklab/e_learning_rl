# 사용 패키지 임포트
import numpy as np
from basic.practice_1.randomwalk import RandomWalk
from basic2.practice_1.grid_world_mc_state_value_prediction import generate_random_episode
import matplotlib.pyplot as plt
import os

NUM_INTERNAL_STATES = 5

# 0: 왼쪽 종료 상태 T1를 나타냄, 상태 가치는 0.0으로 변하지 않음
# 6: 오른쪽 종료 상태 T2를 나타냄, 상태 가치는 1.0으로 변하지 않음
# 1부터 5는 각각 차례로 상태 A부터 상태 E를 나타냄, 각 상태 가치는 0.5로 초기화됨
VALUES = np.zeros(NUM_INTERNAL_STATES)
VALUES[0:NUM_INTERNAL_STATES] = 0.5

# 올바른 상태 가치 값 저장
TRUE_VALUES = np.zeros(NUM_INTERNAL_STATES)
TRUE_VALUES[0:NUM_INTERNAL_STATES] = np.arange(1, 6) / 6.0


# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for state in env.observation_space.STATES:
        actions = []
        prob = []
        for action in range(env.action_space.NUM_ACTIONS):
            actions.append(action)
            prob.append(0.5)
        policy[state] = (actions, prob)

    return policy


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
def temporal_difference(env, policy, state_values, alpha=0.1, gamma=1.0):
    env.reset()

    done = False
    state = env.current_state

    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        if done:
            state_values[state] += alpha * (reward - state_values[state])
        else:
            state_values[state] += alpha * (reward + gamma * state_values[next_state] - state_values[state])

        state = next_state


# 환경에서 무작위로 에피소드 생성
def generate_random_episode(env, policy):
    episode = []
    visited_states = []

    env.reset()
    state = env.current_state
    done = False

    while not done:
        # 상태에 관계없이 항상 4가지 행동 중 하나를 선택하여 수행
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        episode.append((state, reward))
        visited_states.append(state)

        state = next_state

    return episode, visited_states


# 첫 방문 행동 가치 MC 예측
def first_visit_mc_prediction(env, policy, values, alpha=0.1, gamma=1.0):
    returns = dict()
    for state in range(NUM_INTERNAL_STATES):
        returns[state] = list()

    episode, visited_states = generate_random_episode(env, policy)

    G = 0
    for idx, (state, reward) in enumerate(reversed(episode)):
        G = gamma * G + reward

        state_value_prediction_conditions = [
            state not in visited_states[:len(visited_states) - idx - 1],
            state not in env.observation_space.TERMINAL_STATES
        ]

        if all(state_value_prediction_conditions):
            returns[state].append(G)
            values[state] += alpha * (returns[state] - values[state])


# TD(0)와 MC의 상태 가치 예측 성능 비교
def mc_td_comparison(env):
    # 두 가지 방법에 동일한 스텝 사이즈 적용
    td_alphas = [0.15, 0.1, 0.05, 0.025]
    mc_alphas = [0.15, 0.1, 0.05, 0.025]
    # mc_alphas = [0.01, 0.02, 0.03, 0.04]
    total_runs = 100
    episodes = 200
    plt.figure()

    policy = generate_initial_random_policy(env)
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD(0)'
            linestyle = '-'
        else:
            method = 'MC'
            linestyle = '-.'

        for _ in range(total_runs):
            errors = []
            state_values = np.copy(VALUES)

            for i in range(episodes):
                if method == 'TD(0)':
                    temporal_difference(env, policy, state_values, alpha=alpha)
                else:
                    first_visit_mc_prediction(env, policy, state_values, alpha=alpha)
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUES - state_values, 2)) / NUM_INTERNAL_STATES))
            total_errors += np.asarray(errors)
        total_errors /= total_runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = {0:.2f}'.format(alpha))

    plt.xlabel('Episodes')
    plt.ylabel('RMS error')
    plt.legend()
    plt.savefig('images/random_walk_mc_td_comparison.png')
    plt.close()


def main():
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    env = RandomWalk(
        num_internal_states=NUM_INTERNAL_STATES,
        transition_reward=0.0,
        left_terminal_reward=0.0,
        right_terminal_reward=1.0
    )
    mc_td_comparison(env)


if __name__ == '__main__':
    main()
