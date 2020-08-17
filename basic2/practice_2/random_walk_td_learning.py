# 사용 패키지 임포트
import numpy as np
from basic.practice_1.randomwalk import RandomWalk
from utils.util import draw_random_walk_policy_image
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


# TD(0)를 활용한 상태 가치 추정
def compute_state_values(env):
    episodes = [3, 10, 100]
    markers = ['o', '+', 'D']
    plt.figure()
    plt.plot(
        ['A', 'B', 'C', 'D', 'E'],
        VALUES, label='Initial values', linestyle=":"
    )

    policy = generate_initial_random_policy(env)
    for i in range(len(episodes)):
        state_values = VALUES.copy()
        for _ in range(episodes[i]):
            temporal_difference(env, policy, state_values)
        plt.plot(
            ['A', 'B', 'C', 'D', 'E'],
            state_values, label=str(episodes[i]) + ' episodes', marker=markers[i-1]
        )

    plt.plot(
        ['A', 'B', 'C', 'D', 'E'],
        TRUE_VALUES, label='True values', linestyle="--"
    )

    plt.xlabel('States')
    plt.ylabel('Predicted values')
    plt.legend()
    plt.savefig('images/random_walk_td_prediction.png')
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
    compute_state_values(env)


if __name__ == '__main__':
    main()
