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

# 종료 상태를 제외한 임의의 상태에서 동일한 확률로 왼쪽 이동 또는 오른쪽 이동
LEFT_ACTION = 0
RIGHT_ACTION = 1


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
# @batch: 배치 업데이트 유무
def temporal_difference(env, policy, state_values, alpha=0.1, batch=False):
    env.reset()
    batch_list = []

    done = False
    state = env.current_state

    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        # TD 갱신 수행
        if not batch:
            if done:
                state_values[state] += alpha * (reward - state_values[state])
            else:
                state_values[state] += alpha * (reward + state_values[next_state] - state_values[state])

        batch_list.append([state, action, next_state, reward, done])

        state = next_state

    if batch:
        return batch_list


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def constant_alpha_monte_carlo(env, policy, values, alpha=0.1, batch=False):
    env.reset()
    batch_list = []

    done = False
    state = env.current_state
    return_ = None
    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        if done:
            if next_state == 'T2':
                return_ = 1.0
            elif next_state == 'T1':
                return_ = 0.0
            else:
                raise ValueError()

        batch_list.append([state, action, next_state, return_, done])
        state = next_state

    for sample in batch_list:
        if not sample[3]:
            sample[3] = return_

    if batch:
        return batch_list
    else:
        for sample in batch_list:
            state = sample[0]
            return_ = sample[3]

            # MC 갱신
            values[state] += alpha * (return_ - values[state])


# TD(0)를 활용한 상태 가치 추정
def compute_state_values(env):
    policy = generate_initial_random_policy(env)
    episodes = [3, 10, 100]
    markers = ['o', '+', 'D']
    plt.figure()
    plt.plot(['A', 'B', 'C', 'D', 'E'], VALUES, label='Initial values', linestyle=":")

    for i in range(len(episodes)):
        state_values = VALUES.copy()
        for _ in range(episodes[i]):
            temporal_difference(env, policy, state_values)
        plt.plot(['A', 'B', 'C', 'D', 'E'], state_values, label=str(episodes[i]) + ' episodes', marker=markers[i-1])

    plt.plot(['A', 'B', 'C', 'D', 'E'], TRUE_VALUES, label='True values', linestyle="--")

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
