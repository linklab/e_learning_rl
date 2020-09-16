import random
import numpy as np
from basic.practice_1.gridworld import GridWorld
import matplotlib.pyplot as plt

GRID_HEIGHT = 4
GRID_WIDTH = 4
NUM_ACTIONS = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]

MAX_EPISODES = 5000

VALUES = np.zeros(shape=(GRID_HEIGHT, GRID_WIDTH))
for i in range(GRID_HEIGHT):
    for j in range(GRID_WIDTH):
        VALUES[i, j] = 0.0

# 올바른 상태 가치 값 저장
TRUE_VALUES = np.zeros(shape=(GRID_HEIGHT, GRID_WIDTH))
TRUE_VALUES[0, 0] = 0.0;   TRUE_VALUES[0, 1] = -14.0; TRUE_VALUES[0, 2] = -20.0; TRUE_VALUES[0, 3] = -22.0
TRUE_VALUES[1, 0] = -14.0; TRUE_VALUES[1, 1] = -18.0; TRUE_VALUES[1, 2] = -20.0; TRUE_VALUES[1, 3] = -20.0
TRUE_VALUES[2, 0] = -20.0; TRUE_VALUES[2, 1] = -20.0; TRUE_VALUES[2, 2] = -18.0; TRUE_VALUES[2, 3] = -14.0
TRUE_VALUES[3, 0] = -22.0; TRUE_VALUES[3, 1] = -20.0; TRUE_VALUES[3, 2] = -14.0; TRUE_VALUES[3, 3] = 0.0


def generate_initial_state_values():
    state_values = np.zeros(shape=(GRID_HEIGHT, GRID_WIDTH))
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            state_values[i, j] = 0.0
    return state_values


# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            actions = []
            prob = []
            for action in range(NUM_ACTIONS):
                actions.append(action)
                prob.append(0.25)
            policy[(i, j)] = (actions, prob)

    return policy


def get_exploring_start_state():
    while True:
        i = random.randrange(GRID_HEIGHT)
        j = random.randrange(GRID_WIDTH)
        if (i, j) not in TERMINAL_STATES:
            break
    return (i, j)


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def temporal_difference(env, policy, state_values, alpha, gamma=1.0):
    env.reset()

    initial_state = get_exploring_start_state()
    env.moveto(initial_state)

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


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def temporal_difference_batch(env, policy, state_values, alpha=0.1, gamma=1.0, num_batch_updates=10):
    env.reset()

    initial_state = get_exploring_start_state()
    env.moveto(initial_state)

    batch_list = []

    done = False
    state = env.current_state
    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        batch_list.append([state, action, next_state, reward, done])

        state = next_state

    for i in range(num_batch_updates):
        for sample in batch_list:
            state = sample[0]
            next_state = sample[2]
            reward = sample[3]
            done = sample[4]

            if done:
                state_values[state] += alpha * (reward - state_values[state])
            else:
                state_values[state] += alpha * (reward + gamma * state_values[next_state] - state_values[state])


# @method: 'TD(0)' 또는 'TD(0)_batch'
def batch_updating(env, method, alpha):
    policy = generate_initial_random_policy(env)

    # episodes 수행 마다 독립적인 에러값 모음
    total_errors = np.zeros(MAX_EPISODES)

    total_runs = 10
    for run in range(total_runs):
        errors = []
        state_values = np.copy(VALUES)

        for _ in range(MAX_EPISODES):
            if method == 'TD(0)':
                temporal_difference(env, policy, state_values, alpha=alpha)
            else:
                temporal_difference_batch(env, policy, state_values, alpha=alpha, num_batch_updates=10)
            errors.append(np.sqrt(np.sum(np.power(TRUE_VALUES - state_values, 2)) / 25))
        total_errors += np.asarray(errors)
        print("method: {0}, run: {1}".format(method, run))
    total_errors /= total_runs

    return total_errors


def mc_td_batch_comparison(env, alpha):
    td_errors = batch_updating(env, 'TD(0)', alpha=alpha)
    td_batch_errors = batch_updating(env, 'TD(0)_batch', alpha=alpha)

    plt.plot(td_errors, label='TD(0), alpha={0}'.format(alpha))
    plt.plot(td_batch_errors, label='TD(0)_batch, alpha={0}'.format(alpha))
    plt.xlabel('Episodes')
    plt.ylabel('RMS error')
    plt.legend()

    plt.savefig('images/grid_world_td_batch_alpha_{0}.png'.format(alpha))
    plt.close()


def td_batch_comparison_main():
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(2, 2),
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    print("alpha={0}".format(0.01))
    mc_td_batch_comparison(env, alpha=0.01)

    print("alpha={0}".format(0.005))
    mc_td_batch_comparison(env, alpha=0.005)


if __name__ == '__main__':
    td_batch_comparison_main()