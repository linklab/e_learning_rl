import random
import numpy as np
from basic.practice_1.gridworld import GridWorld
from utils.util import draw_grid_world_state_values_image

GRID_HEIGHT = 4
GRID_WIDTH = 4
NUM_ACTIONS = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]


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


def temporal_difference(env, policy, state_values, alpha=0.1):
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
            state_values[state] += alpha * (reward + state_values[next_state] - state_values[state])

        state = next_state


def compute_state_values(env, alpha):
    policy = generate_initial_random_policy(env)

    state_values = generate_initial_state_values()
    max_episodes = 300
    for _ in range(max_episodes):
        temporal_difference(env, policy, state_values, alpha)

    draw_grid_world_state_values_image(
        state_values, 'images/grid_world_td_prediction_300.png', GRID_HEIGHT, GRID_WIDTH
    )

    state_values = generate_initial_state_values()
    max_episodes = 3000
    for _ in range(max_episodes):
        temporal_difference(env, policy, state_values, alpha)

    draw_grid_world_state_values_image(
        state_values, 'images/grid_world_td_prediction_3000.png', GRID_HEIGHT, GRID_WIDTH
    )

    state_values = generate_initial_state_values()
    max_episodes = 10000
    for _ in range(max_episodes):
        temporal_difference(env, policy, state_values, alpha)

    draw_grid_world_state_values_image(
        state_values, 'images/grid_world_td_prediction_10000.png', GRID_HEIGHT, GRID_WIDTH
    )


if __name__ == '__main__':
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(2, 2),
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    compute_state_values(env, alpha=0.05)