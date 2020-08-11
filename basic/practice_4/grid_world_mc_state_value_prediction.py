# 사용 패키지 임포트
import os
import numpy as np
import random
from basic.practice_1.gridworld import GridWorld
from utils.util import draw_grid_world_state_values_image

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]


def get_exploring_start_state():
    while True:
        i = random.randrange(GRID_HEIGHT)
        j = random.randrange(GRID_WIDTH)
        if (i, j) not in TERMINAL_STATES:
            break
    return (i, j)


# 환경에서 무작위로 에피소드 생성
def generate_random_episode(env):
    episode = []
    visited_states = []

    initial_state = get_exploring_start_state()
    env.moveto(initial_state)

    state = initial_state
    done = False
    while not done:
        # 상태에 관계없이 항상 4가지 행동 중 하나를 선택하여 수행
        action = random.randrange(env.action_space.NUM_ACTIONS)

        next_state, reward, done, _ = env.step(action)

        episode.append((state, reward))
        visited_states.append(state)

        state = next_state

    return episode, visited_states


# 첫 방문 행동 가치 MC 추정 함수
def first_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_values = np.zeros(shape=(GRID_HEIGHT, GRID_WIDTH))
    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            state_values[i, j] = 0.0
            returns[(i, j)] = list()

    for i in range(num_iter):
        episode, visited_states = generate_random_episode(env)

        G = 0
        for idx, (state, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            state_value_prediction_conditions = [
                state not in visited_states[:len(visited_states) - idx - 1],
                state not in TERMINAL_STATES
            ]

            if all(state_value_prediction_conditions):
                returns[state].append(G)
                state_values[state] = np.mean(returns[state])

    return state_values, returns


# 모든 방문 행동 가치 MC 예측
def every_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_values = np.zeros(shape=(GRID_HEIGHT, GRID_WIDTH))
    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            state_values[i, j] = 0.0
            returns[(i, j)] = list()

    for i in range(num_iter):
        episode, _ = generate_random_episode(env)

        G = 0
        for idx, (state, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            state_value_prediction_conditions = [
                state not in TERMINAL_STATES
            ]

            if all(state_value_prediction_conditions):
                returns[state].append(G)
                state_values[state] = np.mean(returns[state])

    return state_values, returns


def main():
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(0, 0),
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    env.reset()

    state_values, returns = first_visit_mc_prediction(env, 1.0, 10000)
    print("First Visit")
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            print("({0}, {1}): {2:5.2f}".format(i, j, state_values[i, j]))
        print()

    draw_grid_world_state_values_image(
        state_values,
        'images/grid_world_mc_state_values_first_visit.png',
        GRID_HEIGHT, GRID_WIDTH
    )
    print()

    state_values, returns = every_visit_mc_prediction(env, 1.0, 10000)
    print("Every Visit")
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            print("({0}, {1}): {2:5.2f}".format(i, j, state_values[i, j]))
        print()

    draw_grid_world_state_values_image(
        state_values,
        'images/grid_world_mc_state_values_every_visit.png',
        GRID_HEIGHT, GRID_WIDTH
    )


if __name__ == "__main__":
    main()
