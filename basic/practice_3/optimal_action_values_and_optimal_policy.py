import numpy as np
import os

from basic.practice_1.gridworld import GridWorld

# 이미지 저장 경로 확인 및 생성
from utils.util import draw_grid_world_action_values_image, draw_grid_world_policy_image

GRID_HEIGHT = 5
GRID_WIDTH = 5
NUM_ACTIONS = 4

DISCOUNT_RATE = 0.9      # 감쇄율

A_POSITION = (0, 1)         # 임의로 지정한 특별한 상태 A 좌표
B_POSITION = (0, 3)         # 임의로 지정한 특별한 상태 B 좌표

A_PRIME_POSITION = (4, 1)   # 상태 A에서 행동시 도착할 위치 좌표
B_PRIME_POSITION = (2, 3)   # 상태 B에서 행동시 도착할 위치 좌표


# 그리드 월드에서 최적 행동 가치 산출
def calculate_grid_world_optimal_action_values(env):
    action_value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))

    # 행동 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        # action_value_function과 동일한 형태를 가지면서 값은 모두 0인 배열을 new_action_value_function에 저장
        new_action_value_function = np.zeros_like(action_value_function)

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                values = []
                # 주어진 상태에서 가능한 모든 행동들의 결과로 다음 상태 및 보상 정보 갱신
                for action in env.action_space.ACTIONS:
                    (next_i, next_j), reward, prob = env.get_state_action_probability(state=(i, j), action=action)

                    # Bellman Optimality Equation, 벨만 최적 방정식 적용
                    # 새로운 행동 가치 갱신
                    new_action_value_function[i, j, action] = prob * (reward + DISCOUNT_RATE * np.max(action_value_function[next_i, next_j, :]))

        # 행동 가치 함수 수렴 여부 판단
        if np.sum(np.abs(new_action_value_function - action_value_function)) < 1e-4:
            break

        action_value_function = new_action_value_function

    return new_action_value_function


def calculate_optimal_policy(optimal_action_value):
    optimal_policy = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            m = np.amax(optimal_action_value[i, j, :])
            indices = np.nonzero(optimal_action_value[i, j, :] == m)[0]
            optimal_policy[(i, j)] = indices

    return optimal_policy


def main():
    # 5x5 맵 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=None,
        terminal_states=[],
        transition_reward=0,
        outward_reward=-1.0,
        warm_hole_states=[(A_POSITION, A_PRIME_POSITION, 10.0), (B_POSITION, B_PRIME_POSITION, 5.0)]
    )

    optimal_action_values = calculate_grid_world_optimal_action_values(env)

    draw_grid_world_action_values_image(
        optimal_action_values, 'images/grid_world_optimal_action_values.png',
        GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS, env.action_space.ACTION_SYMBOLS
    )

    print()

    optimal_policy = calculate_optimal_policy(optimal_action_values)
    draw_grid_world_policy_image(
        optimal_policy, "images/grid_world_optimal_policy.png",
        GRID_HEIGHT, GRID_WIDTH, env.action_space.ACTION_SYMBOLS
    )


# MAIN
if __name__ == '__main__':
    if not os.path.exists('images/'):
        os.makedirs('images/')

    main()