# 사용 패키지 임포트
import numpy as np
from basic.practice_1.gridworld import GridWorld
from utils.util import softmax, draw_grid_world_policy_image
import random

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]
DISCOUNT_RATE = 1.0
MAX_EPISODES = 100


def get_exploring_start_state():
    while True:
        i = random.randrange(GRID_HEIGHT)
        j = random.randrange(GRID_WIDTH)
        if (i, j) not in TERMINAL_STATES:
            break
    return (i, j)


# 비어있는 행동 가치 함수를 0으로 초기화하며 생성함
def generate_initial_q_value_and_return(env):
    state_action_values = dict()
    returns = dict()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            for action in env.action_space.ACTIONS:
                state_action_values[((i, j), action)] = 0.0
                returns[((i, j), action)] = list()

    return state_action_values, returns

# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            actions = []
            prob = []
            for action in env.action_space.ACTIONS:
                actions.append(action)
                prob.append(0.25)
            policy[(i, j)] = (actions, prob)

    return policy


# 환경에서 현재 정책에 입각하여 에피소드(현재 상태, 행동, 다음 상태, 보상) 생성
def generate_random_episode(env, policy):
    episode = []
    visited_state_actions = []

    state = get_exploring_start_state() # exploring start
    env.moveto(state)

    done = False
    trajectory_size = 0
    while trajectory_size < 10000 and not done:
        trajectory_size += 1
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        episode.append(((state, action), reward))
        visited_state_actions.append((state, action))

        state = next_state

    return episode, visited_state_actions


# 첫 방문 행동 가치 MC 추정 함수
def first_visit_mc_prediction(state_action_values, returns, episode, visited_state_actions):
    G = 0
    for idx, ((state, action), reward) in enumerate(reversed(episode)):
        G = DISCOUNT_RATE * G + reward

        value_prediction_conditions = [
            (state, action) not in visited_state_actions[:len(visited_state_actions) - idx - 1],
            state not in TERMINAL_STATES
        ]

        if all(value_prediction_conditions):
            returns[(state, action)].append(G)
            state_action_values[(state, action)] = np.mean(returns[(state, action)])


# 탐욕적인 정책 생성
def generate_greedy_policy(env, state_action_values, policy):
    new_policy = dict()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            actions = []
            action_probs = []
            if (i, j) in TERMINAL_STATES:
                for action in env.action_space.ACTIONS:
                    actions.append(action)
                    action_probs.append(0.25)
                new_policy[(i, j)] = (actions, action_probs)
            else:
                for action in env.action_space.ACTIONS:
                    actions.append(action)
                    action_probs.append(state_action_values[((i, j), action)])

                new_policy[(i, j)] = (actions, softmax(action_probs))

    error = 0.0
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            error += np.sum(
                np.absolute(
                    np.array(policy[(i, j)][1]) - np.array(new_policy[(i, j)][1])
                )
            )

    return new_policy, error


def main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=None,   # exploring start
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )

    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_action_values, returns = generate_initial_q_value_and_return(env)

    # 초기 임의 정책 생성
    policy = generate_initial_random_policy(env)

    iter_num = 0

    print("[[[ MC 제어 반복 시작! ]]]")
    while iter_num < MAX_EPISODES:
        iter_num += 1

        episode, visited_state_actions = generate_random_episode(env, policy)
        print("*** 에피소드 생성 완료 ***")

        first_visit_mc_prediction(state_action_values, returns, episode, visited_state_actions)
        print("*** MC 예측 수행 완료 ***")

        policy, error = generate_greedy_policy(env, state_action_values, policy)
        print("*** 정책 개선 [에러 값: {0:9.7f}], 총 반복 수: {1} ***\n".format(error, iter_num))

    print("[[[ MC 제어 반복 종료! ]]]\n\n")

    draw_grid_world_policy_image(
        policy,
        "images/grid_world_mc_exploring_start_policy.png",
        GRID_HEIGHT, GRID_WIDTH,
        env.action_space.ACTION_SYMBOLS
    )


if __name__ == "__main__":
    main()
