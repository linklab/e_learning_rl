# 사용 패키지 임포트
import numpy as np
from basic.practice_1.randomwalk import RandomWalk
from utils.util import draw_random_walk_policy_image

DISCOUNT_RATE = 0.9
MAX_EPISODES = 500
EPSILON = 0.1


# 비어있는 행동 가치 테이블을 0으로 초기화하며 생성함
def generate_initial_q_value_and_return(env):
    state_action_values = np.zeros((env.num_internal_states, env.NUM_ACTIONS))
    returns = dict()

    for state in env.STATES:
        for action in env.ACTIONS:
            returns[(state, action)] = list()

    return state_action_values, returns


# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for state in env.STATES:
        actions = []
        prob = []
        for action in env.ACTIONS:
            actions.append(action)
            prob.append(0.5)
        policy[state] = (actions, prob)

    return policy


# 환경에서 현재 정책에 입각하여 에피소드(현재 상태, 행동, 다음 상태, 보상) 생성
def generate_episode(env, policy):
    episode = []
    visited_state_actions = []

    state = env.reset()  # exploring start

    done = False
    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]
        next_state, reward, done, _ = env.step(action)

        episode.append(((state, action), reward))
        visited_state_actions.append((state, action))

        state = next_state

    return episode, visited_state_actions


# 첫 방문 행동 가치 MC 추정 함수
def first_visit_mc_prediction(state_action_values, returns, episode, visited_state_actions, env):
    G = 0
    for idx, ((state, action), reward) in enumerate(reversed(episode)):
        G = DISCOUNT_RATE * G + reward

        value_prediction_conditions = [
            (state, action) not in visited_state_actions[:len(visited_state_actions) - idx - 1],
            state not in env.TERMINAL_STATES
        ]

        if all(value_prediction_conditions):
            returns[(state, action)].append(G)
            state_action_values[state, action] = np.mean(returns[(state, action)])


# 소프트 탐욕적 정책 생성
def generate_soft_greedy_policy(env, state_action_values, policy):
    new_policy = dict()

    for state in env.STATES:
        actions = []
        action_probs = []
        if state in env.TERMINAL_STATES:
            for action in range(env.NUM_ACTIONS):
                actions.append(action)
                action_probs.append(0.5)
            new_policy[state] = (actions, action_probs)
        else:
            max_prob_actions = [action_ for action_, value_ in enumerate(state_action_values[state, :]) if
                                value_ == np.max(state_action_values[state, :])]
            for action in range(env.NUM_ACTIONS):
                actions.append(action)
                if action in max_prob_actions:
                    action_probs.append(
                        (1 - EPSILON) / len(max_prob_actions) + EPSILON / env.NUM_ACTIONS
                    )
                else:
                    action_probs.append(
                        EPSILON / env.NUM_ACTIONS
                    )

            new_policy[state] = (actions, action_probs)

    error = 0.0
    for i in env.STATES:
        error += np.sum(
            np.absolute(
                np.array(policy[i][1]) - np.array(new_policy[i][1])
            )
        )

    return new_policy, error


def random_walk_soft_policy_control_main():
    # 랜덤 워크 환경 객체 생성
    env = RandomWalk(
        num_internal_states=5,
        transition_reward=0.0,
        left_terminal_reward=0.0,
        right_terminal_reward=1.0
    )

    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_action_values, returns = generate_initial_q_value_and_return(env)

    # 초기 임의 정책 생성
    policy = generate_initial_random_policy(env)

    iter_num = 0

    print("[[[ MC 제어 반복 시작! ]]]")
    while iter_num < MAX_EPISODES:
        iter_num += 1

        episode, visited_state_actions = generate_episode(env, policy)
        print("*** 에피소드 생성 완료 ***")

        first_visit_mc_prediction(state_action_values, returns, episode, visited_state_actions, env)
        print("*** MC 예측 수행 완료 ***")

        policy, error = generate_soft_greedy_policy(env, state_action_values, policy)
        print("*** 정책 개선 [에러 값: {0:9.7f}], 총 반복 수: {1} ***\n".format(error, iter_num))

    print("[[[ MC 제어 반복 종료! ]]]\n\n")

    draw_random_walk_policy_image(policy, env)


if __name__ == "__main__":
    random_walk_soft_policy_control_main()
