import os
import random
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from basic.practice_1.maze import Maze

# 감가율
from utils.util import draw_q_value_image_for_maze

GAMMA = 0.95

# 탐색(exploration) 확률
EPSILON = 0.1

# 스텝 사이즈
ALPHA = 0.1

# 계획에서의 수행 스텝 수
MAX_PLANNING_STEPS = 5


# Dyna-Q의 계획 과정에서 사용하는 간단한 모델
class EnvModel:
    def __init__(self):
        self.model = dict()

    # 경험 샘플 저장
    def store(self, state, action, reward, next_state):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if state not in self.model.keys():
            self.model[state] = dict()

        self.model[state][action] = [reward, next_state]

    # 저장해 둔 경험 샘플들에서 임으로 선택하여 반환험
    def sample(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        reward, next_state = self.model[state][action]
        return state, action, reward, next_state


# 비어있는 행동 가치 함수를 0~1 사이의 임의의 값으로 초기화하며 생성함
def generate_initial_q_value(env):
    q_table = np.zeros((env.MAZE_HEIGHT, env.MAZE_WIDTH, env.action_space.NUM_ACTIONS))

    for i in range(env.MAZE_HEIGHT):
        for j in range(env.MAZE_WIDTH):
            if (i, j) not in env.observation_space.GOAL_STATES:
                for action in env.action_space.ACTIONS:
                    q_table[i, j, action] = random.random()
    return q_table


# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for i in range(env.MAZE_HEIGHT):
        for j in range(env.MAZE_WIDTH):
            if (i, j) not in env.observation_space.GOAL_STATES:
                actions = []
                prob = []
                for action in env.action_space.ACTIONS:
                    actions.append(action)
                    prob.append(0.25)

                policy[(i, j)] = (actions, prob)

    return policy


# epsilon-탐욕적 정책 갱신
def update_epsilon_greedy_policy(env, state, q_value, policy):
    max_prob_actions = [action_ for action_, value_ in enumerate(q_value[state[0], state[1], :]) if
                        value_ == np.max(q_value[state[0], state[1], :])]

    actions = []
    action_probs = []
    for action in env.action_space.ACTIONS:
        actions.append(action)
        if action in max_prob_actions:
            action_probs.append(
                (1 - EPSILON) / len(max_prob_actions) + EPSILON / env.action_space.NUM_ACTIONS
            )
        else:
            action_probs.append(
                EPSILON / env.action_space.NUM_ACTIONS
            )

    policy[state] = (actions, action_probs)


# Dyna-Q 알고리즘의 각 에피소드 별 학습
# @q_table: 행동 가치 테이블, dyna_q 함수 수행 후 값이 갱신 됨
# @model: 계획시 사용할 모델
# @env: 미로 환경
def dyna_q(q_table, policy, env_model, env, planning_step):
    state = env.reset()
    steps = 0
    rewards = 0.0

    done = False

    while not done:
        # 타임 스텝 기록
        steps += 1

        # 행동 얻어오기
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]

        # 행동 수행후
        reward, next_state, done, _ = env.step(action)

        # Q-러닝 갱신
        target = reward + GAMMA * np.max(q_table[next_state[0], next_state[1], :])
        q_table[state[0], state[1], action] += ALPHA * (target - q_table[state[0], state[1], action])

        # 경험 샘플을 모델에 저장 (모델 구성)
        env_model.store(state, action, reward, next_state)

        # 모델로 부터 샘플 얻어오면서 Q-계획 반복 수행
        for t in range(planning_step):
            state_, action_, reward_, next_state_ = env_model.sample()
            target = reward_ + GAMMA * np.max(q_table[next_state_[0], next_state_[1], :])
            q_table[state_[0], state_[1], action_] += ALPHA * (target - q_table[state_[0], state_[1], action_])

        update_epsilon_greedy_policy(env, state, q_table, policy)
        state = next_state
        rewards += reward

        # 최대 스텝 체크
        if steps > env.max_steps:
            break

    return steps, rewards


def maze_dyna_q(env):
    # 총 수행 횟수 (성능에 대한 평균을 구하기 위함)
    runs = 10

    episodes = 50
    planning_steps = [0, 3, 30]
    steps = np.zeros((len(planning_steps), episodes))

    for run in range(runs):
        for i, planning_step in enumerate(planning_steps):
            # 행동 가치 저장
            q_table = generate_initial_q_value(env)

            # Dyna-Q를 위한 환경 모델 생성
            env_model = EnvModel()

            policy = generate_initial_random_policy(env)

            for episode in range(episodes):
                steps_, rewards_ = dyna_q(q_table, policy, env_model, env, planning_step)
                steps[i, episode] += steps_

    # 총 수행 횟수에 대한 평균 값 산출
    steps /= runs

    linestyles = ['-', '--', ':']
    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], linestyle=linestyles[i], label='Number of planning steps: {0}'.format(planning_steps[i]))
    plt.xlabel('Episode')
    plt.ylabel('Executed steps per episode')
    plt.legend()
    plt.savefig('images/maze_dyna_q.png')
    plt.close()


if __name__ == '__main__':
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    env = Maze()  # 미로 환경 객체 구성
    env.reset()
    print(env)

    maze_dyna_q(env)
