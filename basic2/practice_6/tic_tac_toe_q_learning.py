import pickle
import random
from collections import deque

import numpy as np
from basic2.practice_5.tic_tac_toe import TicTacToe, PLAYER_1, PLAYER_2, Dummy_Agent

# 탐색 확률
from basic2.practice_6.tic_tac_toe_agents import DQN_Agent, Human_Agent

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 50000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 100000

NUM_PLAYER_1_WIN = 0
NUM_PLAYER_2_WIN = 0
NUM_DRAW = 0

def epsilon_scheduled(current_episode):
    fraction = min(current_episode / LAST_SCHEDULED_EPISODES, 1.0)
    epsilon = min(INITIAL_EPSILON + fraction * (FINAL_EPSILON - INITIAL_EPSILON), INITIAL_EPSILON)
    return epsilon

def print_done_stat(info, episode):
    global NUM_PLAYER_1_WIN, NUM_PLAYER_2_WIN, NUM_DRAW
    if info['winner'] == 1:
        NUM_PLAYER_1_WIN += 1
    elif info['winner'] == -1:
        NUM_PLAYER_2_WIN += 1
    else:
        NUM_DRAW += 1
    print("NUM_PLAYER_1_WIN: {0} ({1:.2f}), NUM_PLAYER_2_WIN: {2} ({3:.2f}), NUM_DRAW: {4} ({5:.2f})\n".format(
        NUM_PLAYER_1_WIN, NUM_PLAYER_1_WIN / (episode + 1),
        NUM_PLAYER_2_WIN, NUM_PLAYER_2_WIN / (episode + 1),
        NUM_DRAW, NUM_DRAW / (episode + 1)
    ))


def q_learning_main_1(step_info, render):
    env = TicTacToe()

    player_1 = DQN_Agent(name="PLAYER_1", env=env)
    player_2 = Dummy_Agent(name="PLAYER_2", env=env)

    batch_list_1 = deque(maxlen=10000)

    total_steps = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()

        epsilon = epsilon_scheduled(episode)

        print("GAME: {0}, EPSILON: {1:.2f}".format(episode, epsilon))
        if render:
            env.render()

        done = False
        while not done:
            total_steps += 1
            action_1 = player_1.get_action(state)

            next_state_1, reward_1, done_1, info_1 = env.step(action_1)
            if step_info:
                print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}, total_steps: {5}".format(
                    player_1.name, action_1, reward_1, done_1, info_1, total_steps
                ))
            if render:
                env.render()

            if done_1:
                print_done_stat(info_1, episode)
                done = done_1

                batch_list_1.append([state, action_1, next_state_1, reward_1, done])

                for _ in range(10):
                    transition = random.choice(batch_list_1)
                    state_, action_, next_state_, reward_, done_ = transition
                    player_1.q_learning(state_, action_, next_state_, reward_, done_, epsilon)
            else:
                action_2 = player_2.get_action(next_state_1)

                next_state_2, reward_2, done_2, info_2 = env.step(action_2)
                if step_info:
                    print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}, total_steps: {5}".format(
                        player_2.name, action_2, reward_2, done_2, info_2, total_steps
                    ))
                if render:
                    env.render()

                if done_2:
                    print_done_stat(info_2, episode)
                    done = done_2
                    batch_list_1.append([state, action_1, next_state_1, reward_2, done])
                else:
                    batch_list_1.append([state, action_1, next_state_1, reward_1, done])

                for _ in range(10):
                    transition = random.choice(batch_list_1)
                    state_, action_, next_state_, reward_, done_ = transition
                    player_1.q_learning(state_, action_, next_state_, reward_, done_, epsilon)

                state = next_state_2

    with open('player_1_agent.bin', 'wb') as f:
        pickle.dump(player_1, f)


def q_learning_main_2(step_info, render):
    env = TicTacToe()

    player_3 = DQN_Agent(name="PLAYER_3", env=env)
    player_4 = DQN_Agent(name="PLAYER_4", env=env)

    batch_list_3 = deque(maxlen=10000)
    batch_list_4 = deque(maxlen=10000)

    total_steps = 0
    NUM_PLAYER_3_WIN = 0
    NUM_PLAYER_4_WIN = 0
    NUM_DRAW = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()

        epsilon = epsilon_scheduled(episode)

        print("GAME: {0}, EPSILON: {1:.2f}".format(episode, epsilon))
        if render:
            env.render()

        done = False
        while not done:
            total_steps += 1
            action_3 = player_3.get_action(state)

            next_state_3, reward_3, done_3, info_3 = env.step(action_3)
            if step_info:
                print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}, total_steps: {5}".format(
                    player_3.name, action_3, reward_3, done_3, info_3, total_steps
                ))
            if render:
                env.render()

            if done_3:
                print_done_stat(info_3, episode)
                done = done_3

                batch_list_3.append([state, action_3, next_state_3, reward_3, done])

                for _ in range(10):
                    transition = random.choice(batch_list_3)
                    state_, action_, next_state_, reward_, done_ = transition
                    player_3.q_learning(state_, action_, next_state_, reward_, done_, epsilon)
            else:
                action_4 = player_4.get_action(next_state_3)

                next_state_4, reward_4, done_4, info_4 = env.step(action_4)
                if step_info:
                    print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}, total_steps: {5}".format(
                        player_4.name, action_4, reward_4, done_4, info_4, total_steps
                    ))
                if render:
                    env.render()

                if done_4:
                    print_done_stat(info_4, episode)
                    done = done_4
                    batch_list_3.append([state, action_3, next_state_3, reward_4, done])
                    batch_list_4.append([state, action_3, next_state_3, reward_4, done])
                else:
                    batch_list_3.append([state, action_3, next_state_3, reward_3, done])

                for _ in range(10):
                    transition = random.choice(batch_list_1)
                    state_, action_, next_state_, reward_, done_ = transition
                    player_1.q_learning(state_, action_, next_state_, reward_, done_, epsilon)

                state = next_state_2







            if current_player == player_3:
                batch_list_3.append([state, action, next_state, reward, done])
            else:
                batch_list_4.append([state, action, next_state, reward, done])

            for _ in range(10):
                if current_player == player_3:
                    transition = random.choice(batch_list_3)
                else:
                    transition = random.choice(batch_list_4)
                state_, action_, next_state_, reward_, done_ = transition
                current_player.q_learning(state_, action_, next_state_, reward_, done_, epsilon)

            if done:
                if info['winner'] == 1:
                    NUM_PLAYER_3_WIN += 1
                elif info['winner'] == -1:
                    NUM_PLAYER_4_WIN += 1
                else:
                    NUM_DRAW += 1
                print("NUM_PLAYER_3_WIN: {0} ({1:.2f}), NUM_PLAYER_4_WIN: {2} ({3:.2f}), NUM_DRAW: {4} ({5:.2f})\n".format(
                    NUM_PLAYER_3_WIN, NUM_PLAYER_3_WIN / (episode + 1),
                    NUM_PLAYER_4_WIN, NUM_PLAYER_4_WIN / (episode + 1),
                    NUM_DRAW, NUM_DRAW / (episode + 1),
                ))
            else:
                state = next_state

            if current_player == player_3:
                current_player = player_4
            else:
                current_player = player_3

    with open('player_3_agent.bin', 'wb') as f:
        pickle.dump(player_3, f)

    with open('player_4_agent.bin', 'wb') as f:
        pickle.dump(player_4, f)


def play_with_agent():
    env = TicTacToe()
    env.print_board_idx()

    with open('player_1_agent.bin', 'rb') as f:
        player_1 = pickle.load(f)
        player_1.make_greedy_policy()

    player_2 = Human_Agent(name="PLAYER_2", env=env)

    state = env.reset()
    current_player = player_1

    print("[시작 상태]")
    env.render()

    done = False
    while not done:
        action = current_player.get_action(state)

        next_state, _, done, info = env.step(action)

        print("[{0}]".format("DQN 에이전트" if current_player == player_1 else "당신(사람)"))
        env.render()

        if done:
            if info['winner'] == 1:
                print("DQN 에이전트가 이겼습니다.")
            elif info['winner'] == -1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            else:
                print("비겼습니다. 대단합니다!")
        else:
            state = next_state

        if current_player == player_1:
            current_player = player_2
        else:
            current_player = player_1


if __name__ == '__main__':
    step_info = False
    render = False
    #q_learning_main_1(step_info, render)
    play_with_agent()
