import os
import pickle
import random
from collections import deque
from basic2.practice_5.tic_tac_toe import TicTacToe
from basic2.practice_5.tic_tac_toe_dummy_agents import Dummy_Agent
from basic2.practice_6.tic_tac_toe_agents import Q_Learning_Agent
import matplotlib.pyplot as plt

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 20000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 200000

STEP_VERBOSE = False
BOARD_RENDER = False

linestyles = ['-', '--', ':']
legends = ["AGENT-1 WIN", "AGENT-2 WIN", "DRAW"]


def draw_performance(game_status, file_name):
    plt.clf()
    for i in range(3):
        if i == 0:
            values = game_status.player_1_win_rate_list[::10]
        elif i == 1:
            values = game_status.player_2_win_rate_list[::10]
        else:
            values = game_status.draw_rate_list[::10]
        plt.plot(range(1, MAX_EPISODES + 1, 10), values, linestyle=linestyles[i], label=legends[i])

    plt.xlabel('Episode')
    plt.ylabel('Winning Rate')
    plt.legend()
    plt.savefig(os.path.join('images', file_name))
    plt.close()


class GameStatus:
    def __init__(self):
        self.num_player_1_win = 0
        self.num_player_2_win = 0
        self.num_draw = 0

        self.player_1_win_rate_list = []
        self.player_2_win_rate_list = []
        self.draw_rate_list = []


def epsilon_scheduled(current_episode):
    fraction = min(current_episode / LAST_SCHEDULED_EPISODES, 1.0)
    epsilon = min(INITIAL_EPSILON + fraction * (FINAL_EPSILON - INITIAL_EPSILON), INITIAL_EPSILON)
    return epsilon


def print_game_status(info, episode, epsilon, total_steps, game_status):
    if info['winner'] == 1:
        game_status.num_player_1_win += 1
    elif info['winner'] == -1:
        game_status.num_player_2_win += 1
    else:
        game_status.num_draw += 1

    game_status.player_1_win_rate_list.append(100 * game_status.num_player_1_win / (episode + 1))
    game_status.player_2_win_rate_list.append(100 * game_status.num_player_2_win / (episode + 1))
    game_status.draw_rate_list.append(100 * game_status.num_draw / (episode + 1))

    print("games done: {0}, episolon: {1:.2f}, total_steps: {2}, "
          "num_player_1_win: {3} ({4:.1f}%), num_player_2_win: {5} ({6:.1f}%), num_draw: {7} ({8:.1f}%)".format(
        episode, epsilon, total_steps,
        game_status.num_player_1_win, game_status.player_1_win_rate_list[-1],
        game_status.num_player_2_win, game_status.player_2_win_rate_list[-1],
        game_status.num_draw, game_status.draw_rate_list[-1]
    ))


def print_step_status(agent, action, reward, done, info, env):
    if STEP_VERBOSE:
        print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}".format(
            agent.name, action, reward, done, info
        ))
    if BOARD_RENDER:
        env.BOARD_RENDER()


def q_learning_for_agent_1_vs_dummy():
    game_status = GameStatus()
    env = TicTacToe()

    agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    batch_list_1 = deque(maxlen=1000)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        current_agent = agent_1

        epsilon = epsilon_scheduled(episode)

        if BOARD_RENDER:
            env.render()

        episode_done = False
        state_1, action_1, next_state_1, reward_1, done_1 = None, None, None, None, None

        while not episode_done:
            total_steps += 1

            action = current_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(current_agent, action, reward, done, info, env)

            if current_agent == agent_1:
                if done:
                    print_game_status(info, episode, epsilon, total_steps, game_status)
                    episode_done = done

                    # reward: agent_1가 착수하여 done=True ==> agent_1이 이기면 1.0, 비기면 0.0
                    batch_list_1.append([state, action, None, reward, done])
                else:
                    # agent_1을 위한 현재 transition 정보를 저장해 두었다가 추후 활용
                    state_1, action_1, next_state_1, reward_1, done_1 = state, action, next_state, reward, done
            else:
                if done:
                    print_game_status(info, episode, epsilon, total_steps, game_status)
                    episode_done = done

                    # 미루워 두었던 agent_1의 배치에 transition 정보 추가
                    if state_1 is not None:
                        # reward: agent_2가 착수하여 done=True ==> agent_2가 이기면 -1.0, 비기면 0.0
                        batch_list_1.append([state_1, action_1, None, reward, done])
                else:
                    # 미루워 두었던 agent_1의 배치에 transition 정보 추가
                    if state_1 is not None:
                        batch_list_1.append([state_1, action_1, next_state_1, reward_1, done_1])

            if current_agent == agent_1 and len(batch_list_1) >= 3:
                for _ in range(3):
                    transition = random.choice(batch_list_1)
                    state_, action_, next_state_, reward_, done_ = transition
                    current_agent.q_learning(state_, action_, next_state_, reward_, done_, epsilon)

            state = next_state

            if current_agent == agent_1:
                current_agent = agent_2
            else:
                current_agent = agent_1

    draw_performance(game_status, 'q_learning_for_agent_1_vs_dummy.png')

    with open(os.path.join('models', 'agent_1.bin'), 'wb') as f:
        q_table_and_policy = {
            'q_table': agent_1.q_table,
            'policy': agent_1.policy
        }
        pickle.dump(q_table_and_policy, f)


def q_learning_for_dummy_vs_agent_2():
    game_status = GameStatus()

    env = TicTacToe()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    batch_list_2 = deque(maxlen=1000)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        current_agent = agent_1

        epsilon = epsilon_scheduled(episode)

        if BOARD_RENDER:
            env.render()

        episode_done = False
        state_2, action_2, next_state_2, reward_2, done_2 = None, None, None, None, None

        while not episode_done:
            total_steps += 1

            action = current_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(current_agent, action, reward, done, info, env)

            if current_agent == agent_1:
                if done:
                    print_game_status(info, episode, epsilon, total_steps, game_status)
                    episode_done = done

                    # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                    if state_2 is not None:
                        batch_list_2.append([state_2, action_2, None, -1.0 * reward, done])
                else:
                    # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                    if state_2 is not None:
                        batch_list_2.append([state_2, action_2, next_state_2, reward_2, done_2])
            else:
                if done:
                    print_game_status(info, episode, epsilon, total_steps, game_status)
                    episode_done = done

                    # reward: agent_2가 착수하여 done=True ==> agent_2가 이기면 -1.0, 비기면 0.0
                    batch_list_2.append([state, action, None, -1.0 * reward, done])
                else:
                    # agent_2을 위한 현재 transition 정보를 저장해 두었다가 추후 활용
                    state_2, action_2, next_state_2, reward_2, done_2 = state, action, next_state, reward, done

            if current_agent == agent_2 and len(batch_list_2) >= 3:
                for _ in range(3):
                    transition = random.choice(batch_list_2)
                    state_, action_, next_state_, reward_, done_ = transition
                    current_agent.q_learning(state_, action_, next_state_, reward_, done_, epsilon)

            state = next_state

            if current_agent == agent_1:
                current_agent = agent_2
            else:
                current_agent = agent_1

    draw_performance(game_status, 'q_learning_for_dummy_vs_agent_2.png')

    with open(os.path.join('models', 'agent_2.bin'), 'wb') as f:
        q_table_and_policy = {
            'q_table': agent_2.q_table,
            'policy': agent_2.policy
        }
        pickle.dump(q_table_and_policy, f)


def q_learning_for_self_play():
    game_status = GameStatus()

    env = TicTacToe()

    agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    agent_2.q_table = agent_1.q_table
    agent_2.policy = agent_2.policy

    batch_list_1 = deque(maxlen=10000)
    batch_list_2 = deque(maxlen=10000)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        current_agent = agent_1

        epsilon = epsilon_scheduled(episode)

        if BOARD_RENDER:
            env.render()

        episode_done = False
        state_1, action_1, next_state_1, reward_1, done_1 = None, None, None, None, None
        state_2, action_2, next_state_2, reward_2, done_2 = None, None, None, None, None

        while not episode_done:
            total_steps += 1

            action = current_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(current_agent, action, reward, done, info, env)

            if current_agent == agent_1:
                if done:
                    print_game_status(info, episode, epsilon, total_steps, game_status)
                    episode_done = done

                    # reward: agent_1가 착수하여 done=True ==> agent_1이 이기면 1.0, 비기면 0.0
                    batch_list_1.append([state, action, None, reward, done])

                    # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                    if state_2 is not None:
                        batch_list_2.append([state_2, action_2, None, -1.0 * reward, done])
                else:
                    # agent_1을 위한 현재 transition 정보를 저장해 두었다가 추후 활용
                    state_1 = state
                    action_1 = action
                    next_state_1 = next_state
                    reward_1 = reward
                    done_1 = done

                    # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                    if state_2 is not None:
                        batch_list_2.append([state_2, action_2, next_state_2, reward_2, done_2])
            else:
                if done:
                    print_game_status(info, episode, epsilon, total_steps, game_status)
                    episode_done = done

                    # reward: agent_2가 착수하여 done=True ==> agent_2가 이기면 -1.0, 비기면 0.0
                    batch_list_2.append([state, action, None, -1.0 * reward, done])

                    # 미루워 두었던 agent_1의 배치에 transition 정보 추가
                    if state_1 is not None:
                        batch_list_1.append([state_1, action_1, None, reward, done])
                else:
                    # agent_2을 위한 현재 transition 정보를 저장해 두었다가 추후 활용
                    state_2 = state
                    action_2 = action
                    next_state_2 = next_state
                    reward_2 = reward
                    done_2 = done

                    # 미루워 두었던 agent_1의 배치에 transition 정보 추가
                    if state_1 is not None:
                        batch_list_1.append([state_1, action_1, next_state_1, reward_1, done_1])

            if current_agent == agent_1 and len(batch_list_1) >= 3:
                for _ in range(3):
                    transition = random.choice(batch_list_1)
                    state_, action_, next_state_, reward_, done_ = transition
                    current_agent.q_learning(state_, action_, next_state_, reward_, done_, epsilon)

            if current_agent == agent_2 and len(batch_list_2) >= 3:
                for _ in range(3):
                    transition = random.choice(batch_list_2)
                    state_, action_, next_state_, reward_, done_ = transition
                    current_agent.q_learning(state_, action_, next_state_, reward_, done_, epsilon)

            state = next_state

            if current_agent == agent_1:
                current_agent = agent_2
            else:
                current_agent = agent_1

    draw_performance(game_status, "q_learning_for_self_play.png")

    with open(os.path.join('models', 'self_agent.bin'), 'wb') as f:
        q_table_and_policy = {
            'q_table': agent_1.q_table,
            'policy': agent_1.policy
        }
        pickle.dump(q_table_and_policy, f)


if __name__ == '__main__':
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    # 모델 저장 경로 확인 및 생성
    if not os.path.exists('models/'):
        os.makedirs('models/')

    q_learning_for_agent_1_vs_dummy()
    q_learning_for_dummy_vs_agent_2()
    q_learning_for_self_play()
