import math
import os
import matplotlib.pyplot as plt
import numpy as np

from basic2.practice_6.tic_tac_toe_agents import Q_Learning_Agent

linestyles = ['-', '--', ':']
legends = ["AGENT-1 WIN", "AGENT-2 WIN", "DRAW"]


class GameStatus:
    def __init__(self):
        self.num_player_1_win = 0
        self.num_player_2_win = 0
        self.num_draw = 0

        self.player_1_win_info_list = []
        self.player_2_win_info_list = []
        self.draw_info_list = []
        self.agent_1_avg_q_list = []
        self.agent_2_avg_q_list = []
        self.agent_1_episode_td_error = []
        self.agent_2_episode_td_error = []
        self.epsilon_list = []

        self.player_1_win_rate_over_100_games = []
        self.player_2_win_rate_over_100_games = []
        self.draw_rate_over_100_games = []

        self.agent_1_count_state_updates = None
        self.agent_2_count_state_updates = None

    def set_agent_1_episode_td_error(self, agent_1_episode_td_error):
        self.agent_1_episode_td_error.append(agent_1_episode_td_error)

    def set_agent_2_episode_td_error(self, agent_2_episode_td_error):
        self.agent_2_episode_td_error.append(agent_2_episode_td_error)


def draw_performance(game_status, file_name, max_episodes):
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 12)

    figure, _ = plt.subplots(nrows=4, ncols=1)

    plt.subplot(511)
    for i in range(3):
        if i == 0:
            values = game_status.player_1_win_rate_over_100_games[::100]
        elif i == 1:
            values = game_status.player_2_win_rate_over_100_games[::100]
        else:
            values = game_status.draw_rate_over_100_games[::100]
        plt.plot(range(1, max_episodes + 1, 100), values, linestyle=linestyles[i], label=legends[i])

    plt.xlabel('Episode')
    plt.ylabel('Winning Rate')
    plt.legend()

    plt.subplot(512)
    if game_status.agent_1_avg_q_list:
        plt.plot(range(1, max_episodes + 1, 10000), game_status.agent_1_avg_q_list, label="Agent 1")
    if game_status.agent_2_avg_q_list:
        plt.plot(range(1, max_episodes + 1, 10000), game_status.agent_2_avg_q_list, label="Agent 2")

    plt.xlabel('Episode')
    plt.ylabel('Average Q Values')
    plt.legend()

    plt.subplot(513)
    if game_status.agent_1_episode_td_error:
        plt.plot(range(1, max_episodes + 1, 100), game_status.agent_1_episode_td_error[::100], label="Agent 1")
        #plt.plot(range(1, max_episodes + 1, 10000), game_status.agent_1_episode_td_error[::10000], label="Agent 1")
    if game_status.agent_2_episode_td_error:
        plt.plot(range(1, max_episodes + 1, 100), game_status.agent_2_episode_td_error[::100], label="Agent 1")
        #plt.plot(range(1, max_episodes + 1, 10000), game_status.agent_2_episode_td_error[::10000], label="Agent 2")

    plt.xlabel('Episode')
    plt.ylabel('Q Learning TD Error')
    plt.legend()

    plt.subplot(514)
    plt.plot(range(1, max_episodes + 1, 10000), game_status.epsilon_list, label="Epsilon")

    plt.subplot(515)
    if game_status.agent_1_count_state_updates:
        data = [(k, v) for k, v in game_status.agent_1_count_state_updates.items()]
        data.sort(key=lambda x: x[1], reverse=True)
        keys = [k[0] for k in data]
        values = [k[1] for k in data]

        plt.bar(range(0, len(values)), values, label="Number of updates (agent 1)")
        plt.yscale('log')

    if game_status.agent_2_count_state_updates:
        data = [(k, v) for k, v in game_status.agent_2_count_state_updates.items()]
        data.sort(key=lambda x: x[1], reverse=True)
        keys = [k[0] for k in data]
        values = [k[1] for k in data]

        plt.bar(range(0, len(values)), values, label="Number of updates (agent 2)")
        plt.yscale('log')

    plt.xlabel('Episode')
    plt.ylabel('Number of updates')
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join('images', file_name))
    plt.close()


# def epsilon_scheduled(current_episode, last_scheduled_episodes, initial_epsilon, final_epsilon):
#     fraction = min(current_episode / last_scheduled_episodes, 1.0)
#     epsilon = min(initial_epsilon + fraction * (final_epsilon - initial_epsilon), initial_epsilon)
#     return epsilon


# https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
def epsilon_scheduled(current_episode, max_episodes, initial_epsilon, final_epsilon):
    A = 0.3
    B = 0.2
    C = 0.2
    standardized_time = (current_episode - A * max_episodes)/(B * max_episodes)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = max(initial_epsilon - (1 / cosh + (current_episode * C / max_episodes)), final_epsilon)
    return epsilon


def print_game_statistics(info, episode, epsilon, total_steps, game_status, agent_1=None, agent_2=None):
    if info['winner'] == 1:
        game_status.num_player_1_win += 1
    elif info['winner'] == -1:
        game_status.num_player_2_win += 1
    else:
        game_status.num_draw += 1

    game_status.player_1_win_info_list.append(1 if info['winner'] == 1 else 0)
    game_status.player_2_win_info_list.append(1 if info['winner'] == -1 else 0)
    game_status.draw_info_list.append(1 if info['winner'] == 0 else 0)

    game_status.player_1_win_rate_over_100_games.append(np.average(game_status.player_1_win_info_list[-100:]) * 100)
    game_status.player_2_win_rate_over_100_games.append(np.average(game_status.player_2_win_info_list[-100:]) * 100)
    game_status.draw_rate_over_100_games.append(np.average(game_status.draw_info_list[-100:]) * 100)

    if episode % 10000 == 0:
        if isinstance(agent_1, Q_Learning_Agent):
            game_status.agent_1_avg_q_list.append(agent_1.avg_q_value())

        if isinstance(agent_2, Q_Learning_Agent):
            game_status.agent_2_avg_q_list.append(agent_2.avg_q_value())

        game_status.epsilon_list.append(epsilon)

    print("### GAMES DONE: {0} | episolon: {1:.2f} | total_steps: {2} | "
          "agent_1_win : agent_2_win : draw = {3} : {4} : {5} | "
          "winning_rate_over_recent_100_games --> {6:4.1f}% : {7:4.1f}% : {8:4.1f}%".format(
        episode, epsilon, total_steps,
        game_status.num_player_1_win, game_status.num_player_2_win, game_status.num_draw,
        game_status.player_1_win_rate_over_100_games[-1],
        game_status.player_2_win_rate_over_100_games[-1],
        game_status.draw_rate_over_100_games[-1]
    ))


def print_step_status(agent, state, action, next_state, reward, done, info, env, step_verbose, board_render):
    if step_verbose:
        state_q_values = agent.q_table[state.identifier()] if isinstance(agent, Q_Learning_Agent) else {}
        state_q_value_list = []
        for action_id, q_value in state_q_values.items():
            state_q_value_list.append("{0}:{1:.3f}".format(action_id, q_value))

        print("[{0}]|{1}:{2:80s}|action: {3}|next_state: {4}|reward: {5:4.1f}|done: {6:5s}|info: {7}".format(
            agent.name, state, ", ".join(state_q_value_list), action, next_state, reward, str(done), info
        ))
    if board_render:
        env.BOARD_RENDER()