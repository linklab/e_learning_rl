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

        self.player_1_win_rate_over_100_games = []
        self.player_2_win_rate_over_100_games = []
        self.draw_rate_over_100_games = []


def draw_performance(game_status, file_name, max_episodes):
    plt.clf()
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
    plt.savefig(os.path.join('images', file_name))
    plt.close()


def epsilon_scheduled(current_episode, last_scheduled_episodes, initial_epsilon, final_epsilon):
    fraction = min(current_episode / last_scheduled_episodes, 1.0)
    epsilon = min(initial_epsilon + fraction * (final_epsilon - initial_epsilon), initial_epsilon)
    return epsilon


def print_game_statistics(info, episode, epsilon, total_steps, game_status):
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

    print("### GAMES DONE: {0} | episolon: {1:.2f} | total_steps: {2} | "
          "agent_1_win : agent_2_win : draw = {3} : {4} : {5} | winning_rate_over_recent_100_games --> {6:.1f}% : {7:.1f}% : {8:.1f}%".format(
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