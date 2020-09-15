import os
import pickle
from basic2.practice_5.tic_tac_toe import TicTacToe

# 탐색 확률
from basic2.practice_6.tic_tac_toe_agents import Q_Learning_Agent
from basic2.practice_6.tic_tac_toe_q_learning import GameStatus, print_game_statistics

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 100000
VERBOSE = True


def self_play():
    env = TicTacToe()

    agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    with open(os.path.join('models', 'self_agent.bin'), 'rb') as f:
        q_table_and_policy = pickle.load(f)
        agent_1.q_table = q_table_and_policy['q_table']
        agent_1.policy = q_table_and_policy['policy']

    agent_2.q_table = agent_1.q_table
    agent_2.policy = agent_1.policy

    game_status = GameStatus()
    total_steps = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()
        current_agent = agent_1

        if VERBOSE:
            print("[시작 상태]")
            env.render()

        done = False
        while not done:
            total_steps += 1
            action = current_agent.get_action(state)

            next_state, _, done, info = env.step(action)

            if VERBOSE:
                print("[{0}]".format("Q-Learning 에이전트 1" if current_agent == agent_1 else "Q-Learning 에이전트 2"))
                env.render()

            if done:
                if VERBOSE:
                    if info['winner'] == 1:
                        print("Q-Learning 에이전트 1이 이겼습니다.")
                    elif info['winner'] == -1:
                        print("Q-Learning 에이전트 2가 이겼습니다!")
                    else:
                        print("비겼습니다!")

                done = done
                print_game_statistics(info, episode, 0.0, total_steps, game_status)
            else:
                state = next_state

            if current_agent == agent_1:
                current_agent = agent_2
            else:
                current_agent = agent_1


if __name__ == '__main__':
    self_play()
