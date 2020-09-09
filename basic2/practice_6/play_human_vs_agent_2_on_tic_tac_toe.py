import os
import pickle
from basic2.practice_5.tic_tac_toe import TicTacToe
from basic2.practice_6.tic_tac_toe_agents import Human_Agent, Q_Learning_Agent


def play_with_agent_2():
    env = TicTacToe()
    env.print_board_idx()

    agent_1 = Human_Agent(name="AGENT_1", env=env)
    agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    with open(os.path.join('models', 'agent_2.bin'), 'rb') as f:
        q_table_and_policy = pickle.load(f)
        agent_2.q_table = q_table_and_policy['q_table']
        agent_2.policy = q_table_and_policy['policy']
        agent_2.make_greedy_policy()

    state = env.reset()
    current_player = agent_1

    print("[시작 상태]")
    env.render()

    done = False
    while not done:
        action = current_player.get_action(state)

        next_state, _, done, info = env.step(action)

        print("[{0}]".format("당신[사람]" if current_player == agent_1 else "Q-Learning 에이전트"))
        env.render()

        if done:
            if info['winner'] == 1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            elif info['winner'] == -1:
                print("Q-Learning 에이전트가 이겼습니다.")
            else:
                print("비겼습니다. 잘했습니다!")
        else:
            state = next_state

        if current_player == agent_1:
            current_player = agent_2
        else:
            current_player = agent_1


if __name__ == '__main__':
    play_with_agent_2()
