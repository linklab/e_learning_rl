import os
import pickle
from basic2.practice_5.tic_tac_toe import TicTacToe
from basic2.practice_5.tic_tac_toe_dummy_agents import Dummy_Agent
from basic2.practice_6.tic_tac_toe_agents import Q_Learning_Agent
from basic2.practice_6.tic_tac_toe_utils import GameStatus, epsilon_scheduled
from basic2.practice_6.tic_tac_toe_utils import print_step_status, print_game_statistics, draw_performance

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 10000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 100000

STEP_VERBOSE = False
BOARD_RENDER = False


# 선수 에이전트: Q-Learning 에이전트, 후수 에이전트: Dummy 에이전트
def q_learning_for_agent_1_vs_dummy():
    game_status = GameStatus()
    env = TicTacToe()

    agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        #epsilon = epsilon_scheduled(episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON)
        epsilon = epsilon_scheduled(episode, MAX_EPISODES, INITIAL_EPSILON, FINAL_EPSILON)

        if BOARD_RENDER:
            env.render()

        done = False

        agent_1_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(
                agent_1, state, action, next_state, reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # reward: agent_1이 착수하여 done=True ==> agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error += agent_1.q_learning(state, action, None, reward, done, epsilon)

                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps, game_status, agent_1, agent_2
                )
            else:
                # agent_2 스텝 수행
                action_2 = agent_2.get_action(next_state)
                next_state, reward, done, info = env.step(action_2)
                print_step_status(
                    agent_2, state, action_2, next_state, reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # reward: agent_2가 착수하여 done=True ==> agent_2가 이기면 -1.0, 비기면 0.0
                    agent_1_episode_td_error += agent_1.q_learning(state, action, None, reward, done, epsilon)

                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps, game_status, agent_1, agent_2
                    )
                else:
                    agent_1_episode_td_error += agent_1.q_learning(state, action, next_state, reward, done, epsilon)

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)

    game_status.agent_1_count_state_updates = agent_1.count_state_updates
    draw_performance(game_status, 'q_learning_for_agent_1_vs_dummy.png', MAX_EPISODES)

    agent_1.make_greedy_policy()
    with open(os.path.join('models', 'agent_1.bin'), 'wb') as f:
        q_table_and_policy = {
            'q_table': agent_1.q_table,
            'policy': agent_1.policy
        }
        pickle.dump(q_table_and_policy, f)


# 선수 에이전트: Dummy 에이전트, 후수 에이전트: Q-Learning 에이전트
def q_learning_for_dummy_vs_agent_2():
    game_status = GameStatus()

    env = TicTacToe()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        current_agent = agent_1

        #epsilon = epsilon_scheduled(episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON)
        epsilon = epsilon_scheduled(episode, MAX_EPISODES, INITIAL_EPSILON, FINAL_EPSILON)

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action_1 = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action_1)
            print_step_status(
                current_agent, state, action_1, next_state, reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps, game_status, agent_1, agent_2
                )

                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                # reward: agent_1이 착수하여 done=True ==> agent_1이 이기면 1.0, 비기면 0.0
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.q_learning(STATE_2, ACTION_2, None, -1.0 * reward, done, epsilon)
            else:
                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.q_learning(STATE_2, ACTION_2, next_state, reward, done, epsilon)

                # agent_2 스텝 수행
                state = next_state
                action = agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)
                print_step_status(
                    agent_2, state, action, next_state, reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps, game_status, agent_1, agent_2
                    )

                    # reward: agent_2가 착수하여 done=True ==> agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error += agent_2.q_learning(state, action, None, -1.0 * reward, done, epsilon)
                else:
                    # agent_2에 방문한 현재 상태 및 수행한 행동 정보를 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

            state = next_state

        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

    game_status.agent_2_count_state_updates = agent_2.count_state_updates
    draw_performance(game_status, 'q_learning_for_dummy_vs_agent_2.png', MAX_EPISODES)

    agent_2.make_greedy_policy()
    with open(os.path.join('models', 'agent_2.bin'), 'wb') as f:
        q_table_and_policy = {
            'q_table': agent_2.q_table,
            'policy': agent_2.policy
        }
        pickle.dump(q_table_and_policy, f)


# 선수 에이전트: Q-Learning 에이전트, 후수 에이전트: Q-Learning 에이전트
def q_learning_for_self_play():
    game_status = GameStatus()

    env = TicTacToe()

    agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    agent_2.q_table = agent_1.q_table
    agent_2.policy = agent_1.policy

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        #epsilon = epsilon_scheduled(episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON)
        epsilon = epsilon_scheduled(episode, MAX_EPISODES, INITIAL_EPSILON, FINAL_EPSILON)

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_1_episode_td_error = 0.0
        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(
                agent_1, state, action, next_state, reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps, game_status, agent_1, agent_2
                )

                # reward: agent_1가 착수하여 done=True ==> agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error += agent_1.q_learning(state, action, None, reward, done, epsilon)

                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.q_learning(STATE_2, ACTION_2, None, -1.0 * reward, done, epsilon)
            else:
                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.q_learning(STATE_2, ACTION_2, next_state, reward, done, epsilon)

                # agent_1이 방문한 현재 상태 및 수행한 행동 정보를 저장해 두었다가 추후 활용
                STATE_1 = state
                ACTION_1 = action

                # agent_2 스텝 수행
                state = next_state
                action = agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)
                print_step_status(
                    agent_2, state, action, next_state, reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps, game_status, agent_1, agent_2
                    )

                    # reward: agent_2가 착수하여 done=True ==> agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error += agent_2.q_learning(state, action, None, -1.0 * reward, done, epsilon)

                    # 미루워 두었던 agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += agent_1.q_learning(STATE_1, ACTION_1, None, reward, done, epsilon)
                else:
                    # agent_2이 방문한 현재 상태 및 수행한 행동 정보를 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

                    # 미루워 두었던 agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += agent_1.q_learning(STATE_1, ACTION_1, next_state, reward, done, epsilon)

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)
        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

    game_status.agent_1_count_state_updates = agent_1.count_state_updates
    game_status.agent_2_count_state_updates = agent_2.count_state_updates
    draw_performance(game_status, "q_learning_for_self_play.png", MAX_EPISODES)

    agent_1.make_greedy_policy()
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
