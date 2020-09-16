import time
import numpy as np

from basic2.practice_5.tic_tac_toe_dummy_agents import Dummy_Agent

PLAYER_TO_SYMBOL = ['-', 'O', 'X']
PLAYER_1_INT = 1
PLAYER_2_INT = -1
BOARD_ROWS = 3
BOARD_COLS = 3

############################################
##  (0,0) -> 7, (0,1) -> 8, (0,2) -> 9    ##
##  (1,0) -> 4, (1,1) -> 5, (1,2) -> 6    ##
##  (2,0) -> 1, (2,1) -> 2, (2,2) -> 3    ##
############################################
def position_to_action_idx(row_idx, col_idx):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if row_idx == i and col_idx == j:
                return (2 - i) * 3 + j + 1

############################################
##  7 -> (0,0), 8 -> (0,1), 9 -> (0,2)    ##
##  4 -> (1,0), 5 -> (1,1), 6 -> (1,2)    ##
##  1 -> (2,0), 2 -> (2,1), 3 -> (2,2)    ##
############################################
def action_idx_to_position(idx):
    i, j = divmod(idx, 3)
    i = 2 - i if j != 0 else 3 - i
    j = j - 1 if j != 0 else 2
    return (i, j)


#########################################################
# 게임판 상태의 저장, 출력 그리고 종료 판정을 수행하는 State 클래스   #
#########################################################
class State:
    def __init__(self, board_rows=3, board_cols=3):
        # 게임판 상태는 board_rows * board_cols 크기의 배열로 표현
        # 게임판에서 플레이어는 정수값으로 구분
        # 1 : 선공 플레이어, -1 : 후공 플레이어, 0 : 초기 공백 상태
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board_size = board_rows * board_cols

        self.data = np.zeros(shape=[board_rows, board_cols], dtype=np.int)
        self.winner = None
        self.id = None  # 게임의 각 상태들을 구분짓기 위한 해시값
        self.end = None

    #특정 상태에서의 유일한 해시 ID값 계산
    # def identifier(self):
    #     if self.id is None:
    #         identifier = 0
    #         for i in range(self.board_rows):
    #             for j in range(self.board_cols):
    #                 identifier = identifier * 3 + self.data[i, j] + 1
    #         self.id = identifier
    #
    #     return self.id

    def identifier(self):
        if self.id is None:
            identifier = 0
            k = 0
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    identifier += self.data[i, j] * 3 ** k
                    k += 1
            self.id = identifier
        return self.id

    # 현 상태에서 유효한 행동 ID 리스트 반환
    def get_available_actions(self):
        if self.is_end_state():
            available_positions = []
        else:
            available_positions = [(i, j) for i in range(BOARD_ROWS) for j in range(BOARD_COLS) if self.data[i, j] == 0]

        available_action_ids = []
        for available_position in available_positions:
            available_action_ids.append(position_to_action_idx(available_position[0], available_position[1]))

        return available_action_ids

    # 플레이어가 종료 상태에 있는지 판단.
    # 플레이어가 게임을 이기거나, 지거나, 비겼다면 True 반환, 그 외는 False 반환
    def is_end_state(self):
        if self.end is not None:
            return self.end

        results = []

        # 게임판 가로 3칸 승리조건 확인
        for i in range(self.board_rows):
            results.append(np.sum(self.data[i, :]))

        # 게임판 세로 3칸 승리조건 확인
        for i in range(self.board_cols):
            results.append(np.sum(self.data[:, i]))

        # 게임판의 두 개 대각선에 대 대각선 3칸 승리조건 확인
        trace = 0
        reverse_trace = 0
        for i in range(self.board_rows):
            trace += self.data[i, i]
            reverse_trace += self.data[i, self.board_rows - 1 - i]

        results.append(trace)
        results.append(reverse_trace)

        # results에는 총 8(=3 + 3 + 1 + 1)개의 값이 원소로 존재함
        # PLAYER_1 또는 PLAYER_2 승리 조건 확인
        for result in results:
            if result == 3 or result == -3:
                self.end = True
                if result == 3:
                    self.winner = PLAYER_1_INT
                else:
                    self.winner = PLAYER_2_INT
                return self.end

        # 무승부 확인
        sum_values = np.sum(np.abs(self.data))
        if sum_values == self.board_size:
            self.winner = 0
            self.end = True
            return self.end

        # 게임이 아직 종료되지 않음
        self.end = False
        return self.end

    # 게임판 상태 출력
    def get_state_as_board(self):
        board_str = ""
        for i in range(self.board_rows):
            board_str += '-------------\n'
            out = '| '
            for j in range(self.board_cols):
                out += PLAYER_TO_SYMBOL[int(self.data[i, j])] + ' | '
            board_str += out + "\n"
        board_str += '-------------\n'
        return board_str

    def __str__(self):
        return str([''.join(['O' if x == 1 else 'X' if x == -1 else '-' for x in y]) for y in self.data])

################################################################
# 플레이어 1,2 간의 게임 진행을 담당하는 Env 클래스
class TicTacToe:
    def __init__(self):
        self.BOARD_SIZE = BOARD_ROWS * BOARD_COLS
        self.current_state = None       # 현재 상태 관리
        self.current_agent_int = None   # 현재 에이전트(플레이어) 관리

        self.INITIAL_STATE = State()    # 초기 상태 설정

        # 모든 상태가 ‘상태 식별자: 상태’로 관리되는 사전 생성
        self.ALL_STATES = {}
        self.ALL_STATES[self.INITIAL_STATE.identifier()] = self.INITIAL_STATE

        self.generate_all_states(state=self.INITIAL_STATE, player_int=PLAYER_1_INT)
        print("####### Tic-Tac-Toe Env Initialized with {0} States #######".format(len(self.ALL_STATES)))

        # for id, state in self.ALL_STATES.items():
        #     print(id, state)

    def reset(self):
        self.current_agent_int = PLAYER_1_INT
        self.current_state = self.INITIAL_STATE
        return self.current_state

    # 게임 진행을 위해 각 플레이어의 착수 때 마다 호출
    def step(self, action=None):
        # 플레이어의 행동에 의한 다음 상태 갱신
        position = action_idx_to_position(action)
        next_state = self.get_new_state(
            i=position[0], j=position[1], state_data=self.current_state.data, player_int=self.current_agent_int
        )

        next_state_hash = next_state.identifier()
        assert next_state_hash in self.ALL_STATES, print(next_state.get_state_as_board())

        next_state = self.ALL_STATES[next_state_hash]

        done = next_state.is_end_state()

        if done:
            info = {'current_agent_int': self.current_agent_int, 'winner': next_state.winner}
            if next_state.winner == PLAYER_1_INT:
                reward = 1.0
            elif next_state.winner == PLAYER_2_INT:
                reward = -1.0
            else:
                reward = 0.0
        else:
            info = {'current_agent_int': self.current_agent_int}
            reward = 0.0

        self.current_state = next_state

        if self.current_agent_int == PLAYER_1_INT:
            self.current_agent_int = PLAYER_2_INT
        else:
            self.current_agent_int = PLAYER_1_INT

        return next_state, reward, done, info

    def render(self, mode='human'):
        print(self.current_state.get_state_as_board())

    # 주어진 상태 및 현재 플레이어 심볼에 대하여 발생 가능한 모든 게임 상태 집합 생성
    def generate_all_states(self, state, player_int):
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i][j] == 0:
                    # 도달 가능한 새로운 상태 생성
                    new_state = self.get_new_state(i, j, state.data, player_int)

                    # 새로운 상태의 해시값 가져오기
                    new_hash = new_state.identifier()

                    if new_hash not in self.ALL_STATES:
                        # 모든 게임 상태 집합 갱신
                        self.ALL_STATES[new_hash] = new_state

                        # 게임 상태가 종료 상태가 아닌 경우 재귀 호출로 새로운 상태 생성 계속 지속
                        if not new_state.is_end_state():
                            self.generate_all_states(new_state, -1 * player_int)

    def get_new_state(self, i, j, state_data, player_int):
        new_state = State()

        # 주어진 상태의 게임판 상황 복사
        new_state.data = np.copy(state_data)

        # 플레이어의 행동(i, j 위치에 표시) 반영
        new_state.data[i, j] = player_int

        return new_state

    def print_board_idx(self):
        print()
        print("[[[Tic-Tac-Toe 보드 내 각 셀을 선택할 때 다음 숫자 키패드를 사용하세요.]]]")
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                out += str(position_to_action_idx(i, j)) + ' | '
            print(out)
        print('-------------')



def main():
    env = TicTacToe()
    state = env.reset()
    env.render()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    current_agent = agent_1

    done = False
    total_steps = 0
    while not done:
        total_steps += 1
        action = current_agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        print("[{0}] action: {1}, reward: {2}, done: {3}, info: {4}, total_steps: {5}".format(
            current_agent.name, action, reward, done, info, total_steps
        ))
        env.render()

        state = next_state
        time.sleep(2)

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent =agent_1


if __name__ == "__main__":
    main()