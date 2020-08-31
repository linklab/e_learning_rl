import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False


def softmax(x):
    B = np.exp(x - np.max(x))
    C = np.sum(B)
    return B/C


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_grid_world_state_values_image(state_values, filename, GRID_HEIGHT, GRID_WIDTH):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = GRID_HEIGHT, GRID_WIDTH
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            table.add_cell(
                i, j, width, height,
                text=np.round(state_values[i][j], decimals=2),
                loc='center', facecolor='white'
            )

    # 행, 열 라벨 추가
    for i in range(len(state_values)):
        table.add_cell(
            i, -1, width, height,
            text=i+1,
            loc='right', edgecolor='none', facecolor='none'
        )
        table.add_cell(
            -1, i, width, height/2,
            text=i+1,
            loc='center', edgecolor='none', facecolor='none'
        )

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    ax.add_table(table)

    plt.savefig(filename)
    plt.close()


def draw_grid_world_action_values_image(action_values, filename, GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS, ACTION_SYMBOLS):
    action_str_values = []
    for i in range(GRID_HEIGHT):
        action_str_values.append([])
        for j in range(GRID_WIDTH):
            str_values = []
            for action in range(NUM_ACTIONS):
                str_values.append("{0} ({1}): {2:.2f}".format(
                    ACTION_SYMBOLS[action],
                    action,
                    np.round(action_values[i, j, action], decimals=2)
                ))
            action_str_values[i].append("\n".join(str_values))

    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = GRID_HEIGHT, GRID_WIDTH
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            table.add_cell(
                i, j, width, height,
                text=action_str_values[i][j],
                loc='center', facecolor='white'
            )

    # 행, 열 라벨 추가
    for i in range(len(action_str_values)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(10)

    ax.add_table(table)

    plt.savefig(filename)
    plt.close()


def draw_grid_world_optimal_policy_image(policy, filename, GRID_HEIGHT, GRID_WIDTH, ACTION_SYMBOLS, TERMINAL_STATES=None):
    action_str_values = []
    for i in range(GRID_HEIGHT):
        action_str_values.append([])
        for j in range(GRID_WIDTH):
            if TERMINAL_STATES and (i, j) in TERMINAL_STATES:
                continue
            str_values = []
            for action in policy[(i, j)]:
                str_values.append("{0} ({1})".format(
                    ACTION_SYMBOLS[action],
                    np.round(action, decimals=2)
                ))
            action_str_values[i].append("\n".join(str_values))

    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = GRID_HEIGHT, GRID_WIDTH
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if TERMINAL_STATES and (i, j) in TERMINAL_STATES:
                continue
            table.add_cell(i, j, width, height, text=action_str_values[i][j], loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(action_str_values)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(10)

    ax.add_table(table)

    plt.savefig(filename)
    plt.close()


def draw_grid_world_policy_image(policy, filename, GRID_HEIGHT, GRID_WIDTH, ACTION_SYMBOLS, TERMINAL_STATES=None):
    action_str_values = []
    for i in range(GRID_HEIGHT):
        action_str_values.append([])
        for j in range(GRID_WIDTH):
            if TERMINAL_STATES and (i, j) in TERMINAL_STATES:
                continue
            str_values = []
            actions, probs = policy[(i, j)]
            for action in actions:
                str_values.append("{0} ({1})".format(
                    ACTION_SYMBOLS[action],
                    np.round(probs[action], decimals=3)
                ))
            action_str_values[i].append("\n".join(str_values))

    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = GRID_HEIGHT, GRID_WIDTH
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if TERMINAL_STATES and (i, j) in TERMINAL_STATES:
                continue
            table.add_cell(i, j, width, height, text=action_str_values[i][j], loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(action_str_values)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(10)

    ax.add_table(table)

    plt.savefig(filename)
    plt.close()


def draw_random_walk_policy_image(policy, env):
    randomwalk_str = ""
    randomwalk_str += " T1      " + "       ".join(["{0}".format(i) for i in range(env.num_internal_states)]) + "      T2\n"

    randomwalk_str += "       "
    for state in env.observation_space.STATES:
        actions, prob = policy[state]
        randomwalk_str += "{0}:{1:4}  ".format(
            env.action_space.ACTION_SYMBOLS[env.action_space.ACTION_LEFT],
            np.round(prob[env.action_space.ACTION_LEFT], decimals=2)
        )
    randomwalk_str += "\n"

    randomwalk_str += "       "
    for state in env.observation_space.STATES:
        actions, prob = policy[state]
        randomwalk_str += "{0}:{1:4}  ".format(
            env.action_space.ACTION_SYMBOLS[env.action_space.ACTION_RIGHT],
            np.round(prob[env.action_space.ACTION_RIGHT], decimals=2)
        )

    print(randomwalk_str)


# 행동 가치함수를 표 형태로 그리는 함수
def draw_q_value_image_for_maze(env, q_value, run, planning_steps, episode):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, axis = plt.subplots()
    axis.set_axis_off()
    table = Table(axis, bbox=[0, 0, 1, 1])

    num_rows, num_cols = env.MAZE_HEIGHT, env.MAZE_WIDTH
    width, height = 1.0 / num_cols, 1.0 / num_rows

    for i in range(env.MAZE_HEIGHT):
        for j in range(env.MAZE_WIDTH):
            if np.sum(q_value[i][j]) == 0.0:
                symbol = " "
            else:
                action_idx = np.argmax(q_value[i][j])
                symbol = env.action_space.ACTION_SYMBOLS[action_idx]
            table.add_cell(i, j, width, height, text=symbol, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(env.MAZE_HEIGHT):
        table.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')

    for j in range(env.MAZE_WIDTH):
        table.add_cell(-1, j, width, height/2, text=j, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    axis.add_table(table)
    plt.savefig('images/maze_action_values_{0}_{1}_{2}.png'.format(run, planning_steps, episode))
    plt.close()
