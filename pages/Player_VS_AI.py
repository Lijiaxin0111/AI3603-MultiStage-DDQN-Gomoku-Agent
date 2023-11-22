"""
FileName: app.py
Author: Benhao Huang
Create Date: 2023/11/19
Description: this file is used to display our project and add visualization elements to the game, using Streamlit
"""

import time
import pandas as pd
from copy import deepcopy

# import torch
import numpy as np
import streamlit as st
from scipy.signal import convolve  # this is used to check if any player wins
from streamlit import session_state
from streamlit_server_state import server_state, server_state_lock
from Gomoku_MCTS import MCTSpure, alphazero, Board, PolicyValueNet
import matplotlib.pyplot as plt

from const import (
    _BLACK,  # 1, for human
    _WHITE,  # 2 , for AI
    _BLANK,
    _PLAYER_COLOR,
    _PLAYER_SYMBOL,
    _ROOM_COLOR,
    _VERTICAL,
    _HORIZONTAL,
    _DIAGONAL_UP_LEFT,
    _DIAGONAL_UP_RIGHT,
    _BOARD_SIZE,
    _BOARD_SIZE_1D
)


# Utils
class Room:
    def __init__(self, room_id) -> None:
        self.ROOM_ID = room_id
        # self.BOARD = np.zeros(shape=(_BOARD_SIZE, _BOARD_SIZE), dtype=int)
        self.BOARD = Board(width=_BOARD_SIZE, height=_BOARD_SIZE, n_in_row=5, players=[_BLACK, _WHITE])
        self.PLAYER = _BLACK
        self.TURN = self.PLAYER
        self.HISTORY = (0, 0)
        self.WINNER = _BLANK
        self.TIME = time.time()
        self.MCTS = MCTSpure(c_puct=5, n_playout=10)
        self.MCTS = alphazero(PolicyValueNet(_BOARD_SIZE, _BOARD_SIZE).policy_value_fn, c_puct=5, n_playout=10)
        self.COORDINATE_1D = [_BOARD_SIZE_1D + 1]
        self.current_move = -1
        self.simula_time_list = []


def change_turn(cur):
    return cur % 2 + 1


# Initialize the game
if "ROOM" not in session_state:
    session_state.ROOM = Room("local")
if "OWNER" not in session_state:
    session_state.OWNER = False
if "USE_AIAID" not in session_state:
    session_state.USE_AIAID = False

# Check server health
if "ROOMS" not in server_state:
    with server_state_lock["ROOMS"]:
        server_state.ROOMS = {}

# # Layout
# Main
TITLE = st.empty()
TITLE.header("🤖 AI 3603 Gomoku")
ROUND_INFO = st.empty()
st.markdown("<br>", unsafe_allow_html=True)
BOARD_PLATE = [
    [cell.empty() for cell in st.columns([1 for _ in range(_BOARD_SIZE)])] for _ in range(_BOARD_SIZE)
]
LOG = st.empty()

# Sidebar
SCORE_TAG = st.sidebar.empty()
SCORE_PLATE = st.sidebar.columns(2)
# History scores
SCORE_TAG.subheader("Scores")

PLAY_MODE_INFO = st.sidebar.container()
MULTIPLAYER_TAG = st.sidebar.empty()
with st.sidebar.container():
    ANOTHER_ROUND = st.empty()
    RESTART = st.empty()
    AIAID = st.empty()
    EXIT = st.empty()
GAME_INFO = st.sidebar.container()
message = st.empty()
PLAY_MODE_INFO.write("---\n\n**You are Black, AI agent is White.**")
GAME_INFO.markdown(
    """
    ---
    # <span style="color:black;">Freestyle Gomoku game. 🎲</span>
    - no restrictions 🚫
    - no regrets 😎
    - swap players after one round is over 🔁
    Powered by an AlphaZero approach with our own improvements! 🚀 For the specific details, please check out our <a href="insert_report_link_here" style="color:blue;">report</a>.
    ##### Adapted and improved by us! 🌟  <a href="https://github.com/Lijiaxin0111/AI_3603_BIGHOME" style="color:blue;">Our Github repo</a>
    """,
    unsafe_allow_html=True,
)


def restart() -> None:
    """
    Restart the game.
    """
    session_state.ROOM = Room(session_state.ROOM.ROOM_ID)


RESTART.button(
    "Reset",
    on_click=restart,
    help="Clear the board as well as the scores",
)


# Draw the board
def gomoku():
    """
    Draw the board.
    Handle the main logic.
    """

    # Restart the game

    # Continue new round
    def another_round() -> None:
        """
        Continue new round.
        """
        session_state.ROOM = deepcopy(session_state.ROOM)
        session_state.ROOM.BOARD = Board(width=_BOARD_SIZE, height=_BOARD_SIZE, n_in_row=5)
        session_state.ROOM.PLAYER = session_state.ROOM.PLAYER
        session_state.ROOM.TURN = session_state.ROOM.PLAYER
        session_state.ROOM.WINNER = _BLANK  # 0
        session_state.ROOM.COORDINATE_1D = [_BOARD_SIZE_1D + 1]

    # Room status sync
    def sync_room() -> bool:
        room_id = session_state.ROOM.ROOM_ID
        if room_id not in server_state.ROOMS.keys():
            session_state.ROOM = Room("local")
            return False
        elif server_state.ROOMS[room_id].TIME == session_state.ROOM.TIME:
            return False
        elif server_state.ROOMS[room_id].TIME < session_state.ROOM.TIME:
            # Only acquire the lock when writing to the server state
            with server_state_lock["ROOMS"]:
                server_rooms = server_state.ROOMS
                server_rooms[room_id] = session_state.ROOM
                server_state.ROOMS = server_rooms
            return True
        else:
            session_state.ROOM = server_state.ROOMS[room_id]
            return True

    # Check if winner emerge from move
    def check_win() -> int:
        """
        Use convolution to check if any player wins.
        """
        vertical = convolve(
            session_state.ROOM.BOARD.board_map,
            _VERTICAL,
            mode="same",
        )
        horizontal = convolve(
            session_state.ROOM.BOARD.board_map,
            _HORIZONTAL,
            mode="same",
        )
        diagonal_up_left = convolve(
            session_state.ROOM.BOARD.board_map,
            _DIAGONAL_UP_LEFT,
            mode="same",
        )
        diagonal_up_right = convolve(
            session_state.ROOM.BOARD.board_map,
            _DIAGONAL_UP_RIGHT,
            mode="same",
        )
        if (
                np.max(
                    [
                        np.max(vertical),
                        np.max(horizontal),
                        np.max(diagonal_up_left),
                        np.max(diagonal_up_right),
                    ]
                )
                == 5 * _BLACK
        ):
            winner = _BLACK
        elif (
                np.min(
                    [
                        np.min(vertical),
                        np.min(horizontal),
                        np.min(diagonal_up_left),
                        np.min(diagonal_up_right),
                    ]
                )
                == 5 * _WHITE
        ):
            winner = _WHITE
        else:
            winner = _BLANK
        return winner

    def ai_aid() -> None:
        """
        Use AI Aid.
        """
        session_state.USE_AIAID = not session_state.USE_AIAID
        print('Use AI Aid: ', session_state.USE_AIAID)
        draw_board(False)

    # Triggers the board response on click
    def handle_click(x, y):
        """
        Controls whether to pass on / continue current board / may start new round
        """
        if session_state.ROOM.BOARD.board_map[x][y] != _BLANK:
            pass
        elif (
                session_state.ROOM.ROOM_ID in server_state.ROOMS.keys()
                and _ROOM_COLOR[session_state.OWNER]
                != server_state.ROOMS[session_state.ROOM.ROOM_ID].TURN
        ):
            sync_room()

        # normal play situation
        elif session_state.ROOM.WINNER == _BLANK:
            # session_state.ROOM = deepcopy(session_state.ROOM)
            print("View of human player: ", session_state.ROOM.BOARD.board_map)
            move = session_state.ROOM.BOARD.location_to_move((x, y))
            session_state.ROOM.current_move = move
            session_state.ROOM.BOARD.do_move(move)
            session_state.ROOM.BOARD.board_map[x][y] = session_state.ROOM.TURN
            session_state.ROOM.COORDINATE_1D.append(x * _BOARD_SIZE + y)

            session_state.ROOM.TURN = change_turn(session_state.ROOM.TURN)
            win, winner = session_state.ROOM.BOARD.game_end()
            if win:
                session_state.ROOM.WINNER = winner
            session_state.ROOM.HISTORY = (
                session_state.ROOM.HISTORY[0]
                + int(session_state.ROOM.WINNER == _WHITE),
                session_state.ROOM.HISTORY[1]
                + int(session_state.ROOM.WINNER == _BLACK),
            )
            session_state.ROOM.TIME = time.time()

    def forbid_click(x, y):
        # st.warning('This posistion has been occupied!!!!', icon="⚠️")
        st.error("({}, {}) has been occupied!!)".format(x, y), icon="🚨")
        print("asdas")

    # Draw board
    def draw_board(response: bool):
        """construct each buttons for all cells of the board"""
        if session_state.USE_AIAID and session_state.ROOM.WINNER == _BLANK:
            copy_mcts = deepcopy(session_state.ROOM.MCTS.mcts)
            _, acts, probs, simul_mean_time = copy_mcts.get_move_probs(session_state.ROOM.BOARD)
            sorted_acts_probs = sorted(zip(acts, probs), key=lambda x: x[1], reverse=True)
            top_five_acts = [act for act, prob in sorted_acts_probs[:5]]
            top_five_probs = [prob for act, prob in sorted_acts_probs[:5]]
        if response and session_state.ROOM.TURN == _BLACK:  # human turn
            print("Your turn")
            # construction of clickable buttons
            for i, row in enumerate(session_state.ROOM.BOARD.board_map):
                # print("row:", row)
                for j, cell in enumerate(row):
                    if (
                            i * _BOARD_SIZE + j
                            in (session_state.ROOM.COORDINATE_1D)
                    ):
                        # disable click for GPT choices
                        BOARD_PLATE[i][j].button(
                            _PLAYER_SYMBOL[cell],
                            key=f"{i}:{j}",
                            args=(i, j),
                            on_click=forbid_click
                        )
                    else:
                        if session_state.USE_AIAID and i * _BOARD_SIZE + j in top_five_acts:
                            # enable click for other cells available for human choices
                            prob = top_five_probs[top_five_acts.index(i * _BOARD_SIZE + j)]
                            BOARD_PLATE[i][j].button(
                                _PLAYER_SYMBOL[cell] + f"({round(prob, 2)})",
                                key=f"{i}:{j}",
                                on_click=handle_click,
                                args=(i, j),
                            )
                        else:
                            # enable click for other cells available for human choices
                            BOARD_PLATE[i][j].button(
                                _PLAYER_SYMBOL[cell],
                                key=f"{i}:{j}",
                                on_click=handle_click,
                                args=(i, j),
                            )


        elif response and session_state.ROOM.TURN == _WHITE:  # AI turn
            message.empty()
            with st.spinner('🔮✨ Waiting for AI response... ⏳🚀'):
                time.sleep(0.1)
                print("AI's turn")
                print("Below are current board under AI's view")
                print(session_state.ROOM.BOARD.board_map)
                move, simul_time = session_state.ROOM.MCTS.get_action(session_state.ROOM.BOARD, return_time=True)
                session_state.ROOM.simula_time_list.append(simul_time)
                print("AI takes move: ", move)
                session_state.ROOM.current_move = move
                gpt_response = move
                gpt_i, gpt_j = gpt_response // _BOARD_SIZE, gpt_response % _BOARD_SIZE
                print("AI's move is located at ({}, {}) :".format(gpt_i, gpt_j))
                move = session_state.ROOM.BOARD.location_to_move((gpt_i, gpt_j))
                print("Location to move: ", move)
                session_state.ROOM.BOARD.do_move(move)
                # session_state.ROOM.BOARD[gpt_i][gpt_j] = session_state.ROOM.TURN
                session_state.ROOM.COORDINATE_1D.append(gpt_i * _BOARD_SIZE + gpt_j)

                # construction of clickable buttons
                for i, row in enumerate(session_state.ROOM.BOARD.board_map):
                    # print("row:", row)
                    for j, cell in enumerate(row):
                        if (
                                i * _BOARD_SIZE + j
                                in (session_state.ROOM.COORDINATE_1D)
                        ):
                            # disable click for GPT choices
                            BOARD_PLATE[i][j].button(
                                _PLAYER_SYMBOL[cell],
                                key=f"{i}:{j}",
                                args=(i, j),
                                on_click=forbid_click
                            )
                        else:
                            if session_state.USE_AIAID and i * _BOARD_SIZE + j in top_five_acts:
                                # enable click for other cells available for human choices
                                prob = top_five_probs[top_five_acts.index(i * _BOARD_SIZE + j)]
                                BOARD_PLATE[i][j].button(
                                    _PLAYER_SYMBOL[cell] + f"({round(prob, 2)})",
                                    key=f"{i}:{j}",
                                    on_click=handle_click,
                                    args=(i, j),
                                )
                            else:
                                # enable click for other cells available for human choices
                                BOARD_PLATE[i][j].button(
                                    _PLAYER_SYMBOL[cell],
                                    key=f"{i}:{j}",
                                    on_click=handle_click,
                                    args=(i, j),
                                )


            message.markdown(
                'AI agent has calculated its strategy, which takes <span style="color: blue; font-size: 20px;">{:.3e}</span>s per simulation.'.format(
                    simul_time),
                unsafe_allow_html=True
            )
            LOG.subheader("Logs")
            # change turn
            session_state.ROOM.TURN = change_turn(session_state.ROOM.TURN)
            # session_state.ROOM.WINNER = check_win()

            win, winner = session_state.ROOM.BOARD.game_end()
            if win:
                session_state.ROOM.WINNER = winner

            session_state.ROOM.HISTORY = (
                session_state.ROOM.HISTORY[0]
                + int(session_state.ROOM.WINNER == _WHITE),
                session_state.ROOM.HISTORY[1]
                + int(session_state.ROOM.WINNER == _BLACK),
            )
            session_state.ROOM.TIME = time.time()

        if not response or session_state.ROOM.WINNER != _BLANK:
            print("Game over")
            for i, row in enumerate(session_state.ROOM.BOARD.board_map):
                for j, cell in enumerate(row):
                    BOARD_PLATE[i][j].write(
                        _PLAYER_SYMBOL[cell],
                        # key=f"{i}:{j}",
                    )

    # Game process control
    def game_control():
        if session_state.ROOM.WINNER != _BLANK:
            draw_board(False)
        else:
            draw_board(True)
        if session_state.ROOM.WINNER != _BLANK or 0 not in session_state.ROOM.BOARD.board_map:
            ANOTHER_ROUND.button(
                "Play Next round!",
                on_click=another_round,
                help="Clear board and swap first player",
            )

    # Infos
    def update_info() -> None:
        # Additional information
        SCORE_PLATE[0].metric("Gomoku-Agent", session_state.ROOM.HISTORY[0])
        SCORE_PLATE[1].metric("Black", session_state.ROOM.HISTORY[1])
        if session_state.ROOM.WINNER != _BLANK:
            st.balloons()
            ROUND_INFO.write(
                f"#### **{_PLAYER_COLOR[session_state.ROOM.WINNER]} WIN!**\n**Click buttons on the left for more plays.**"
            )

        # elif 0 not in session_state.ROOM.BOARD.board_map:
        #     ROUND_INFO.write("#### **Tie**")
        # else:
        #     ROUND_INFO.write(
        #         f"#### **{_PLAYER_SYMBOL[session_state.ROOM.TURN]} {_PLAYER_COLOR[session_state.ROOM.TURN]}'s turn...**"
        #     )

        # draw the plot for simulation time
        # 创建一个 DataFrame

        print(session_state.ROOM.simula_time_list)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        chart_data = pd.DataFrame(session_state.ROOM.simula_time_list, columns=["Simulation Time"])
        st.line_chart(chart_data)

    # The main game loop
    AIAID.button(
        "Use AI Aid",
        on_click=ai_aid,
        help="Use AI Aid to help you make moves",
    )
    game_control()
    update_info()


if __name__ == "__main__":
    gomoku()
