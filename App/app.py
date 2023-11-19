"""
- This is a simple gomoku game built with Streamlit by TeddyHuang-00 (huang_nan_2019@pku.edu.cn).
- For Gomoku-GPT2, please refer to Young-Jin Ahn (young_ahn@yonsei.ac.kr).

Shared under MIT license
"""

import time
from copy import deepcopy
from uuid import uuid4

import torch
import numpy as np
import streamlit as st
from scipy.signal import convolve # this is used to check if any player wins
from streamlit import session_state
from streamlit_server_state import server_state, server_state_lock

from ai import (
    BOS_TOKEN_ID,
    generate_gpt2,
    load_model,
)


# Utils
class Room:
    def __init__(self, room_id) -> None:
        self.ROOM_ID = room_id
        self.BOARD = np.zeros(shape=(20, 20), dtype=int)
        self.PLAYER = _BLACK
        self.TURN = self.PLAYER
        self.HISTORY = (0, 0)
        self.WINNER = _BLANK
        self.TIME = time.time()
        self.COORDINATE_1D = [BOS_TOKEN_ID]


gpt2 = load_model()


_BLANK = 0
_BLACK = 1
_WHITE = -1
_PLAYER_SYMBOL = {
    _WHITE: "⚪",
    _BLANK: "➕",
    _BLACK: "⚫",
}
_PLAYER_COLOR = {
    _WHITE: "Gomoku-GPT",
    _BLANK: "Blank",
    _BLACK: "YOU HUMAN",
}
_HORIZONTAL = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
_VERTICAL = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)
_DIAGONAL_UP_LEFT = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
)
_DIAGONAL_UP_RIGHT = np.array(
    [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
)

_ROOM_COLOR = {
    True: _BLACK,
    False: _WHITE,
}

# Initialize the game
if "ROOM" not in session_state:
    session_state.ROOM = Room("local")
if "OWNER" not in session_state:
    session_state.OWNER = False

# Check server health
if "ROOMS" not in server_state:
    with server_state_lock["ROOMS"]:
        server_state.ROOMS = {}

# # Layout
# Main
TITLE = st.empty()
ROUND_INFO = st.empty()
BOARD_PLATE = [
    [cell.empty() for cell in st.columns([1 for _ in range(20)])] for _ in range(20)
]
WAIT_FOR_OPPONENT = st.empty()

# Sidebar
SCORE_TAG = st.sidebar.empty()
SCORE_PLATE = st.sidebar.columns(2)
PLAY_MODE_INFO = st.sidebar.container()
MULTIPLAYER_TAG = st.sidebar.empty()
with st.sidebar.container():
    ANOTHER_ROUND = st.empty()
    RESTART = st.empty()
    EXIT = st.empty()
GAME_INFO = st.sidebar.container()


# Draw the board
def gomoku():
    """
    Draw the board.

    Handle the main logic.
    """

    # Restart the game
    def restart() -> None:
        """
        Restart the game.
        """
        session_state.ROOM = Room(session_state.ROOM.ROOM_ID)

    # Continue new round
    def another_round() -> None:
        """
        Continue new round.
        """
        session_state.ROOM = deepcopy(session_state.ROOM)
        session_state.ROOM.BOARD = np.zeros(shape=(20, 20), dtype=int)
        session_state.ROOM.PLAYER = -session_state.ROOM.PLAYER
        session_state.ROOM.TURN = session_state.ROOM.PLAYER
        session_state.ROOM.WINNER = _BLANK
        session_state.ROOM.COORDINATE_1D = [BOS_TOKEN_ID]

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
            session_state.ROOM.BOARD,
            _VERTICAL,
            mode="same",
        )
        horizontal = convolve(
            session_state.ROOM.BOARD,
            _HORIZONTAL,
            mode="same",
        )
        diagonal_up_left = convolve(
            session_state.ROOM.BOARD,
            _DIAGONAL_UP_LEFT,
            mode="same",
        )
        diagonal_up_right = convolve(
            session_state.ROOM.BOARD,
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

    # Triggers the board response on click
    def handle_click(x, y):
        """
        Controls whether to pass on / continue current board / may start new round
        """
        if session_state.ROOM.BOARD[x][y] != _BLANK:
            pass
        elif (
            session_state.ROOM.ROOM_ID in server_state.ROOMS.keys()
            and _ROOM_COLOR[session_state.OWNER]
            != server_state.ROOMS[session_state.ROOM.ROOM_ID].TURN
        ):
            sync_room()

        # normal play situation
        elif session_state.ROOM.WINNER == _BLANK:
            session_state.ROOM = deepcopy(session_state.ROOM)

            session_state.ROOM.BOARD[x][y] = session_state.ROOM.TURN
            session_state.ROOM.COORDINATE_1D.append(x * 20 + y)

            session_state.ROOM.TURN = -session_state.ROOM.TURN
            session_state.ROOM.WINNER = check_win()
            session_state.ROOM.HISTORY = (
                session_state.ROOM.HISTORY[0]
                + int(session_state.ROOM.WINNER == _WHITE),
                session_state.ROOM.HISTORY[1]
                + int(session_state.ROOM.WINNER == _BLACK),
            )
            session_state.ROOM.TIME = time.time()

    # Draw board
    def draw_board(response: bool):
        """construct each buttons for all cells of the board"""

        if response and session_state.ROOM.TURN == 1:  # human turn
            print("Your turn")
            # construction of clickable buttons
            for i, row in enumerate(session_state.ROOM.BOARD):
                for j, cell in enumerate(row):
                    BOARD_PLATE[i][j].button(
                        _PLAYER_SYMBOL[cell],
                        key=f"{i}:{j}",
                        on_click=handle_click,
                        args=(i, j),
                    )

        elif response and session_state.ROOM.TURN == -1:  # AI turn
            print("AI's turn")
            gpt_predictions = generate_gpt2(
                gpt2,
                torch.tensor(session_state.ROOM.COORDINATE_1D).unsqueeze(0),
            )
            print(gpt_predictions)
            gpt_response = gpt_predictions[len(session_state.ROOM.COORDINATE_1D)]
            gpt_i, gpt_j = gpt_response // 20, gpt_response % 20
            print(gpt_i, gpt_j)
            session_state.ROOM.BOARD[gpt_i][gpt_j] = session_state.ROOM.TURN
            session_state.ROOM.COORDINATE_1D.append(gpt_i * 20 + gpt_j)

            # construction of clickable buttons
            for i, row in enumerate(session_state.ROOM.BOARD):
                for j, cell in enumerate(row):
                    if (
                        i * 20 + j
                        in gpt_predictions[: len(session_state.ROOM.COORDINATE_1D)]
                    ):
                        # disable click for GPT choices
                        BOARD_PLATE[i][j].button(
                            _PLAYER_SYMBOL[cell],
                            key=f"{i}:{j}",
                            on_click=False,
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

            # change turn
            session_state.ROOM.TURN = -session_state.ROOM.TURN
            session_state.ROOM.WINNER = check_win()
            session_state.ROOM.HISTORY = (
                session_state.ROOM.HISTORY[0]
                + int(session_state.ROOM.WINNER == _WHITE),
                session_state.ROOM.HISTORY[1]
                + int(session_state.ROOM.WINNER == _BLACK),
            )
            session_state.ROOM.TIME = time.time()

        if not response or session_state.ROOM.WINNER != _BLANK:
            print("Game over")
            for i, row in enumerate(session_state.ROOM.BOARD):
                for j, cell in enumerate(row):
                    BOARD_PLATE[i][j].write(
                        _PLAYER_SYMBOL[cell],
                        key=f"{i}:{j}",
                    )

    # Game process control
    def game_control():
        if session_state.ROOM.WINNER != _BLANK:
            draw_board(False)
        else:
            draw_board(True)
        if session_state.ROOM.WINNER != _BLANK or 0 not in session_state.ROOM.BOARD:
            ANOTHER_ROUND.button(
                "Play Next round!",
                on_click=another_round,
                help="Clear board and swap first player",
            )
        if session_state.ROOM.ROOM_ID == "local" or session_state.OWNER:
            RESTART.button(
                "Reset",
                on_click=restart,
                help="Clear the board as well as the scores",
            )

    # Infos
    def draw_info() -> None:
        # Text information
        TITLE.subheader("**🤖 Do you wanna have a bad time?**")
        PLAY_MODE_INFO.write("---\n\n**You are Black, AI is White.**")
        GAME_INFO.markdown(
            """
            ---

            ## Freestyle Gomoku game.


            <a href="https://en.wikipedia.org/wiki/Gomoku#Freestyle_Gomoku" style="color:#FFFFFF">Freestyle Gomoku</a>

            - no restrictions
            - no regrets
            - swap players after one round is over

            ##### Design by <a href="https://github.com/TeddyHuang-00" style="color:#FFFFFF">TeddyHuang-00</a> • <a href="https://github.com/TeddyHuang-00/streamlit-gomoku" style="color:#FFFFFF">Github repo</a>
            ##### Gomoku-GPT by <a href="https://github.com/snoop2head" style="color:#FFFFFF">snoop2head</a> • <a href="https://github.com/snoop2head/" style="color:#FFFFFF">Github repo</a>

            """,
            unsafe_allow_html=True,
        )
        # History scores
        SCORE_TAG.subheader("Scores")
        SCORE_PLATE[0].metric("Gomoku-GPT", session_state.ROOM.HISTORY[0])
        SCORE_PLATE[1].metric("Black", session_state.ROOM.HISTORY[1])

        # Additional information
        if session_state.ROOM.WINNER != _BLANK:
            st.balloons()
            ROUND_INFO.write(
                f"#### **{_PLAYER_COLOR[session_state.ROOM.WINNER]} wins!**\n**Click buttons on the left for more plays.**"
            )

        elif 0 not in session_state.ROOM.BOARD:
            ROUND_INFO.write("#### **Tie**")
        else:
            ROUND_INFO.write(
                f"#### **{_PLAYER_SYMBOL[session_state.ROOM.TURN]} {_PLAYER_COLOR[session_state.ROOM.TURN]}'s turn...**"
            )

    # The main game loop
    game_control()
    draw_info()


if __name__ == "__main__":
    gomoku()
