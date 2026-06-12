"""
self_play/player.py
Baseline players and wrappers for trained agents.
"""

from __future__ import annotations
import random
from typing import Protocol

from env.board import Board, COLS, PLAYERS, opponents


class Player(Protocol):
    def choose_move(self, board: Board, player: int) -> int:
        ...


class RandomPlayer:
    def choose_move(self, board: Board, player: int) -> int:
        return random.choice(board.legal_moves())


class GreedyPlayer:
    """
    Simple three-player baseline:
    1. Win immediately if possible.
    2. Block any opponent's immediate winning move.
    3. Prefer center columns.
    """

    def choose_move(self, board: Board, player: int) -> int:
        legal = board.legal_moves()

        # Take winning move.
        for col in legal:
            b = board.copy()
            b.drop(col, player)
            if b.check_win(player):
                return col

        # Block either opponent's immediate winning move.
        for opponent in opponents(player):
            for col in legal:
                b = board.copy()
                b.drop(col, opponent)
                if b.check_win(opponent):
                    return col

        # Prefer center columns on the larger board.
        center = COLS // 2
        preferred = sorted(legal, key=lambda c: abs(c - center))
        return preferred[0] if preferred else random.choice(legal)


class HumanPlayer:
    def choose_move(self, board: Board, player: int) -> int:
        while True:
            try:
                col = int(input(f"Player {player}, choose column {board.legal_moves()}: "))
                if col in board.legal_moves():
                    return col
                print("Illegal column. Try again.")
            except ValueError:
                print(f"Please enter a number from 0 to {COLS - 1}.")


class QLearningPlayer:
    def __init__(self, agent: object) -> None:
        self.agent = agent

    def choose_move(self, board: Board, player: int) -> int:
        return self.agent.choose_action(board, player, training=False)
