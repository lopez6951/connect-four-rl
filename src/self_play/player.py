"""
self_play/player.py
Baseline players and wrappers for trained agents.
"""

from __future__ import annotations
import random
from typing import Protocol

from env.board import Board, P1, P2, other_player


class Player(Protocol):
    def choose_move(self, board: Board, player: int) -> int:
        ...


class RandomPlayer:
    def choose_move(self, board: Board, player: int) -> int:
        return random.choice(board.legal_moves())


class GreedyPlayer:
    """
    Simple baseline:
    1. Win immediately if possible.
    2. Block opponent's immediate win if possible.
    3. Prefer center columns.
    """

    def choose_move(self, board: Board, player: int) -> int:
        legal = board.legal_moves()
        opponent = other_player(player)

        # Take winning move.
        for col in legal:
            b = board.copy()
            b.drop(col, player)
            if b.check_win(player):
                return col

        # Block opponent winning move.
        for col in legal:
            b = board.copy()
            b.drop(col, opponent)
            if b.check_win(opponent):
                return col

        # Prefer center columns.
        preferred = [3, 2, 4, 1, 5, 0, 6]
        for col in preferred:
            if col in legal:
                return col

        return random.choice(legal)


class HumanPlayer:
    def choose_move(self, board: Board, player: int) -> int:
        while True:
            try:
                col = int(input(f"Player {player}, choose column {board.legal_moves()}: "))
                if col in board.legal_moves():
                    return col
                print("Illegal column. Try again.")
            except ValueError:
                print("Please enter a number from 0 to 6.")


class QLearningPlayer:
    def __init__(self, agent: object) -> None:
        self.agent = agent

    def choose_move(self, board: Board, player: int) -> int:
        # Agent object must implement choose_action(board, player, training=False)
        return self.agent.choose_action(board, player, training=False)
