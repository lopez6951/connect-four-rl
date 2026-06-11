"""
env/game.py
Two-player Connect Four game loop.
"""

from __future__ import annotations
from typing import Any, List, Tuple

from env.board import Board, P1, P2, other_player


class Game:
    def __init__(self, p1: Any, p2: Any, verbose: bool = False) -> None:
        self.board = Board()
        self.players = {P1: p1, P2: p2}
        self.current = P1
        self.verbose = verbose

    def play(self) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Play until terminal.
        Returns (winner, history). winner is P1/P2/0 for draw.
        history contains (player, column) moves.
        """
        history: List[Tuple[int, int]] = []

        while True:
            if self.verbose:
                print(self.board)
                print()

            player_obj = self.players[self.current]
            col = player_obj.choose_move(self.board.copy(), self.current)
            if col not in self.board.legal_moves():
                # Illegal move loses immediately.
                return other_player(self.current), history

            self.board.drop(col, self.current)
            history.append((self.current, col))

            if self.board.check_win(self.current):
                if self.verbose:
                    print(self.board)
                    print(f"Player {self.current} wins!")
                return self.current, history

            if self.board.is_draw():
                if self.verbose:
                    print(self.board)
                    print("Draw!")
                return 0, history

            self.current = other_player(self.current)
