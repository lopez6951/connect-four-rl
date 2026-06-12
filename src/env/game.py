"""
game.py - Two-player Connect Four game loop
"""

from typing import Any
from env.board import Board, P1, P2

class Game:
    '''to run game'''

    def __init__(self, player1: Any, player2: Any, verbose: bool = True) -> None:
        '''Store players in a dict'''
        self.players = {P1: player1, P2: player2}
        self.verbose = verbose

    def play(self) -> tuple[int, list[tuple[Any, int, int]]]:
        '''Run a full game to completion'''
        board   = Board()
        history: list[tuple[Any, int, int]] = []
        current = P1

        while True:
            if self.verbose:
                print(f"\n{'Player 1 (X)' if current == P1 else 'Player 2 (O)'}'s turn")
                print(board)

            # Ask the current player to pick a column
            col = self.players[current].choose_action(board, current)

            # If player returns illegal move, opp wins
            if not board.is_legal(col):
                winner = P2 if current == P1 else P1
                if self.verbose:
                    print("Illegal move! Opponent wins.")
                return winner, history

            # record + apply to board
            history.append((board.to_state(current), col, current))
            board.drop(col, current)

            # check if won game
            if board.check_win(current):
                if self.verbose:
                    print(board)
                    print(f"\n{'Player 1 (X)' if current == P1 else 'Player 2 (O)'} wins!")
                return current, history

            # check if draw
            if board.is_draw():
                if self.verbose:
                    print(board)
                    print("\nIt's a draw!")
                return 0, history

            # then switch
            current = P2 if current == P1 else P1
