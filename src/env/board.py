"""
env/board.py
Core Connect Four board logic.

Cells:
    EMPTY = 0
    P1    = 1
    P2    = 2
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Tuple

ROWS = 6
COLS = 7
CONNECT_N = 4

EMPTY = 0
P1 = 1
P2 = 2


class Board:
    def __init__(self, grid: Optional[List[List[int]]] = None) -> None:
        if grid is None:
            self.grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        else:
            self.grid = deepcopy(grid)

    def copy(self) -> "Board":
        return Board(self.grid)

    def legal_moves(self) -> List[int]:
        """Return columns that are not full."""
        return [c for c in range(COLS) if self.grid[0][c] == EMPTY]

    def is_valid_move(self, col: int) -> bool:
        return 0 <= col < COLS and self.grid[0][col] == EMPTY

    def drop(self, col: int, player: int) -> int:
        """
        Drop a piece into a column.
        Returns the row where the piece lands.
        Raises ValueError if the move is illegal.
        """
        if player not in (P1, P2):
            raise ValueError("player must be P1 or P2")
        if not self.is_valid_move(col):
            raise ValueError(f"Illegal move: column {col}")

        for r in range(ROWS - 1, -1, -1):
            if self.grid[r][col] == EMPTY:
                self.grid[r][col] = player
                return r
        raise ValueError(f"Column {col} is full")

    def undo_drop(self, col: int) -> None:
        """Remove the top-most piece from a column. Useful for search."""
        for r in range(ROWS):
            if self.grid[r][col] != EMPTY:
                self.grid[r][col] = EMPTY
                return
        raise ValueError(f"Column {col} is already empty")

    def is_draw(self) -> bool:
        return len(self.legal_moves()) == 0 and not self.check_win(P1) and not self.check_win(P2)

    def check_win(self, player: int) -> bool:
        """Check horizontal, vertical, and diagonal connect-four."""
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - CONNECT_N + 1):
                if all(self.grid[r][c + i] == player for i in range(CONNECT_N)):
                    return True

        # Vertical
        for r in range(ROWS - CONNECT_N + 1):
            for c in range(COLS):
                if all(self.grid[r + i][c] == player for i in range(CONNECT_N)):
                    return True

        # Diagonal down-right
        for r in range(ROWS - CONNECT_N + 1):
            for c in range(COLS - CONNECT_N + 1):
                if all(self.grid[r + i][c + i] == player for i in range(CONNECT_N)):
                    return True

        # Diagonal up-right
        for r in range(CONNECT_N - 1, ROWS):
            for c in range(COLS - CONNECT_N + 1):
                if all(self.grid[r - i][c + i] == player for i in range(CONNECT_N)):
                    return True

        return False

    def terminal_winner(self) -> Optional[int]:
        """Return P1/P2 for winner, 0 for draw, or None if game is not over."""
        if self.check_win(P1):
            return P1
        if self.check_win(P2):
            return P2
        if self.is_draw():
            return 0
        return None

    def encode_for_player(self, player: int) -> Tuple[int, ...]:
        """
        Encode board from current player's perspective for RL:
        current player's pieces -> 1, opponent pieces -> -1, empty -> 0.
        """
        opponent = P2 if player == P1 else P1
        out: list[int] = []
        for row in self.grid:
            for cell in row:
                if cell == player:
                    out.append(1)
                elif cell == opponent:
                    out.append(-1)
                else:
                    out.append(0)
        return tuple(out)

    def pretty(self) -> str:
        chars = {EMPTY: ".", P1: "X", P2: "O"}
        lines = [" ".join(chars[cell] for cell in row) for row in self.grid]
        lines.append("0 1 2 3 4 5 6")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()


def other_player(player: int) -> int:
    if player == P1:
        return P2
    if player == P2:
        return P1
    raise ValueError("player must be P1 or P2")
