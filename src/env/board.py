"""
env/board.py
Core board logic for a larger three-player Connect Four game.

Board size:
    ROWS = 8
    COLS = 10

Cells:
    EMPTY = 0
    P1    = 1
    P2    = 2
    P3    = 3
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Tuple

ROWS = 8
COLS = 10
CONNECT_N = 4

EMPTY = 0
P1 = 1
P2 = 2
P3 = 3
PLAYERS = (P1, P2, P3)


class Board:
    def __init__(self, grid: Optional[List[List[int]]] = None) -> None:
        if grid is None:
            self.grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        else:
            if len(grid) != ROWS or any(len(row) != COLS for row in grid):
                raise ValueError(f"grid must be {ROWS}x{COLS}")
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
        """
        if player not in PLAYERS:
            raise ValueError(f"player must be one of {PLAYERS}")
        if not self.is_valid_move(col):
            raise ValueError(f"Illegal move: column {col}")

        for r in range(ROWS - 1, -1, -1):
            if self.grid[r][col] == EMPTY:
                self.grid[r][col] = player
                return r
        raise ValueError(f"Column {col} is full")

    def undo_drop(self, col: int) -> None:
        """Remove the top-most piece from a column."""
        for r in range(ROWS):
            if self.grid[r][col] != EMPTY:
                self.grid[r][col] = EMPTY
                return
        raise ValueError(f"Column {col} is already empty")

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
        """Return winning player, 0 for draw, or None if game is not over."""
        for player in PLAYERS:
            if self.check_win(player):
                return player
        if self.is_draw():
            return 0
        return None

    def is_draw(self) -> bool:
        return len(self.legal_moves()) == 0 and all(not self.check_win(p) for p in PLAYERS)

    def encode_for_player(self, player: int) -> Tuple[int, ...]:
        """
        Encode board from current player's perspective for RL:
        current player's pieces -> 1
        any opponent piece      -> -1
        empty                   -> 0
        """
        if player not in PLAYERS:
            raise ValueError(f"player must be one of {PLAYERS}")

        out: list[int] = []
        for row in self.grid:
            for cell in row:
                if cell == EMPTY:
                    out.append(0)
                elif cell == player:
                    out.append(1)
                else:
                    out.append(-1)
        return tuple(out)

    def pretty(self) -> str:
        chars = {EMPTY: ".", P1: "X", P2: "O", P3: "A"}
        lines = [" ".join(chars[cell] for cell in row) for row in self.grid]
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()


def next_player(player: int) -> int:
    """Return the next player in cyclic turn order P1 -> P2 -> P3 -> P1."""
    if player not in PLAYERS:
        raise ValueError(f"player must be one of {PLAYERS}")
    idx = PLAYERS.index(player)
    return PLAYERS[(idx + 1) % len(PLAYERS)]


def opponents(player: int) -> tuple[int, int]:
    """Return the two opponents of a player."""
    if player not in PLAYERS:
        raise ValueError(f"player must be one of {PLAYERS}")
    return tuple(p for p in PLAYERS if p != player)  # type: ignore[return-value]


def other_player(player: int) -> int:
    """
    Backward-compatible helper from the earlier two-player version.
    In the three-player version, this means the next player in turn order.
    """
    return next_player(player)
