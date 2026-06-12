"""
board.py — Connect Four board state, legal moves, win/draw detection
"""

import numpy as np
from typing import List

ROWS = 6
COLS = 7
EMPTY = 0
P1 = 1
P2 = -1


class Board:
    '''Create a 6x7 grid filled with empty cells'''
    def __init__(self) -> None:
        self.grid = np.zeros((ROWS, COLS), dtype=int)

    def copy(self) -> "Board":
        '''Return deep copy so changes dont affect orig'''
        b = Board()
        b.grid = self.grid.copy()
        return b

    # Moves
    def legal_moves(self) -> List[int]:
        '''A column is legal if the top row is still empty'''
        return [c for c in range(COLS) if self.grid[0][c] == EMPTY]

    def is_legal(self, col: int) -> bool:
        '''Check column is in bounds and not full'''
        return 0 <= col < COLS and self.grid[0][col] == EMPTY

    def drop(self, col: int, player: int) -> int:
        '''Drop a piece. Returns the row where the piece landed, or -1 if illegal'''
        if not self.is_legal(col):
            return -1
        for row in range(ROWS - 1, -1, -1):
            if self.grid[row][col] == EMPTY:
                self.grid[row][col] = player
                return row
        return -1  # unreachable if is_legal passed


    # Terminal checks
    def check_win(self, player: int) -> bool:
        '''Return True if player has four in a row'''
        g = self.grid
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(g[r][c + i] == player for i in range(4)):
                    return True
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(g[r + i][c] == player for i in range(4)):
                    return True
        # Diagonal top-left
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(g[r + i][c + i] == player for i in range(4)):
                    return True
        # Diagonal top-right
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                if all(g[r + i][c - i] == player for i in range(4)):
                    return True
        return False

    def is_draw(self) -> bool:
        '''Return True + draw if no legal moves remain'''
        return len(self.legal_moves()) == 0

    def is_terminal(self) -> bool:
        '''Game over if either player has won or the board is full'''
        return self.check_win(P1) or self.check_win(P2) or self.is_draw()


    # State representation for Q-learning
    def to_state(self, perspective: int = P1) -> np.ndarray:
        '''flatten grid into 42 elem array'''
        # for agent- own pieces = 1, opponent = -1, empty = 0
        flat = self.grid.flatten().astype(float)
        if perspective == P2:
            flat = flat * -1
        return flat

    def encode_for_player(self, player: int) -> tuple:
        '''Encode board as tuple for Q-table lookup from player perspective'''
        opponent = P2 if player == P1 else P1
        out = []
        for row in self.grid:
            for cell in row:
                if cell == player:
                    out.append(1)
                elif cell == opponent:
                    out.append(-1)
                else:
                    out.append(0)
        return tuple(out)

    # Display
    def __str__(self) -> str:
        # Print a bordered grid with X for P1, O for P2, empty for blank
        lines = []
        lines.append("┌" + "───┬" * (COLS - 1) + "───┐")
        for r in range(ROWS):
            row = "|"
            for c in range(COLS):
                v = self.grid[r][c]
                if v == P1:
                    row += " X |"
                elif v == P2:
                    row += " O |"
                else:
                    row += "   |"
            lines.append(row)
            if r < ROWS - 1:
                lines.append("├" + "───┼" * (COLS - 1) + "───┤")
        lines.append("└" + "───┴" * (COLS - 1) + "───┘")
        lines.append("  " + "   ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)

    #other player
def other_player(player: int) -> int:
    '''Return the other player constant'''
    if player == P1:
        return P2
    if player == P2:
        return P1
    raise ValueError("player must be P1 or P2")
