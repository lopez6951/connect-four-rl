"""
tree/search.py
Optional minimax baseline for comparison with RL.
"""

from __future__ import annotations
import math

from env.board import Board, P1, P2, other_player


def score_window(window: list[int], player: int) -> int:
    opponent = other_player(player)
    if window.count(player) == 4:
        return 1000
    if window.count(player) == 3 and window.count(0) == 1:
        return 50
    if window.count(player) == 2 and window.count(0) == 2:
        return 10
    if window.count(opponent) == 3 and window.count(0) == 1:
        return -80
    return 0


def evaluate(board: Board, player: int) -> int:
    if board.check_win(player):
        return 100000
    if board.check_win(other_player(player)):
        return -100000

    score = 0
    center = [board.grid[r][3] for r in range(6)]
    score += center.count(player) * 6

    # Horizontal windows
    for r in range(6):
        for c in range(4):
            score += score_window([board.grid[r][c + i] for i in range(4)], player)

    # Vertical windows
    for r in range(3):
        for c in range(7):
            score += score_window([board.grid[r + i][c] for i in range(4)], player)

    # Diagonals
    for r in range(3):
        for c in range(4):
            score += score_window([board.grid[r + i][c + i] for i in range(4)], player)
    for r in range(3, 6):
        for c in range(4):
            score += score_window([board.grid[r - i][c + i] for i in range(4)], player)
    return score


def minimax(board: Board, depth: int, player: int, maximizing: bool, root_player: int) -> tuple[int, int | None]:
    winner = board.terminal_winner()
    if depth == 0 or winner is not None:
        return evaluate(board, root_player), None

    legal = board.legal_moves()
    best_col = legal[0] if legal else None

    if maximizing:
        best_score = -math.inf
        for col in legal:
            b = board.copy()
            b.drop(col, player)
            score, _ = minimax(b, depth - 1, other_player(player), False, root_player)
            if score > best_score:
                best_score = score
                best_col = col
        return int(best_score), best_col
    else:
        best_score = math.inf
        for col in legal:
            b = board.copy()
            b.drop(col, player)
            score, _ = minimax(b, depth - 1, other_player(player), True, root_player)
            if score < best_score:
                best_score = score
                best_col = col
        return int(best_score), best_col
