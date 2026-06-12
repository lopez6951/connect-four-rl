"""
tree/search.py
Optional shallow heuristic search helpers for three-player Connect Four.

The main project uses Q-learning. This file is kept only as an optional
baseline/extension.
"""

from __future__ import annotations

from env.board import Board, COLS, ROWS, CONNECT_N, EMPTY, opponents


def score_window(window: list[int], player: int) -> int:
    opps = opponents(player)
    if window.count(player) == CONNECT_N:
        return 1000
    if window.count(player) == 3 and window.count(EMPTY) == 1:
        return 50
    if window.count(player) == 2 and window.count(EMPTY) == 2:
        return 10
    for opp in opps:
        if window.count(opp) == 3 and window.count(EMPTY) == 1:
            return -80
    return 0


def evaluate(board: Board, player: int) -> int:
    if board.check_win(player):
        return 100000
    for opp in opponents(player):
        if board.check_win(opp):
            return -100000

    score = 0
    center = COLS // 2
    center_cells = [board.grid[r][center] for r in range(ROWS)]
    score += center_cells.count(player) * 6

    for r in range(ROWS):
        for c in range(COLS - CONNECT_N + 1):
            score += score_window([board.grid[r][c + i] for i in range(CONNECT_N)], player)

    for r in range(ROWS - CONNECT_N + 1):
        for c in range(COLS):
            score += score_window([board.grid[r + i][c] for i in range(CONNECT_N)], player)

    for r in range(ROWS - CONNECT_N + 1):
        for c in range(COLS - CONNECT_N + 1):
            score += score_window([board.grid[r + i][c + i] for i in range(CONNECT_N)], player)

    for r in range(CONNECT_N - 1, ROWS):
        for c in range(COLS - CONNECT_N + 1):
            score += score_window([board.grid[r - i][c + i] for i in range(CONNECT_N)], player)

    return score


def greedy_search_move(board: Board, player: int) -> int:
    """Choose the legal move with the highest immediate heuristic score."""
    legal = board.legal_moves()
    best_col = legal[0]
    best_score = -10**18
    for col in legal:
        b = board.copy()
        b.drop(col, player)
        score = evaluate(b, player)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col
