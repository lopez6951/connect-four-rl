import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.board import Board, P1, P2, P3, ROWS, COLS


def test_board_size_is_larger_for_three_players():
    b = Board()
    assert ROWS == 8
    assert COLS == 10
    assert len(b.grid) == 8
    assert len(b.grid[0]) == 10


def test_empty_board_has_all_legal_moves():
    b = Board()
    assert b.legal_moves() == list(range(COLS))


def test_drop_piece_goes_to_bottom_row():
    b = Board()
    row = b.drop(4, P1)
    assert row == ROWS - 1
    assert b.grid[ROWS - 1][4] == P1


def test_full_column_is_not_legal():
    b = Board()
    players = [P1, P2, P3]
    for i in range(ROWS):
        b.drop(0, players[i % 3])
    assert 0 not in b.legal_moves()
