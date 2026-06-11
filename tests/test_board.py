import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.board import Board, P1, P2, ROWS


def test_empty_board_has_all_legal_moves():
    b = Board()
    assert b.legal_moves() == [0, 1, 2, 3, 4, 5, 6]


def test_drop_piece_goes_to_bottom_row():
    b = Board()
    row = b.drop(3, P1)
    assert row == ROWS - 1
    assert b.grid[ROWS - 1][3] == P1


def test_full_column_is_not_legal():
    b = Board()
    for i in range(ROWS):
        b.drop(0, P1 if i % 2 == 0 else P2)
    assert 0 not in b.legal_moves()
