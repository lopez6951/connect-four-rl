import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.board import Board, P1, P2


def test_horizontal_win():
    b = Board()
    for c in [0, 1, 2, 3]:
        b.drop(c, P1)
    assert b.check_win(P1)


def test_vertical_win():
    b = Board()
    for _ in range(4):
        b.drop(0, P2)
    assert b.check_win(P2)


def test_diagonal_win():
    b = Board()
    # Build positive-slope diagonal for P1.
    b.drop(0, P1)
    b.drop(1, P2); b.drop(1, P1)
    b.drop(2, P2); b.drop(2, P2); b.drop(2, P1)
    b.drop(3, P2); b.drop(3, P2); b.drop(3, P2); b.drop(3, P1)
    assert b.check_win(P1)
