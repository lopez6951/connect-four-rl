"""
self_play/loop.py
Helper for running repeated three-player games.
"""

from __future__ import annotations
from typing import Any

from env.board import P1, P2, P3
from env.game import Game


def play_many(p1: Any, p2: Any, p3: Any, n_games: int = 100) -> dict[str, int]:
    results = {"p1_wins": 0, "p2_wins": 0, "p3_wins": 0, "draws": 0}
    for _ in range(n_games):
        winner, _ = Game(p1, p2, p3, verbose=False).play()
        if winner == P1:
            results["p1_wins"] += 1
        elif winner == P2:
            results["p2_wins"] += 1
        elif winner == P3:
            results["p3_wins"] += 1
        else:
            results["draws"] += 1
    return results
