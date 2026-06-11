"""
tree/node.py
Small node container for optional game-tree search experiments.
"""

from __future__ import annotations
from dataclasses import dataclass
from env.board import Board


@dataclass
class Node:
    board: Board
    player: int
    depth: int
    move: int | None = None
