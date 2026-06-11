"""
eval/tournament.py — Round-robin evaluation tournaments.

Usage:
    python src/eval/tournament.py
"""

from __future__ import annotations
import os
import sys
from typing import Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.board import P1, P2
from env.game import Game
from self_play.player import RandomPlayer, GreedyPlayer, QLearningPlayer
from model.trainer import QLearningAgent
from eval.elo import EloTracker

N_GAMES = 200
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "q_table.pkl")


def run_match(
    p1: Any,
    p2: Any,
    p1_name: str,
    p2_name: str,
    n: int = N_GAMES,
    tracker: Optional[EloTracker] = None,
) -> Tuple[int, int, int]:
    wins = losses = draws = 0
    lengths = []

    for _ in range(n):
        g = Game(p1, p2, verbose=False)
        winner, history = g.play()
        lengths.append(len(history))
        if winner == P1:
            wins += 1
        elif winner == P2:
            losses += 1
        else:
            draws += 1

    if tracker:
        tracker.update_match(p1_name, p2_name, wins, losses, draws)

    avg = sum(lengths) / len(lengths) if lengths else 0
    print(
        f"{p1_name:<16} vs {p2_name:<16} | "
        f"W {wins:3d} ({wins/n:4.0%})  "
        f"L {losses:3d} ({losses/n:4.0%})  "
        f"D {draws:3d} ({draws/n:4.0%})  "
        f"avg {avg:.1f} moves"
    )
    return wins, losses, draws


def main() -> None:
    tracker = EloTracker()

    agents: dict[str, Any] = {
        "Random": RandomPlayer(),
        "Greedy": GreedyPlayer(),
    }

    if os.path.exists(MODEL_PATH):
        agent = QLearningAgent()
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0
        agents["Q-Learning"] = QLearningPlayer(agent)
    else:
        print("No trained Q-table found. Run: python src/main.py train\n")

    names = list(agents.keys())
    print(f"\nTournament — {N_GAMES} games per match\n")
    print("-" * 75)

    for n1 in names:
        for n2 in names:
            if n1 != n2:
                run_match(agents[n1], agents[n2], n1, n2, tracker=tracker)

    print()
    print(tracker)


if __name__ == "__main__":
    main()
