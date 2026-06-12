"""
eval/tournament.py — Three-player evaluation tournament.

Usage:
    python3 src/eval/tournament.py
"""

from __future__ import annotations
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.board import P1, P2, P3
from env.game import Game
from self_play.player import RandomPlayer, GreedyPlayer, QLearningPlayer
from model.trainer import QLearningAgent
from eval.elo import EloTracker

N_GAMES = 200
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "q_table_3p.pkl")


def run_match(p1: Any, p2: Any, p3: Any, names: list[str], n: int = N_GAMES, tracker: EloTracker | None = None) -> dict[str, int]:
    results = {"p1_wins": 0, "p2_wins": 0, "p3_wins": 0, "draws": 0}
    lengths = []

    for _ in range(n):
        g = Game(p1, p2, p3, verbose=False)
        winner, history = g.play()
        lengths.append(len(history))

        if winner == P1:
            results["p1_wins"] += 1
            if tracker:
                tracker.update_three_player_result(names, names[0])
        elif winner == P2:
            results["p2_wins"] += 1
            if tracker:
                tracker.update_three_player_result(names, names[1])
        elif winner == P3:
            results["p3_wins"] += 1
            if tracker:
                tracker.update_three_player_result(names, names[2])
        else:
            results["draws"] += 1
            if tracker:
                tracker.update_three_player_result(names, None)

    avg = sum(lengths) / len(lengths) if lengths else 0
    print(f"{names[0]} vs {names[1]} vs {names[2]}")
    print(f"  {results} | avg {avg:.1f} moves")
    return results


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
        agents["Q-learning"] = QLearningPlayer(agent)
    else:
        print("No trained 3-player Q-table found. Run: python3 src/main.py train\n")

    print(f"\nThree-player tournament — {N_GAMES} games per match\n")
    print("-" * 80)

    run_match(RandomPlayer(), GreedyPlayer(), GreedyPlayer(), ["Random", "Greedy-1", "Greedy-2"], tracker=tracker)

    if "Q-learning" in agents:
        run_match(agents["Q-learning"], RandomPlayer(), RandomPlayer(), ["Q-learning", "Random-1", "Random-2"], tracker=tracker)
        run_match(agents["Q-learning"], GreedyPlayer(), GreedyPlayer(), ["Q-learning", "Greedy-1", "Greedy-2"], tracker=tracker)
        run_match(agents["Q-learning"], RandomPlayer(), GreedyPlayer(), ["Q-learning", "Random", "Greedy"], tracker=tracker)

    print()
    print(tracker)


if __name__ == "__main__":
    main()
