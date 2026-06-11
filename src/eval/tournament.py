"""
tournament.py — evaluation tournaments

Usage:
    python src/eval/tournament.py
"""

import os
from typing import Any, Optional, Tuple

from ..env.board import P1, P2
from ..env.game import Game
from ..self_play.player import RandomPlayer, GreedyPlayer
from .elo import EloTracker

# 
N_GAMES    = 200
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "q_table.pkl")


def run_match(p1: Any, p2: Any, p1_name: str, p2_name: str, n: int = N_GAMES,
              tracker: Optional[EloTracker] = None) -> Tuple[int, int, int]:
    '''Play n games between p1 and p2 and return/update from p1 perspective'''
    wins = losses = draws = 0
    lengths = []

    for _ in range(n):
        g = Game(p1, p2, verbose=False)
        winner, history = g.play()
        lengths.append(len(history))

        # Count result from p1 view
        if winner == P1:
            wins += 1
        elif winner == P2:
            losses += 1
        else:
            draws += 1

    # Update Elo ratings
    if tracker:
        tracker.update_match(p1_name, p2_name, wins, losses, draws)

    # one sum line/per
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
    '''main'''
    tracker = EloTracker()
    
    # Baseline
    agents = {
        "Random": RandomPlayer(),
        "Greedy": GreedyPlayer(),
    }

    # Load Q-learning agent if exists
    if os.path.exists(MODEL_PATH):
        try:
            from model.trainer import QLearningAgent
            from self_play.player import QLearningPlayer
            agent = QLearningAgent()
            agent.load(MODEL_PATH)
            agent.epsilon = 0.0
            agents["Q-Learning"] = QLearningPlayer(agent)
        except Exception as e:
            print(f"Could not load Q-learning agent: {e}")
    else:
        print("No trained Q-table found. Run: python src/main.py train\n")

    names = list(agents.keys())
    print(f"\nTournament — {N_GAMES} games per match\n")
    print("-" * 75)

    # every agent plays every other agent
    for n1 in names:
        for n2 in names:
            if n1 != n2:
                run_match(agents[n1], agents[n2], n1, n2, tracker=tracker)

    # leaderboard
    print()
    print(tracker)

if __name__ == "__main__":
    main()
