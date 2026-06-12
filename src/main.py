"""
src/main.py
Command-line entry point for three-player Connect Four RL.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from env.board import Board, P1, P2, P3
from env.game import Game
from self_play.player import HumanPlayer, RandomPlayer, GreedyPlayer, QLearningPlayer
from self_play.loop import play_many
from model.trainer import QLearningAgent, train_q_learning

MODEL_PATH = "results/q_table_3p.pkl"


def load_q_player() -> QLearningPlayer:
    agent = QLearningAgent()
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0
        print(f"Loaded trained 3-player Q-learning agent from {MODEL_PATH}")
    else:
        print("No trained 3-player Q-table found. Using untrained Q-agent.")
    return QLearningPlayer(agent)


def cmd_train(args: argparse.Namespace) -> None:
    train_q_learning(
        episodes=args.episodes,
        opponent_name=args.opponent,
        save_path=MODEL_PATH,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    q_player = load_q_player()
    print("\nEvaluating three-player agents...\n")

    comparisons: list[tuple[str, Any, Any, Any]] = [
        ("Random vs Greedy vs Greedy", RandomPlayer(), GreedyPlayer(), GreedyPlayer()),
        ("Q-learning vs Random vs Random", q_player, RandomPlayer(), RandomPlayer()),
        ("Q-learning vs Greedy vs Greedy", q_player, GreedyPlayer(), GreedyPlayer()),
        ("Q-learning vs Random vs Greedy", q_player, RandomPlayer(), GreedyPlayer()),
    ]

    for name, p1, p2, p3 in comparisons:
        results = play_many(p1, p2, p3, n_games=args.games)
        print(name)
        print(results)
        print()


def cmd_play(args: argparse.Namespace) -> None:
    q_player = load_q_player()
    print("Terminal mode: You are Player 1, Q-agent is Player 2, Greedy is Player 3.\n")
    winner, history = Game(HumanPlayer(), q_player, GreedyPlayer(), verbose=True).play()
    print("Final result:")
    if winner == 0:
        print("Draw")
    else:
        print(f"Player {winner} wins")
    print(f"Moves played: {len(history)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-player Connect Four RL")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train Q-learning agent")
    p_train.add_argument("--episodes", type=int, default=5000)
    p_train.add_argument("--opponent", choices=["random", "greedy"], default="random")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate agents")
    p_eval.add_argument("--games", type=int, default=200)
    p_eval.set_defaults(func=cmd_eval)

    p_play = sub.add_parser("play", help="Play terminal game")
    p_play.set_defaults(func=cmd_play)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        print("Three-Player Connect Four RL Agent")
        print("Board: 8 rows x 10 columns")
        print("Commands:")
        print("  python3 src/main.py train --episodes 5000")
        print("  python3 src/main.py train --episodes 5000 --opponent greedy")
        print("  python3 src/main.py eval --games 200")
        print("  python3 src/main.py play")
        print("  python3 src/main_visual.py")
        return

    args.func(args)


if __name__ == "__main__":
    main()
