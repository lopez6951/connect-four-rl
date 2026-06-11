"""
main.py - Command line entry point.

Examples:
    python src/main.py
    python src/main.py train --episodes 5000
    python src/main.py eval --games 200
    python src/main.py play
"""

from __future__ import annotations
import argparse
import os

from env.board import P1, P2
from env.game import Game
from self_play.player import HumanPlayer, RandomPlayer, GreedyPlayer, QLearningPlayer
from model.trainer import QLearningAgent, train_q_learning
from self_play.loop import play_many

MODEL_PATH = "results/q_table.pkl"


def cmd_train(args: argparse.Namespace) -> None:
    train_q_learning(episodes=args.episodes, opponent_name=args.opponent, save_path=MODEL_PATH)


def cmd_eval(args: argparse.Namespace) -> None:
    print("Evaluating agents...")

    random_player = RandomPlayer()
    greedy_player = GreedyPlayer()

    if os.path.exists(MODEL_PATH):
        agent = QLearningAgent()
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0
        q_player = QLearningPlayer(agent)
        print(f"Loaded trained Q-learning agent from {MODEL_PATH}")
    else:
        print("No trained model found yet. Run: python src/main.py train")
        q_player = None

    print("\nRandom vs Greedy")
    print(play_many(random_player, greedy_player, n_games=args.games))

    if q_player is not None:
        print("\nQ-Learning vs Random")
        print(play_many(q_player, random_player, n_games=args.games))
        print("\nQ-Learning vs Greedy")
        print(play_many(q_player, greedy_player, n_games=args.games))


def cmd_play(args: argparse.Namespace) -> None:
    print("Human is Player 1 (X). AI is Player 2 (O).")

    if os.path.exists(MODEL_PATH):
        agent = QLearningAgent()
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0
        ai = QLearningPlayer(agent)
        print(f"Loaded Q-learning AI from {MODEL_PATH}")
    else:
        ai = GreedyPlayer()
        print("No trained Q-table found. Using Greedy AI instead.")

    game = Game(HumanPlayer(), ai, verbose=True)
    winner, _ = game.play()
    if winner == P1:
        print("You won!")
    elif winner == P2:
        print("AI won!")
    else:
        print("Draw!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Connect Four RL Agent")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train Q-learning agent")
    train_p.add_argument("--episodes", type=int, default=5000)
    train_p.add_argument("--opponent", choices=["random", "greedy"], default="random")
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser("eval", help="Evaluate agents")
    eval_p.add_argument("--games", type=int, default=200)
    eval_p.set_defaults(func=cmd_eval)

    play_p = sub.add_parser("play", help="Play against the trained agent in terminal")
    play_p.set_defaults(func=cmd_play)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        print("Connect Four RL Agent")
        print("Project environment is set up successfully.")
        print("Commands:")
        print("  python src/main.py train --episodes 5000")
        print("  python src/main.py eval --games 200")
        print("  python src/main.py play")
        print("  python src/main_visual.py")
        return

    args.func(args)


if __name__ == "__main__":
    main()
