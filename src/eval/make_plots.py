"""
make_plots.py
Generate graphs for the three-player Connect Four RL final report.

Outputs:
    results/q_agent_move_heatmap_3p.png
    results/three_player_win_comparison.png

Run:
    python3 src/eval/make_plots.py
"""

from __future__ import annotations
import os
import sys
import random
from typing import Any

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.board import Board, P1, P2, P3, ROWS, COLS, next_player
from self_play.player import RandomPlayer, GreedyPlayer, QLearningPlayer
from model.trainer import QLearningAgent

MODEL_PATH = "results/q_table_3p.pkl"
OUT_DIR = "results"
N_GAMES = 200


def load_q_player() -> QLearningPlayer:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Run: python3 src/main.py train --episodes 5000"
        )
    agent = QLearningAgent()
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0
    return QLearningPlayer(agent)


def play_one_game_and_collect_positions(p1: Any, p2: Any, p3: Any, tracked_player: int = P1) -> tuple[int, list[tuple[int, int]]]:
    board = Board()
    players = {P1: p1, P2: p2, P3: p3}
    current = P1
    positions: list[tuple[int, int]] = []

    while True:
        col = players[current].choose_move(board.copy(), current)
        if col not in board.legal_moves():
            return next_player(current), positions

        row = board.drop(col, current)
        if current == tracked_player:
            positions.append((row, col))

        if board.check_win(current):
            return current, positions
        if board.is_draw():
            return 0, positions
        current = next_player(current)


def run_match(p1: Any, p2: Any, p3: Any, n_games: int = N_GAMES, collect_for_player: int | None = None) -> tuple[dict[str, int], list[list[int]]]:
    results = {"p1_wins": 0, "p2_wins": 0, "p3_wins": 0, "draws": 0}
    heatmap = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    for _ in range(n_games):
        if collect_for_player is None:
            winner, positions = play_one_game_and_collect_positions(p1, p2, p3, tracked_player=P1)
            positions = []
        else:
            winner, positions = play_one_game_and_collect_positions(p1, p2, p3, tracked_player=collect_for_player)

        if winner == P1:
            results["p1_wins"] += 1
        elif winner == P2:
            results["p2_wins"] += 1
        elif winner == P3:
            results["p3_wins"] += 1
        else:
            results["draws"] += 1

        for row, col in positions:
            heatmap[row][col] += 1

    return results, heatmap


def save_heatmap(heatmap: list[list[int]], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap)

    ax.set_title("3-Player Q-Learning Agent Move Placement Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(COLS))
    ax.set_yticklabels(range(ROWS))

    for r in range(ROWS):
        for c in range(COLS):
            ax.text(c, r, str(heatmap[r][c]), ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Q-agent pieces placed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_win_bar(match_results: dict[str, dict[str, int]], output_path: str) -> None:
    labels = list(match_results.keys())
    p1_rates, p2_rates, p3_rates, draw_rates = [], [], [], []

    for label in labels:
        result = match_results[label]
        total = sum(result.values())
        p1_rates.append(100 * result["p1_wins"] / total)
        p2_rates.append(100 * result["p2_wins"] / total)
        p3_rates.append(100 * result["p3_wins"] / total)
        draw_rates.append(100 * result["draws"] / total)

    x = list(range(len(labels)))
    width = 0.20
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar([i - 1.5 * width for i in x], p1_rates, width, label="Player 1 wins")
    ax.bar([i - 0.5 * width for i in x], p2_rates, width, label="Player 2 wins")
    ax.bar([i + 0.5 * width for i in x], p3_rates, width, label="Player 3 wins")
    ax.bar([i + 1.5 * width for i in x], draw_rates, width, label="Draws")

    ax.set_title("Three-Player Agent Performance Comparison")
    ax.set_ylabel("Percentage of games")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()

    bars = [p1_rates, p2_rates, p3_rates, draw_rates]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for series, offset in zip(bars, offsets):
        for j, value in enumerate(series):
            if value > 0:
                ax.text(j + offset, value + 1, f"{value:.0f}%", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    q_player = load_q_player()
    print("Generating three-player evaluation results...")

    random_vs_greedy, _ = run_match(RandomPlayer(), GreedyPlayer(), GreedyPlayer(), n_games=N_GAMES)
    q_vs_randoms, heatmap = run_match(q_player, RandomPlayer(), RandomPlayer(), n_games=N_GAMES, collect_for_player=P1)
    q_vs_greedies, _ = run_match(q_player, GreedyPlayer(), GreedyPlayer(), n_games=N_GAMES)
    q_vs_mixed, _ = run_match(q_player, RandomPlayer(), GreedyPlayer(), n_games=N_GAMES)

    match_results = {
        "Random-Greedy-Greedy": random_vs_greedy,
        "Q-Random-Random": q_vs_randoms,
        "Q-Greedy-Greedy": q_vs_greedies,
        "Q-Random-Greedy": q_vs_mixed,
    }

    print("\nResults:")
    for name, result in match_results.items():
        print(f"{name}: {result}")

    heatmap_path = os.path.join(OUT_DIR, "q_agent_move_heatmap_3p.png")
    bar_path = os.path.join(OUT_DIR, "three_player_win_comparison.png")

    save_heatmap(heatmap, heatmap_path)
    save_win_bar(match_results, bar_path)

    print("\nSaved graphs:")
    print(f"  {heatmap_path}")
    print(f"  {bar_path}")


if __name__ == "__main__":
    main()
