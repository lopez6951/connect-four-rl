# Connect Four RL Agent

## Project Overview

This project builds a Connect Four environment and a tabular Q-learning agent from scratch in Python. The project supports two versions: a standard two-player 6×7 game and an extended three-player 8×10 game. The Q-learning agent learns by playing repeated games and updating its strategy based on wins, losses, and draws. No pre-built RL libraries are used.

## Why This Is AI

The AI component is a tabular Q-learning agent. The agent observes the board state, chooses a legal column, receives rewards based on the final game outcome, and updates Q-values using the Bellman update rule. During training, the Q-learning agent plays as Player 1 against two baseline opponents. During evaluation and visual demo, it can compete against random and greedy players.

## Reinforcement Learning Setup

- **State:** flattened board encoded from the current player's perspective (own pieces = 1, opponents = -1, empty = 0)
- **Action:** a legal column (0–6 for two-player, 0–9 for three-player)
- **Players:** 2-player (P1 orange, P2 blue) or 3-player (P1 orange, P2 blue, P3 green)
- **Reward:** +1 for win, -1 for loss, 0 for draw or non-terminal move
- **Policy:** epsilon-greedy exploration during training, pure exploitation at eval
- **Update:** tabular Q-learning with Bellman equation from scratch

## Setup
```bash
git clone https://github.com/lopez6951/connect-four-rl
cd connect-four-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Results

### Two-Player (6×7 board, branch: two-player)
```bash
git checkout main

# Train the Q-learning agent
python3 src/main.py train --episodes 50000

# Evaluate against baselines
python3 src/main.py eval --games 200

# Run full Elo tournament
python3 src/eval/tournament.py

# Play against the trained agent in terminal
python3 src/main.py play

# Play against the trained agent visually
python3 src/main_visual.py
```

### Three-Player (8×10 board, branch: three-player)
```bash
git checkout three-player

# Train the Q-learning agent
python3 src/main.py train --episodes 50000

# Evaluate against baselines
python3 src/main.py eval --games 200

# Run full three-player Elo tournament
python3 src/eval/tournament.py

# Generate result plots
python3 src/eval/make_plots.py

# Play against the trained agent visually
python3 src/main_visual.py
```

## Test
```bash
pytest
```

## Code Organization
connect-four-rl/
│
├── src/
│   ├── env/
│   │   ├── board.py          # Board state, legal moves, win/draw detection
│   │   └── game.py           # Two/three-player game loop
│   │
│   ├── eval/
│   │   ├── elo.py            # Elo rating tracker for agents
│   │   ├── tournament.py     # Round-robin evaluation tournament
│   │   └── make_plots.py     # Heatmap and win comparison plots (three-player)
│   │
│   ├── model/
│   │   ├── network.py        # Q-table data structure
│   │   └── trainer.py        # Q-learning agent -> epsilon-greedy, Bellman update
│   │
│   ├── self_play/
│   │   ├── player.py         # RandomPlayer, GreedyPlayer, HumanPlayer, QLearningPlayer
│   │   └── loop.py           # play_many() game runner
│   │
│   ├── tree/
│   │   ├── node.py           # Game tree node container
│   │   └── search.py         # Heuristic evaluation and greedy search
│   │
│   ├── main.py               # CLI entry point (train / eval / play)
│   └── main_visual.py        # Pygame visual interface
│
├── tests/
│   ├── test_board.py         # Board logic unit tests
│   ├── test_win_detection.py # Win detection unit tests
│   └── test_ai_moves.py      # Agent move legality tests
│
├── results/                  # Saved Q-tables and plots
└── requirements.txt

## Team Members

- Lauren Lopez
- Karan Mor
