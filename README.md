# Three-Player Connect Four RL Agent

## Project Overview

This project builds a larger three-player Connect Four environment and a reinforcement learning agent from scratch in Python. The game uses a larger 8x10 board to support three players while keeping the Connect Four objective: a player wins by connecting four pieces horizontally, vertically, or diagonally.

## Why This Is AI

The AI component is a tabular Q-learning agent. The agent observes the board state, chooses a legal column, receives rewards based on the final game outcome, and updates Q-values using the Bellman update rule. During training, the Q-learning agent plays as Player 1 against two baseline opponents. During evaluation and visual demo, it can compete against random and greedy players.

## Reinforcement Learning Setup

- **State:** flattened 8x10 board from the current player's perspective
- **Action:** a legal column from 0 to 8
- **Players:** Player 1, Player 2, Player 3
- **Reward:** +1 if the Q-learning agent wins, -1 if another player wins, 0 for a draw/non-terminal move
- **Policy:** epsilon-greedy exploration during training
- **Update:** tabular Q-learning from scratch

## Run

```bash
python3 src/main.py
python3 src/main.py train --episodes 5000
python3 src/main.py eval --games 200
python3 src/main.py play
python3 src/main_visual.py
python3 src/eval/make_plots.py
```

## Test

```bash
pytest
```

## Team Members

- Lauren Lopez
- Karan Mor
