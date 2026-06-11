# Connect Four RL Agent

## Project Overview

This project builds a Connect Four game and reinforcement learning agent from scratch in Python. The goal is to train an AI agent that can learn stronger Connect Four moves by playing repeated games and improving its strategy based on rewards.

## Why This Is AI

This project uses reinforcement learning. The agent observes the current board state, chooses a legal column, receives a reward after the game outcome, and updates Q-values using the Bellman equation. The agent uses epsilon-greedy exploration during training and then uses the learned Q-table for evaluation/play.

## Reinforcement Learning Setup

- **State:** flattened 6x7 board from the current player's perspective
- **Action:** a legal column from 0 to 6
- **Reward:** +1 win, -1 loss, 0 draw/non-terminal move
- **Policy:** epsilon-greedy
- **Update:** tabular Q-learning

## Project Structure

```text
connect-four-rl/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── main_visual.py
│   ├── env/
│   │   ├── board.py
│   │   └── game.py
│   ├── self_play/
│   │   ├── player.py
│   │   └── loop.py
│   ├── model/
│   │   ├── trainer.py
│   │   └── network.py
│   ├── eval/
│   │   ├── elo.py
│   │   └── tournament.py
│   └── tree/
│       ├── node.py
│       └── search.py
├── tests/
└── results/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Show available commands:

```bash
python3 src/main.py
```

Train the Q-learning agent:

```bash
python3 src/main.py train --episodes 5000
```

Evaluate the agent:

```bash
python3 src/main.py eval --games 200
```

Play in terminal:

```bash
python3 src/main.py play
```

Play visually with pygame:

```bash
python3 src/main_visual.py
```

## Test

```bash
pytest
```

## Team Members

- Lauren Lopez
- Karan Mor
