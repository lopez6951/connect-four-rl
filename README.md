# Connect Four RL Agent

## Project Overview

This project builds a Connect Four game and reinforcement learning agent from scratch in Python. The goal is to train an AI agent that can learn stronger Connect Four moves by playing repeated games and improving its strategy based on rewards.

We chose Connect Four because it's simple enough to implement quickly but hard enough that an agent can't just stumble into good play. The agent must choose actions in a competitive environment where each move changes the future state of the game. Instead of manually programming every move, our reinforcement learning agent will learn from experience by receiving rewards for winning, losing, or drawing games.

## Why This Is AI

This project implements Deep Q-Network (DQN) reinforcement learning from scratch. The agent observes the current board state as a numerical array, passes it through a neural network to estimate Q-values for each possible action, and selects moves using an epsilon-greedy policy. Over thousands of training games, the network learns to approximate the value of each (state, action) pair and improves its play accordingly.

The core AI challenge is generalization. Plain tabular Q-learning can't handle this because there are too many unique board states to ever store or explore them all. We use a Deep Q-Network (DQN) instead, where a neural network approximates Q-values across the state space, allowing the agent to make informed decisions on board positions it has never seen before.


## Reinforcement Learning Setup

### State

A state represents the current Connect Four board. The board stores which cells are empty, which cells belong to Player 1, and which cells belong to Player 2. The board is represented as a 42-dimensional numerical array (flattened 6×7 grid), where each cell is encoded as 0 (empty), 1 (agent's piece), or -1 (opponent's piece). This array is fed directly into the neural network.

### Action

An action is a valid column where the agent can drop a piece. The agent will only choose from legal moves.

### Reward

The reward function will guide the agent’s learning:

* `+1` for winning a game
* `-1` for losing a game
* `0` for a draw
* `0` or a small intermediate reward for non-terminal moves

### Policy

The agent will use an epsilon-greedy policy:

* With probability epsilon, the agent explores by choosing a random valid move.
* With probability `1 - epsilon`, the agent exploits by choosing the move with the best known Q-value.

At the start of training, epsilon is high so the agent explores widely. As training progresses, epsilon decays so the agent increasingly exploits what it has learned.

### DQN Update

The agent updates its network weights using the Bellman equation as a training target:
```text
target = reward + gamma * max(Q_target(next_state)) * (1 - done)
loss = MSE(Q(state, action), target)
```

where:
* Q is the online network being trained
* Q_target is a periodically frozen copy of the network used to generate stable targets
* gamma is the discount factor
* done is 1 if the game ended, 0 otherwise
Experience tuples (state, action, reward, next_state, done) are stored in a replay buffer and sampled in random batches to break correlation between consecutive training steps.

## Planned Evaluation

We plan to evaluate the RL agent by comparing it against simple baseline players.

Possible comparisons:

* RL agent vs. random player
* RL agent vs. greedy player
* RL agent before training vs. RL agent after training
* Different training episode counts
* Different epsilon values
* Optional RL agent vs. minimax agent

Possible metrics:

* Win rate
* Loss rate
* Draw rate
* Average game length
* Training progress over episodes
* Performance improvement after training

## Project Structure

```text
connect-four-rl/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── main.py             # Entry point: train, evaluate, or play
│   ├── board.py            # Board state, legal moves, win/draw detection
│   ├── game.py             # Two-player game loop
│   ├── players.py          # Random and greedy baseline agents
│   │
│   └── ai/
│       ├── dqn.py          # DQN agent: network, replay buffer, training loop
│       ├── heuristic.py    # Greedy heuristic player for evaluation
│       └── minimax.py      # Optional minimax agent for benchmarking
│
├── tests/
│   ├── test_board.py       # Unit tests for board logic and win detection
│   ├── test_win_detection.py
│   └── test_ai_moves.py    # Tests for move legality and agent behavior
│
└── experiments/
    │   └── run_matches.py      # Evaluation tournaments and result logging
│
└── results/                    # Saved models and training output
    └── dqn_final.pt
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the main program:

```bash
python3 src/main.py
```

## Test

Run tests using:

```bash
pytest
```

## Team Members

* Lauren Lopez
* Karan Mor

## Current Plan

Our first goal is to build the Connect Four game environment, including the board representation, valid move logic, win detection, and baseline players. After that, we will implement the Q-learning agent from scratch and train it through repeated games. Once the RL agent is working, we will run experiments to measure how its performance changes after training.
