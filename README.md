# Connect Four RL Agent

## Project Overview

This project builds a Connect Four game and reinforcement learning agent from scratch in Python. The goal is to train an AI agent that can learn stronger Connect Four moves by playing repeated games and improving its strategy based on rewards.

Connect Four is a good AI problem because the agent must choose actions in a competitive environment where each move changes the future state of the game. Instead of manually programming every move, our reinforcement learning agent will learn from experience by receiving rewards for winning, losing, or drawing games.

## Why This Is AI

This project focuses on reinforcement learning, where an agent learns a policy through trial and error. The agent observes the current board state, selects a valid column as its action, receives a reward based on the game result, and updates its strategy over many games.

The main AI challenge is that Connect Four has many possible board states, so the agent must learn useful patterns from experience rather than memorize every possible game. We will start with a simple Q-learning approach and may use a smaller board or feature-based state representation if the full state space becomes too large.

## AI Methods

The main AI techniques we plan to implement are:

* Q-learning from scratch
* Epsilon-greedy exploration
* Reward-based policy updates
* Training through repeated games
* Baseline agents such as random and greedy players
* Optional self-play training
* Optional comparison against a minimax search agent if time allows

## Reinforcement Learning Setup

### State

A state represents the current Connect Four board. The board stores which cells are empty, which cells belong to Player 1, and which cells belong to Player 2.

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

### Q-Learning Update

The agent will update its Q-values using the standard Q-learning rule:

```text
Q(state, action) = Q(state, action) + alpha * [reward + gamma * max(Q(next_state, next_action)) - Q(state, action)]
```

where:

* `alpha` is the learning rate
* `gamma` is the discount factor
* `reward` is the result of the move or game
* `max(Q(next_state, next_action))` estimates the best future value from the next state

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
│   ├── main.py
│   ├── board.py
│   ├── game.py
│   ├── players.py
│   │
│   └── ai/
│       ├── q_learning.py
│       ├── heuristic.py
│       └── minimax.py
│
├── tests/
│   ├── test_board.py
│   ├── test_win_detection.py
│   └── test_ai_moves.py
│
└── experiments/
    └── run_matches.py
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
