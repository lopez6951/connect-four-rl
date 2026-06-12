"""
model/trainer.py
Tabular Q-learning agent for Connect Four.

This is reinforcement learning from scratch:
- state = board encoded from current player's perspective
- action = legal column
- update = Q-learning Bellman update
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from env.board import Board, COLS, P1, P2
from self_play.player import RandomPlayer, GreedyPlayer

State = Tuple[int, ...]


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q: Dict[State, List[float]] = defaultdict(lambda: [0.0] * COLS)

    def get_state(self, board: Board, player: int) -> State:
        return board.encode_for_player(player)

    def choose_action(self, board: Board, player: int, training: bool = True) -> int:
        legal = board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        state = self.get_state(board, player)

        if training and random.random() < self.epsilon:
            return random.choice(legal)

        values = self.q[state]
        best_value = max(values[col] for col in legal)
        best_actions = [col for col in legal if values[col] == best_value]
        return random.choice(best_actions)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State | None,
        next_legal: List[int],
        done: bool,
    ) -> None:
        old_q = self.q[state][action]
        if done or next_state is None or not next_legal:
            target = reward
        else:
            target = reward + self.gamma * max(self.q[next_state][a] for a in next_legal)
        self.q[state][action] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q": dict(self.q),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.q = defaultdict(lambda: [0.0] * COLS, payload["q"])
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)


def train_q_learning(
    episodes: int = 5000,
    opponent_name: str = "random",
    save_path: str = "results/q_table.pkl",
) -> QLearningAgent:
    """
    Train Q-learning as Player 1 against a baseline Player 2.

    One training transition is measured from the agent's move to the next
    time the agent gets to move, after the opponent has moved.
    """
    agent = QLearningAgent()
    opponent = GreedyPlayer() if opponent_name == "greedy" else RandomPlayer()

    wins = losses = draws = 0

    for ep in range(1, episodes + 1):
        board = Board()
        done = False

        while not done:
            # Agent turn as P1.
            state = agent.get_state(board, P1)
            action = agent.choose_action(board, P1, training=True)
            board.drop(action, P1)

            if board.check_win(P1):
                agent.update(state, action, 1.0, None, [], True)
                wins += 1
                done = True
                break

            if board.is_draw():
                agent.update(state, action, 0.0, None, [], True)
                draws += 1
                done = True
                break

            # Opponent turn.
            opp_action = opponent.choose_action(board.copy(), P2)
            board.drop(opp_action, P2)

            if board.check_win(P2):
                agent.update(state, action, -1.0, None, [], True)
                losses += 1
                done = True
                break

            if board.is_draw():
                agent.update(state, action, 0.0, None, [], True)
                draws += 1
                done = True
                break

            # Non-terminal: update toward next state value.
            next_state = agent.get_state(board, P1)
            agent.update(state, action, 0.0, next_state, board.legal_moves(), False)

        agent.decay_epsilon()

        if ep % max(1, episodes // 10) == 0:
            total = wins + losses + draws
            win_rate = wins / total if total else 0
            print(
                f"Episode {ep:5d}/{episodes} | "
                f"W {wins:4d} L {losses:4d} D {draws:4d} | "
                f"win rate {win_rate:5.1%} | epsilon {agent.epsilon:.3f}"
            )

    agent.save(save_path)
    print(f"\nSaved trained Q-table to: {save_path}")
    print(f"Learned states: {len(agent.q)}")
    return agent
