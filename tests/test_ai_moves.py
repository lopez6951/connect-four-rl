import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.board import Board, P1, P2
from self_play.player import RandomPlayer, GreedyPlayer
from model.trainer import QLearningAgent


def test_random_player_returns_legal_move():
    b = Board()
    move = RandomPlayer().choose_move(b, P1)
    assert move in b.legal_moves()


def test_greedy_player_takes_winning_move():
    b = Board()
    for c in [0, 1, 2]:
        b.drop(c, P1)
    move = GreedyPlayer().choose_move(b, P1)
    assert move == 3


def test_q_learning_agent_returns_legal_move():
    b = Board()
    agent = QLearningAgent(epsilon=0.0)
    move = agent.choose_action(b, P1, training=False)
    assert move in b.legal_moves()
