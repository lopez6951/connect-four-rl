"""
main_visual.py - Visual three-player Connect Four with pygame.

Demo mode:
    Player 1 = Human
    Player 2 = Q-learning AI, if trained; otherwise Greedy AI
    Player 3 = Greedy baseline AI

Run:
    python3 src/main_visual.py
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional, Tuple, Any

import pygame

sys.path.insert(0, os.path.dirname(__file__))

from env.board import Board, P1, P2, P3, ROWS, COLS, next_player
from self_play.player import GreedyPlayer, QLearningPlayer
from model.trainer import QLearningAgent

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "q_table_3p.pkl")

CELL = 82
RADIUS = CELL // 2 - 7
HEADER = CELL
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL + HEADER
FPS = 60

BG = (245, 243, 238)
BOARD_COLOR = (30, 80, 160)
EMPTY_COLOR = (220, 218, 212)
P1_COLOR = (216, 90, 48)   # human/orange
P2_COLOR = (24, 95, 165)   # q-agent/blue
P3_COLOR = (50, 150, 80)   # third player/green
TEXT_COLOR = (30, 30, 30)
WIN_BANNER = (255, 255, 255)


class DelayedAI:
    """Tiny wrapper to create a readable pause between AI turns."""
    def __init__(self, player: Any, delay: float = 0.25) -> None:
        self.player = player
        self.delay = delay

    def choose_move(self, board: Board, player: int) -> int:
        time.sleep(self.delay)
        return self.player.choose_move(board, player)


def load_ai() -> Any:
    if os.path.exists(MODEL_PATH):
        agent = QLearningAgent()
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0
        print(f"Loaded 3-player Q-learning AI from {MODEL_PATH}")
        return QLearningPlayer(agent)
    print("No trained 3-player Q-table found. Player 2 is using Greedy AI.")
    return GreedyPlayer()


def piece_color(player: int) -> tuple[int, int, int]:
    if player == P1:
        return P1_COLOR
    if player == P2:
        return P2_COLOR
    if player == P3:
        return P3_COLOR
    return EMPTY_COLOR


def draw_board(screen: pygame.Surface, board: Board) -> None:
    screen.fill(BG)
    pygame.draw.rect(screen, BOARD_COLOR, (0, HEADER, WIDTH, ROWS * CELL), border_radius=12)
    for r in range(ROWS):
        for c in range(COLS):
            cx = c * CELL + CELL // 2
            cy = HEADER + r * CELL + CELL // 2
            pygame.draw.circle(screen, piece_color(board.grid[r][c]), (cx, cy), RADIUS)


def draw_hover(screen: pygame.Surface, col: Optional[int], player: int) -> None:
    if col is None or col < 0 or col >= COLS:
        return
    cx = col * CELL + CELL // 2
    cy = HEADER // 2
    pygame.draw.circle(screen, piece_color(player), (cx, cy), RADIUS)
    pygame.draw.circle(screen, BG, (cx, cy), RADIUS - 6, 3)


def draw_status(screen: pygame.Surface, font: Any, message: str, color: Tuple[int, int, int] = TEXT_COLOR) -> None:
    label = font.render(message, True, color)
    rect = label.get_rect(center=(WIDTH // 2, HEADER // 2))
    screen.blit(label, rect)


def draw_banner(screen: pygame.Surface, font_big: Any, font_small: Any, message: str) -> None:
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    screen.blit(overlay, (0, 0))

    box_w, box_h = 500, 170
    box_x = (WIDTH - box_w) // 2
    box_y = (HEIGHT - box_h) // 2
    pygame.draw.rect(screen, WIN_BANNER, (box_x, box_y, box_w, box_h), border_radius=16)

    label = font_big.render(message, True, TEXT_COLOR)
    screen.blit(label, label.get_rect(center=(WIDTH // 2, box_y + 55)))

    sub = font_small.render("Click anywhere to play again", True, (120, 118, 112))
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, box_y + 115)))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Three-Player Connect Four RL Agent")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("helveticaneue", 34, bold=True)
    font_med = pygame.font.SysFont("helveticaneue", 21)
    font_small = pygame.font.SysFont("helveticaneue", 17)

    q_ai = DelayedAI(load_ai())
    p3_ai = DelayedAI(GreedyPlayer())
    ai_players = {P2: q_ai, P3: p3_ai}

    board = Board()
    current = P1
    game_over = False
    winner: Optional[int] = None
    hover_col: Optional[int] = None

    names = {P1: "You", P2: "Q-agent", P3: "Greedy player"}

    def reset() -> None:
        nonlocal board, current, game_over, winner, hover_col
        board = Board()
        current = P1
        game_over = False
        winner = None
        hover_col = None

    def handle_ai_turns() -> None:
        nonlocal current, game_over, winner
        while not game_over and current in (P2, P3):
            ai = ai_players[current]
            col = ai.choose_move(board.copy(), current)
            if col in board.legal_moves():
                board.drop(col, current)
            else:
                game_over = True
                winner = next_player(current)
                return

            if board.check_win(current):
                game_over = True
                winner = current
                return
            if board.is_draw():
                game_over = True
                winner = 0
                return
            current = next_player(current)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                mx, _ = event.pos
                hover_col = mx // CELL if not game_over and current == P1 else None

            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_over:
                    reset()
                    continue

                if current == P1:
                    mx, _ = event.pos
                    col = mx // CELL
                    if col in board.legal_moves():
                        board.drop(col, P1)
                        if board.check_win(P1):
                            game_over = True
                            winner = P1
                        elif board.is_draw():
                            game_over = True
                            winner = 0
                        else:
                            current = next_player(current)
                            handle_ai_turns()

        draw_board(screen, board)

        if game_over:
            if winner == 0:
                draw_banner(screen, font_big, font_small, "It's a draw!")
            else:
                draw_banner(screen, font_big, font_small, f"{names[winner]} wins!")
        else:
            if current == P1:
                draw_hover(screen, hover_col, P1)
                draw_status(screen, font_med, "Your turn: click a column")
            else:
                draw_status(screen, font_med, f"{names[current]} thinking...")

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
