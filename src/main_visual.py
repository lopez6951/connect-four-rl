"""
main_visual.py - Visual Connect Four using pygame.
Two human players, click a column to drop a piece.
 
Run:
    python src/main_visual.py
"""

import sys
import os
import pygame
from typing import List, Optional, Any, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from env.board import Board, P1, P2, ROWS, COLS


# Constants

CELL        = 100          # px per cell
RADIUS      = CELL // 2 - 8
HEADER      = CELL         # space above board for hover preview
WIDTH       = COLS * CELL
HEIGHT      = ROWS * CELL + HEADER

FPS         = 60

# Colors (R, G, B)
BG          = (245, 243, 238)
BOARD_COLOR = (30,  80, 160)
EMPTY_COLOR = (220, 218, 212)
P1_COLOR    = (216, 90,  48)   # orange-red
P2_COLOR    = (24,  95, 165)   # blue
TEXT_COLOR  = (30,  30,  30)
WIN_BANNER  = (255, 255, 255)

# Drawing helpers

def piece_color(player: int) -> tuple:
    if player == P1:
        return P1_COLOR
    if player == P2:
        return P2_COLOR
    return EMPTY_COLOR

def draw_board(screen: pygame.Surface, board: Board) -> None:
    screen.fill(BG)

    # Board rectangle
    pygame.draw.rect(
        screen, BOARD_COLOR,
        (0, HEADER, WIDTH, ROWS * CELL),
        border_radius=12
    )

    # Cells
    for r in range(ROWS):
        for c in range(COLS):
            cx = c * CELL + CELL // 2
            cy = HEADER + r * CELL + CELL // 2
            pygame.draw.circle(screen, piece_color(board.grid[r][c]), (cx, cy), RADIUS)

def draw_hover(screen: Any, col: Optional[int], player: int) -> None:
    if col is None:
        return
    cx = col * CELL + CELL // 2
    cy = HEADER // 2
    color = piece_color(player)
    pygame.draw.circle(screen, color, (cx, cy), RADIUS)
    pygame.draw.circle(screen, BG, (cx, cy), RADIUS - 6, 3)

def draw_status(screen: Any, font: Any, message: str, color: Tuple[int, int, int] = TEXT_COLOR) -> None:
    label = font.render(message, True, color)
    rect  = label.get_rect(center=(WIDTH // 2, HEADER // 2))
    screen.blit(label, rect)

def draw_banner(screen: Any, font_big: Any, font_small: Any, message: str) -> None:
    # Semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    screen.blit(overlay, (0, 0))

    # Banner box
    box_w, box_h = 420, 160
    box_x = (WIDTH - box_w) // 2
    box_y = (HEIGHT - box_h) // 2
    pygame.draw.rect(screen, WIN_BANNER, (box_x, box_y, box_w, box_h), border_radius=16)

    # Message
    label = font_big.render(message, True, TEXT_COLOR)
    screen.blit(label, label.get_rect(center=(WIDTH // 2, box_y + 55)))

    # Sub-prompt
    sub = font_small.render("Click anywhere to play again", True, (120, 118, 112))
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, box_y + 110)))

# Main game loop
def main() -> None:
    pygame.init()
    screen  = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four")
    clock   = pygame.time.Clock()

    font_big   = pygame.font.SysFont("helveticaneue", 38, bold=True)
    font_med   = pygame.font.SysFont("helveticaneue", 22)
    font_small = pygame.font.SysFont("helveticaneue", 17)

    def new_game() -> tuple[List[List[int]], int, bool, Optional[int]]:
        return Board(), P1, False, None

    board, current, game_over, winner = new_game()
    hover_col: Optional[int] = None

    player_names = {P1: "Player 1", P2: "Player 2"}
    player_colors = {1: P1_COLOR, 2: P2_COLOR}

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                mx, _ = event.pos
                hover_col = mx // CELL if not game_over else None

            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_over:
                    board, current, game_over, winner = new_game()
                    hover_col = None
                    continue

                mx, my = event.pos
                col = mx // CELL

                if col not in board.legal_moves():
                    continue

                board.drop(col, current)

                if board.check_win(current):
                    game_over = True
                    winner    = current
                elif board.is_draw():
                    game_over = True
                    winner    = 0
                else:
                    current = P2 if current == P1 else P1

        # Render
        draw_board(screen, board)

        if game_over:
            # winner has type Optional[int] in the signature; assert it's not None
            # so the type checker knows it's an int when we index player_names.
            assert winner is not None
            if winner == 0:
                draw_banner(screen, font_big, font_small, "It's a draw!")
            else:
                name = player_names[winner]
                draw_banner(screen, font_big, font_small, f"{name} wins!")
        else:
            draw_hover(screen, hover_col, current)
            name  = player_names[current]
            draw_status(screen, font_med, f"{name}'s turn")

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
