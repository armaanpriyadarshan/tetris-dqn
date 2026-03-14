#!/usr/bin/env python3
"""
Watch the DQN agent play Tetris with full visuals.

Usage:
    python play_visual.py                  # play with random init
    python play_visual.py best_model.pt    # play with trained model

Controls:
    Space     Pause / Resume
    Up/Down   Speed up / slow down
    R         Restart game
    Q / Esc   Quit
"""

import pygame
import sys
import time
import numpy as np
import torch
from pathlib import Path

from tetris_env import (
    TetrisEnv, PIECE_NAMES, PIECES, VISIBLE_ROWS, COLS,
    HIDDEN_ROWS, TOTAL_ROWS, action_to_tuple, NUM_PIECES,
)
from dqn_agent import DQNAgent


# ── Layout ────────────────────────────────────────────────────────────
BLOCK = 30
PREVIEW_BLOCK = 20
BOARD_X = 180
BOARD_Y = 80
BOARD_W = COLS * BLOCK           # 300
BOARD_H = VISIBLE_ROWS * BLOCK   # 600
WINDOW_W = BOARD_X + BOARD_W + 170
WINDOW_H = BOARD_Y + BOARD_H + 40

# ── Colors ────────────────────────────────────────────────────────────
BG = (15, 15, 26)
EMPTY_CELL = (32, 32, 44)
GRID_LINE = (44, 44, 58)
BORDER = (80, 80, 105)
TEXT = (210, 210, 225)
DIM_TEXT = (115, 115, 140)
LABEL_COLOR = (90, 90, 115)

# Piece colors: index = piece_idx + 1 (0 = empty)
# I=1, O=2, T=3, S=4, Z=5, J=6, L=7
PIECE_COLORS = [
    EMPTY_CELL,
    (0, 220, 220),     # I - cyan
    (220, 220, 0),     # O - yellow
    (160, 0, 220),     # T - purple
    (0, 220, 0),       # S - green
    (220, 0, 0),       # Z - red
    (30, 30, 220),     # J - blue
    (220, 140, 0),     # L - orange
]

# ── Timing ────────────────────────────────────────────────────────────
GHOST_MS = 120       # show ghost before locking
LOCK_MS = 40         # flash after lock
CLEAR_MS = 300       # line clear animation
GAP_MS = 60          # pause between pieces


def lighter(color, amount=50):
    return tuple(min(255, c + amount) for c in color)


def darker(color, amount=60):
    return tuple(max(0, c - amount) for c in color)


def blend(color, bg, alpha):
    """Blend color with bg using alpha (0-255)."""
    a = alpha / 255
    return tuple(int(c * a + b * (1 - a)) for c, b in zip(color, bg))


class TetrisVisualizer:

    def __init__(self, model_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Tetris DQN")

        self.font_title = pygame.font.SysFont("consolas", 34, bold=True)
        self.font_sub = pygame.font.SysFont("consolas", 16)
        self.font_label = pygame.font.SysFont("consolas", 18)
        self.font_value = pygame.font.SysFont("consolas", 26, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 15)
        self.font_big = pygame.font.SysFont("consolas", 40, bold=True)

        self.env = TetrisEnv()
        self.agent = DQNAgent()

        # Load model if available
        if model_path and Path(model_path).exists():
            self.agent.q_net.load_state_dict(
                torch.load(model_path, map_location=self.agent.device,
                           weights_only=True))
            self.agent.q_net.eval()
            print(f"Loaded model: {model_path}")

        # Force greedy play (no exploration)
        self.agent.epsilon_start = 0.0
        self.agent.epsilon_end = 0.0
        self.agent.steps_done = self.agent.epsilon_decay  # ensure epsilon=0

        self.clock = pygame.time.Clock()
        self.speed = 1.0
        self.paused = False
        self.running = True

    # ── Drawing primitives ────────────────────────────────────────────

    def draw_block(self, x, y, color_idx, size=BLOCK, alpha=255):
        """Draw a single beveled 3D block."""
        if color_idx == 0:
            rect = pygame.Rect(x, y, size - 1, size - 1)
            pygame.draw.rect(self.screen, EMPTY_CELL, rect)
            pygame.draw.rect(self.screen, GRID_LINE, rect, 1)
            return

        base = PIECE_COLORS[color_idx]
        if alpha < 255:
            base = blend(base, EMPTY_CELL, alpha)

        s = size - 1
        rect = pygame.Rect(x, y, s, s)
        pygame.draw.rect(self.screen, base, rect)

        # Highlight edges (top + left)
        hi = lighter(base, 45)
        pygame.draw.line(self.screen, hi, (x, y), (x + s - 1, y), 2)
        pygame.draw.line(self.screen, hi, (x, y), (x, y + s - 1), 2)

        # Shadow edges (bottom + right)
        sh = darker(base, 55)
        pygame.draw.line(self.screen, sh, (x + 1, y + s - 1),
                         (x + s - 1, y + s - 1), 2)
        pygame.draw.line(self.screen, sh, (x + s - 1, y + 1),
                         (x + s - 1, y + s - 1), 2)

        # Inner shine
        inner = lighter(base, 20)
        inner_rect = pygame.Rect(x + 3, y + 3, s - 6, s - 6)
        if inner_rect.width > 0 and inner_rect.height > 0:
            pygame.draw.rect(self.screen, inner, inner_rect)

    def draw_board(self, board=None, ghost_cells=None, ghost_color=0,
                   flash_rows=None):
        """Draw the game board."""
        if board is None:
            board = self.env.board

        # Border
        pygame.draw.rect(self.screen, BORDER,
                         pygame.Rect(BOARD_X - 4, BOARD_Y - 4,
                                     BOARD_W + 8, BOARD_H + 8), 3,
                         border_radius=2)

        for r in range(VISIBLE_ROWS):
            br = r + HIDDEN_ROWS  # board row
            for c in range(COLS):
                x = BOARD_X + c * BLOCK
                y = BOARD_Y + r * BLOCK

                # Line clear flash
                if flash_rows and br in flash_rows:
                    pygame.draw.rect(self.screen, (255, 255, 255),
                                     pygame.Rect(x, y, BLOCK - 1, BLOCK - 1))
                    continue

                cell = int(board[br, c])

                # Ghost piece
                if ghost_cells and (br, c) in ghost_cells and cell == 0:
                    self.draw_block(x, y, ghost_color, alpha=90)
                elif cell > 0:
                    self.draw_block(x, y, cell)
                else:
                    self.draw_block(x, y, 0)

    def draw_piece_preview(self, piece_idx, x, y, label):
        """Draw a piece preview box with label."""
        text = self.font_label.render(label, True, LABEL_COLOR)
        self.screen.blit(text, (x, y - 22))

        box_w = PREVIEW_BLOCK * 5 + 10
        box_h = PREVIEW_BLOCK * 4 + 6
        box = pygame.Rect(x - 2, y, box_w, box_h)
        pygame.draw.rect(self.screen, (25, 25, 38), box, border_radius=4)
        pygame.draw.rect(self.screen, (55, 55, 72), box, 2, border_radius=4)

        if piece_idx < 0:
            return

        name = PIECE_NAMES[piece_idx]
        cells = PIECES[name][0]
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        pw = (max_c - min_c + 1) * PREVIEW_BLOCK
        ph = (max_r - min_r + 1) * PREVIEW_BLOCK
        ox = x + (box_w - pw) // 2
        oy = y + (box_h - ph) // 2

        for dr, dc in cells:
            bx = ox + (dc - min_c) * PREVIEW_BLOCK
            by = oy + (dr - min_r) * PREVIEW_BLOCK
            self.draw_block(bx, by, piece_idx + 1, size=PREVIEW_BLOCK)

    def draw_stat(self, label, value, x, y):
        lbl = self.font_label.render(label, True, LABEL_COLOR)
        val = self.font_value.render(str(value), True, TEXT)
        self.screen.blit(lbl, (x, y))
        self.screen.blit(val, (x, y + 22))

    def draw_ui(self):
        """Draw all UI panels."""
        # Title
        title = self.font_title.render("T E T R I S", True, TEXT)
        tx = BOARD_X + (BOARD_W - title.get_width()) // 2
        self.screen.blit(title, (tx, 12))

        sub = self.font_sub.render("DQN Agent", True, DIM_TEXT)
        sx = BOARD_X + (BOARD_W - sub.get_width()) // 2
        self.screen.blit(sub, (sx, 50))

        # ── Left panel ────────────────────────────────────────────────
        lx = 16

        # Hold piece
        self.draw_piece_preview(self.env.hold_piece, lx, BOARD_Y + 8, "HOLD")

        # Stats
        sy = BOARD_Y + 120
        self.draw_stat("SCORE", f"{self.env.score:,}", lx, sy)
        self.draw_stat("LINES", self.env.total_lines, lx, sy + 65)
        level = self.env.total_lines // 10 + 1
        self.draw_stat("LEVEL", level, lx, sy + 130)
        self.draw_stat("PIECES", self.env.pieces_placed, lx, sy + 195)

        if self.env.combo > 0:
            self.draw_stat("COMBO", self.env.combo, lx, sy + 260)
        if self.env.back_to_back:
            b2b = self.font_small.render("B2B", True, (255, 200, 50))
            self.screen.blit(b2b, (lx, sy + 325))

        # ── Right panel ───────────────────────────────────────────────
        rx = BOARD_X + BOARD_W + 22

        # Next piece
        self.draw_piece_preview(
            self.env.next_pieces[0], rx, BOARD_Y + 8, "NEXT")

        # Current piece label
        piece_name = PIECE_NAMES[self.env.current_piece]
        cur = self.font_label.render(f"CURRENT: {piece_name}", True, DIM_TEXT)
        self.screen.blit(cur, (rx, BOARD_Y + 115))

        # Speed
        spd = self.font_small.render(
            f"Speed: {self.speed:.1f}x", True, DIM_TEXT)
        self.screen.blit(spd, (rx, BOARD_Y + 160))

        # Controls
        cy = BOARD_Y + 200
        controls = [
            "[Space] Pause",
            "[Up/Dn] Speed",
            "[R]     Restart",
            "[Q]     Quit",
        ]
        for i, line in enumerate(controls):
            t = self.font_small.render(line, True, (70, 70, 90))
            self.screen.blit(t, (rx, cy + i * 20))

        # Board features
        if hasattr(self.env, '_current_features'):
            fy = BOARD_Y + 310
            feats = self.env._current_features
            feat_lines = [
                f"Holes:  {feats['total_holes']:.0f}",
                f"Height: {feats['max_height']:.0f}",
                f"Bump:   {feats['bumpiness']:.0f}",
            ]
            for i, line in enumerate(feat_lines):
                t = self.font_small.render(line, True, DIM_TEXT)
                self.screen.blit(t, (rx, fy + i * 18))

        # Pause overlay
        if self.paused:
            overlay = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            self.screen.blit(overlay, (BOARD_X, BOARD_Y))
            pt = self.font_big.render("PAUSED", True, (255, 255, 120))
            px = BOARD_X + (BOARD_W - pt.get_width()) // 2
            py = BOARD_Y + BOARD_H // 2 - 25
            self.screen.blit(pt, (px, py))

    def render_frame(self, board=None, ghost_cells=None, ghost_color=0,
                     flash_rows=None):
        """Full frame render."""
        self.screen.fill(BG)
        self.draw_board(board, ghost_cells, ghost_color, flash_rows)
        self.draw_ui()
        pygame.display.flip()

    # ── Event handling ────────────────────────────────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed = min(20.0, self.speed + 0.5)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(0.5, self.speed - 0.5)
                elif event.key == pygame.K_r:
                    return "restart"
        return None

    def wait_ms(self, ms):
        """Wait while processing events. Returns 'restart' or None."""
        end = time.time() + (ms / 1000.0) / self.speed
        while time.time() < end:
            result = self.handle_events()
            if not self.running:
                return "quit"
            if result == "restart":
                return "restart"
            while self.paused and self.running:
                self.render_frame()
                result = self.handle_events()
                if result == "restart":
                    return "restart"
                self.clock.tick(30)
            self.clock.tick(120)
        return None

    # ── Game loop ─────────────────────────────────────────────────────

    def run(self):
        while self.running:
            state = self.env.reset()

            while not self.env.done and self.running:
                result = self.handle_events()
                if not self.running or result == "restart":
                    break
                if self.paused:
                    self.render_frame()
                    self.clock.tick(30)
                    continue

                # ── Agent picks action ────────────────────────────────
                action_idx = self.agent.select_action(state)
                hold, rot, col = action_to_tuple(action_idx)

                # Figure out which piece will be placed (after hold swap)
                piece_idx = self.env.current_piece
                if hold:
                    if self.env.hold_piece < 0:
                        piece_idx = self.env.next_pieces[0]
                    else:
                        piece_idx = self.env.hold_piece

                # Compute ghost (landing position)
                drop_row = self.env._drop_row(piece_idx, rot, col)
                ghost = set(self.env._get_cells(piece_idx, rot, drop_row, col))

                # ── Show ghost piece ──────────────────────────────────
                self.render_frame(ghost_cells=ghost,
                                  ghost_color=piece_idx + 1)
                r = self.wait_ms(GHOST_MS)
                if r:
                    break

                # ── Execute step ──────────────────────────────────────
                # Build temp board showing piece locked (for animation)
                temp_board = self.env.board.copy()
                for br, bc in ghost:
                    if 0 <= br < TOTAL_ROWS and 0 <= bc < COLS:
                        temp_board[br, bc] = float(piece_idx + 1)

                # Find rows that will be cleared
                full_rows = set()
                for row in range(TOTAL_ROWS):
                    if temp_board[row].all():
                        full_rows.add(row)

                # Show locked piece
                self.render_frame(board=temp_board)
                r = self.wait_ms(LOCK_MS)
                if r:
                    break

                # ── Line clear animation ──────────────────────────────
                if full_rows:
                    # Flash cleared rows 3 times
                    for flash in range(3):
                        self.render_frame(board=temp_board,
                                          flash_rows=full_rows)
                        r = self.wait_ms(CLEAR_MS // 6)
                        if r:
                            break
                        self.render_frame(board=temp_board)
                        r = self.wait_ms(CLEAR_MS // 6)
                        if r:
                            break
                    if r:
                        break

                # Actually execute the step
                state, reward, done, info = self.env.step(action_idx)

                # Show result
                self.render_frame()
                r = self.wait_ms(GAP_MS)
                if r:
                    break

            if not self.running:
                break

            # ── Game over screen ──────────────────────────────────────
            if self.env.done:
                self.render_frame()

                # Dark overlay
                overlay = pygame.Surface(
                    (BOARD_W + 8, BOARD_H + 8), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                self.screen.blit(overlay, (BOARD_X - 4, BOARD_Y - 4))

                # Game over text
                go = self.font_big.render("GAME OVER", True, (255, 70, 70))
                gx = BOARD_X + (BOARD_W - go.get_width()) // 2
                gy = BOARD_Y + BOARD_H // 2 - 60
                self.screen.blit(go, (gx, gy))

                # Final stats
                stats = [
                    f"Score: {self.env.score:,}",
                    f"Lines: {self.env.total_lines}",
                    f"Pieces: {self.env.pieces_placed}",
                    f"Level: {self.env.total_lines // 10 + 1}",
                ]
                for i, line in enumerate(stats):
                    t = self.font_value.render(line, True, TEXT)
                    tx = BOARD_X + (BOARD_W - t.get_width()) // 2
                    self.screen.blit(t, (tx, gy + 55 + i * 32))

                hint = self.font_small.render(
                    "Press R to restart, Q to quit", True, DIM_TEXT)
                hx = BOARD_X + (BOARD_W - hint.get_width()) // 2
                self.screen.blit(hint, (hx, gy + 195))

                pygame.display.flip()

                # Wait for restart or quit
                while self.running:
                    result = self.handle_events()
                    if result == "restart":
                        break
                    if not self.running:
                        break
                    self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "best_model.pt"
    viz = TetrisVisualizer(model_path=model)
    viz.run()
