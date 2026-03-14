"""
Tetris RL Environment

Exposes a gym-like interface:
  - reset() -> state
  - step(action) -> (state, reward, done, info)

State representation:
  - 2-channel 20x10 grid (board + current piece)
  - 7-dim one-hot vector for next piece

Action space: 6 discrete actions
  0: move left
  1: move right
  2: rotate clockwise
  3: rotate counter-clockwise
  4: hard drop
  5: no-op (piece falls by gravity)
"""

import numpy as np
from typing import Optional

# ── Tetromino Definitions ──────────────────────────────────────────────
# Each piece is defined as a list of rotations.
# Each rotation is a list of (row, col) offsets from the piece's origin.
# Using the Super Rotation System (SRS) — the modern Tetris standard.

TETROMINOES = {
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],
    "S": [
        [(1, 0), (1, 1), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

PIECE_NAMES = list(TETROMINOES.keys())  # consistent ordering for one-hot
NUM_PIECES = len(PIECE_NAMES)

# Board dimensions
ROWS = 20
COLS = 10

# Actions
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_ROTATE_CW = 2
ACTION_ROTATE_CCW = 3
ACTION_HARD_DROP = 4
ACTION_NOOP = 5
NUM_ACTIONS = 6

# Rewards
REWARD_LINE_CLEAR = {1: 1, 2: 4, 3: 9, 4: 16}
REWARD_GAME_OVER = -2
REWARD_STEP = 0.01


class TetrisEnv:
    """
    Tetris environment with gym-like interface.

    The board is a 2D numpy array of shape (ROWS, COLS).
    0 = empty, 1 = filled.
    Row 0 is the top of the board.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.board = np.zeros((ROWS, COLS), dtype=np.float32)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.current_piece = None
        self.current_rotation = 0
        self.current_pos = [0, 0]  # [row, col] of piece origin
        self.next_piece = None
        self.done = False

    # ── Piece Management ───────────────────────────────────────────────

    def _random_piece(self) -> str:
        return PIECE_NAMES[self.rng.randint(NUM_PIECES)]

    def _get_cells(self, piece: str = None, rotation: int = None,
                   pos: list = None) -> list:
        """Get absolute board coordinates for a piece at a position."""
        piece = piece or self.current_piece
        rotation = rotation if rotation is not None else self.current_rotation
        pos = pos or self.current_pos
        offsets = TETROMINOES[piece][rotation % 4]
        return [(pos[0] + dr, pos[1] + dc) for dr, dc in offsets]

    def _is_valid(self, piece: str = None, rotation: int = None,
                  pos: list = None) -> bool:
        """Check if a piece placement is valid (in bounds, no overlap)."""
        for r, c in self._get_cells(piece, rotation, pos):
            if r < 0 or r >= ROWS or c < 0 or c >= COLS:
                return False
            if self.board[r, c]:
                return False
        return True

    def _spawn_piece(self):
        """Spawn a new piece at the top-center of the board."""
        self.current_piece = self.next_piece
        self.next_piece = self._random_piece()
        self.current_rotation = 0
        # Spawn at top, horizontally centered
        self.current_pos = [0, COLS // 2 - 1]

        # If the spawn position is invalid, game over
        if not self._is_valid():
            self.done = True

    def _lock_piece(self):
        """Lock the current piece into the board."""
        for r, c in self._get_cells():
            self.board[r, c] = 1.0
        self.pieces_placed += 1

    # ── Line Clearing ──────────────────────────────────────────────────

    def _clear_lines(self) -> int:
        """Clear completed lines. Returns number of lines cleared."""
        # A line is complete when every cell in the row is filled
        full_rows = [r for r in range(ROWS) if self.board[r].all()]
        if not full_rows:
            return 0

        num_cleared = len(full_rows)
        # Remove full rows, add empty rows at top
        mask = np.ones(ROWS, dtype=bool)
        for r in full_rows:
            mask[r] = False
        self.board = np.vstack([
            np.zeros((num_cleared, COLS), dtype=np.float32),
            self.board[mask]
        ])
        self.lines_cleared += num_cleared
        return num_cleared

    # ── State Representation ───────────────────────────────────────────

    def _get_state(self) -> dict:
        """
        Build the state that gets fed to the Q-network.

        Returns:
            dict with:
                'grid': (2, 20, 10) float32 array
                    channel 0 = locked board
                    channel 1 = current piece position
                'next_piece': (7,) float32 one-hot vector
        """
        grid = np.zeros((2, ROWS, COLS), dtype=np.float32)
        grid[0] = self.board.copy()

        if not self.done and self.current_piece:
            for r, c in self._get_cells():
                if 0 <= r < ROWS and 0 <= c < COLS:
                    grid[1, r, c] = 1.0

        next_one_hot = np.zeros(NUM_PIECES, dtype=np.float32)
        if self.next_piece:
            next_one_hot[PIECE_NAMES.index(self.next_piece)] = 1.0

        return {"grid": grid, "next_piece": next_one_hot}

    # ── Core Interface ─────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset the environment. Returns initial state."""
        self.board = np.zeros((ROWS, COLS), dtype=np.float32)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.done = False
        self.next_piece = self._random_piece()
        self._spawn_piece()
        return self._get_state()

    def step(self, action: int) -> tuple:
        """
        Take one step in the environment.

        Each step:
          1. Apply the agent's action (move/rotate)
          2. Apply gravity (piece falls one row)
          3. If piece can't fall, lock it and spawn next

        Returns: (state, reward, done, info)
        """
        assert not self.done, "Episode is over. Call reset()."
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        reward = REWARD_STEP  # survival reward

        # 1. Apply action
        if action == ACTION_LEFT:
            new_pos = [self.current_pos[0], self.current_pos[1] - 1]
            if self._is_valid(pos=new_pos):
                self.current_pos = new_pos

        elif action == ACTION_RIGHT:
            new_pos = [self.current_pos[0], self.current_pos[1] + 1]
            if self._is_valid(pos=new_pos):
                self.current_pos = new_pos

        elif action == ACTION_ROTATE_CW:
            new_rot = (self.current_rotation + 1) % 4
            if self._is_valid(rotation=new_rot):
                self.current_rotation = new_rot

        elif action == ACTION_ROTATE_CCW:
            new_rot = (self.current_rotation - 1) % 4
            if self._is_valid(rotation=new_rot):
                self.current_rotation = new_rot

        elif action == ACTION_HARD_DROP:
            # Drop piece as far as it can go
            while self._is_valid(pos=[self.current_pos[0] + 1,
                                      self.current_pos[1]]):
                self.current_pos[0] += 1

        # ACTION_NOOP: do nothing

        # 2. Apply gravity — piece falls one row
        gravity_pos = [self.current_pos[0] + 1, self.current_pos[1]]

        if action == ACTION_HARD_DROP or not self._is_valid(pos=gravity_pos):
            # Piece has landed — lock it
            self._lock_piece()
            lines = self._clear_lines()
            if lines > 0:
                reward += REWARD_LINE_CLEAR[lines]
                self.score += REWARD_LINE_CLEAR[lines]

            # Spawn next piece (may trigger game over)
            self._spawn_piece()
            if self.done:
                reward += REWARD_GAME_OVER
        else:
            # Piece falls
            self.current_pos = gravity_pos

        state = self._get_state()
        info = {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
        }
        return state, reward, self.done, info

    # ── Rendering (for debugging) ──────────────────────────────────────

    def render(self) -> str:
        """Return a string representation of the board."""
        display = self.board.copy()
        if not self.done and self.current_piece:
            for r, c in self._get_cells():
                if 0 <= r < ROWS and 0 <= c < COLS:
                    display[r, c] = 2.0  # distinguish active piece

        lines = []
        lines.append("+" + "-" * COLS + "+")
        for r in range(ROWS):
            row_str = "|"
            for c in range(COLS):
                if display[r, c] == 2.0:
                    row_str += "@"
                elif display[r, c] == 1.0:
                    row_str += "#"
                else:
                    row_str += "."
            row_str += "|"
            lines.append(row_str)
        lines.append("+" + "-" * COLS + "+")
        lines.append(f"Score: {self.score}  Lines: {self.lines_cleared}  "
                     f"Pieces: {self.pieces_placed}")
        return "\n".join(lines)
