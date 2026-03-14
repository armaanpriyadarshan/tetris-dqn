"""
Tetris RL Environment — Guideline-compliant

Features:
  - 10x20 visible board + 2 hidden rows above (22 total rows)
  - All 7 tetrominoes with SRS rotation and wall kicks
  - 7-bag randomizer (guarantees each piece once per 7)
  - Hold piece (one swap per piece placement)
  - Line clearing with T-spin detection (3-corner rule)
  - Scoring: singles/doubles/triples/tetrises, T-spins, combos,
    back-to-back, perfect clears
  - Placement-based action space: agent picks (hold, rotation, column)
    and the piece is hard-dropped to that position

Action space:
  Each action is an index into a fixed array of size MAX_ACTIONS = 2 × 4 × 12 = 96.
  Action index encodes: hold_flag * 48 + rotation * 12 + (col + COL_OFFSET)
  Invalid placements are masked. The agent only picks among valid actions.

State:
  - board: (1, VISIBLE_ROWS, COLS) — the visible board
  - piece_info: (22,) — current(7) + next(7) + hold(7) + can_hold(1)
  - action_mask: (MAX_ACTIONS,) — which actions are valid
"""

import numpy as np
from typing import Optional

# ── Board Dimensions ──────────────────────────────────────────────────
VISIBLE_ROWS = 20
HIDDEN_ROWS = 2
TOTAL_ROWS = VISIBLE_ROWS + HIDDEN_ROWS  # 22
COLS = 10

# ── Action Space ──────────────────────────────────────────────────────
# Actions encode (hold, rotation, column) in a fixed-size array.
# Column offset: board column = action_col - COL_OFFSET
# Range [-2, 9] covers all valid piece positions.
NUM_ROTATIONS = 4
NUM_COLUMNS = 12  # board columns -2 through 9
COL_OFFSET = 2    # action_col 0 = board column -2
MAX_ACTIONS = 2 * NUM_ROTATIONS * NUM_COLUMNS  # 96

NUM_PIECES = 7

# ── SRS Piece Definitions ────────────────────────────────────────────
# Each piece has 4 rotation states. Cells are (row, col) offsets within
# the piece's bounding box. Row 0 = top, col 0 = left.
#
# I piece uses a 4×4 bounding box; all others use 3×3.
# These match the Super Rotation System (SRS) standard exactly.

PIECES = {
    "I": [
        [(1, 0), (1, 1), (1, 2), (1, 3)],  # State 0 (spawn)
        [(0, 2), (1, 2), (2, 2), (3, 2)],  # State R
        [(2, 0), (2, 1), (2, 2), (2, 3)],  # State 2
        [(0, 1), (1, 1), (2, 1), (3, 1)],  # State L
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],  # State 0
        [(0, 1), (1, 1), (1, 2), (2, 1)],  # State R
        [(1, 0), (1, 1), (1, 2), (2, 1)],  # State 2
        [(0, 1), (1, 0), (1, 1), (2, 1)],  # State L
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],  # State 0
        [(0, 0), (1, 0), (1, 1), (2, 1)],  # State R
        [(1, 1), (1, 2), (2, 0), (2, 1)],  # State 2
        [(0, 1), (1, 1), (1, 2), (2, 2)],  # State L  (fixed from old version)
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],  # State 0
        [(0, 1), (1, 0), (1, 1), (2, 0)],  # State R
        [(1, 0), (1, 1), (2, 1), (2, 2)],  # State 2
        [(0, 2), (1, 1), (1, 2), (2, 1)],  # State L
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],  # State 0
        [(0, 1), (0, 2), (1, 1), (2, 1)],  # State R
        [(1, 0), (1, 1), (1, 2), (2, 2)],  # State 2
        [(0, 1), (1, 1), (2, 0), (2, 1)],  # State L
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],  # State 0
        [(0, 1), (1, 1), (2, 1), (2, 2)],  # State R
        [(1, 0), (1, 1), (1, 2), (2, 0)],  # State 2
        [(0, 0), (0, 1), (1, 1), (2, 1)],  # State L
    ],
}

PIECE_NAMES = list(PIECES.keys())  # I, O, T, S, Z, J, L

# ── SRS Wall Kick Data ───────────────────────────────────────────────
# Wall kicks: when a rotation collides, try these (drow, dcol) offsets.
# Derived from the SRS offset tables. Convention: row+ = down, col+ = right.
#
# The key insight: wall kicks let pieces "slide" into tight spaces that
# a naive rotation would reject. This is essential for T-spins and for
# making rotation feel good near walls.

# JLSTZ kicks (3×3 bounding box pieces)
_JLSTZ_KICKS = {
    (0, 1): [(0, 0), (0, -1), (-1, -1), (2, 0), (2, -1)],   # 0→R
    (1, 0): [(0, 0), (0, 1), (1, 1), (-2, 0), (-2, 1)],      # R→0
    (1, 2): [(0, 0), (0, 1), (1, 1), (-2, 0), (-2, 1)],      # R→2
    (2, 1): [(0, 0), (0, -1), (-1, -1), (2, 0), (2, -1)],    # 2→R
    (2, 3): [(0, 0), (0, 1), (-1, 1), (2, 0), (2, 1)],       # 2→L
    (3, 2): [(0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)],   # L→2
    (3, 0): [(0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)],   # L→0
    (0, 3): [(0, 0), (0, 1), (-1, 1), (2, 0), (2, 1)],       # 0→L
}

# I piece kicks (4×4 bounding box — different offsets)
_I_KICKS = {
    (0, 1): [(0, 0), (0, -2), (0, 1), (1, -2), (-2, 1)],
    (1, 0): [(0, 0), (0, 2), (0, -1), (-1, 2), (2, -1)],
    (1, 2): [(0, 0), (0, -1), (0, 2), (-2, -1), (1, 2)],
    (2, 1): [(0, 0), (0, 1), (0, -2), (2, 1), (-1, -2)],
    (2, 3): [(0, 0), (0, 2), (0, -1), (-1, 2), (2, -1)],
    (3, 2): [(0, 0), (0, -2), (0, 1), (1, -2), (-2, 1)],
    (3, 0): [(0, 0), (0, -1), (0, 2), (-2, -1), (1, 2)],
    (0, 3): [(0, 0), (0, 1), (0, -2), (2, 1), (-1, -2)],
}

def _get_kicks(piece: str, from_rot: int, to_rot: int):
    """Get wall kick test offsets for a rotation transition."""
    key = (from_rot % 4, to_rot % 4)
    if piece == "I":
        return _I_KICKS.get(key, [(0, 0)])
    elif piece == "O":
        return [(0, 0)]  # O doesn't kick
    else:
        return _JLSTZ_KICKS.get(key, [(0, 0)])


# ── T-Spin Corner Positions ──────────────────────────────────────────
# For each T rotation state, the center cell and which corners are "front."
# Front corners are the two corners on the flat side of the T.
# T-spin: 3+ of 4 corners filled. T-spin mini: 2 corners, but front < 2.

_T_CENTER = {0: (1, 1), 1: (1, 1), 2: (1, 1), 3: (1, 1)}
_T_CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # relative to center
# Front corners by rotation (indices into _T_CORNERS)
_T_FRONT = {
    0: [0, 1],  # top-left, top-right (T points up)
    1: [1, 3],  # top-right, bottom-right (T points right)
    2: [2, 3],  # bottom-left, bottom-right (T points down)
    3: [0, 2],  # top-left, bottom-left (T points left)
}


# ── Reward Shaping ────────────────────────────────────────────────────
# Raw guideline scores are huge. We normalize to keep Q-values manageable.
# These are the RL rewards, not display scores.

REWARD_PER_PIECE = 0.01     # small survival reward
REWARD_GAME_OVER = -2.0
REWARD_LINES = {1: 1.0, 2: 4.0, 3: 9.0, 4: 16.0}  # quadratic scaling
REWARD_TSPIN = {0: 1.0, 1: 3.0, 2: 7.0, 3: 12.0}   # T-spin with N lines
REWARD_TSPIN_MINI = {0: 0.5, 1: 1.5}
REWARD_COMBO_BONUS = 0.5     # per combo count
REWARD_B2B_MULTIPLIER = 1.5  # back-to-back multiplier
REWARD_PERFECT_CLEAR = 10.0


def action_to_tuple(action_idx: int) -> tuple:
    """Decode action index -> (hold, rotation, board_col)."""
    hold = action_idx // 48
    remainder = action_idx % 48
    rotation = remainder // NUM_COLUMNS
    col = (remainder % NUM_COLUMNS) - COL_OFFSET
    return hold, rotation, col


def tuple_to_action(hold: int, rotation: int, col: int) -> int:
    """Encode (hold, rotation, board_col) -> action index."""
    return hold * 48 + rotation * NUM_COLUMNS + (col + COL_OFFSET)


class TetrisEnv:
    """
    Guideline-compliant Tetris with placement-based actions.

    Each step, the agent picks one action = (hold_flag, rotation, column).
    The environment hard-drops the piece to that position, clears lines,
    computes rewards, and spawns the next piece.

    The board uses a 22-row grid (rows 0-1 hidden, rows 2-21 visible).
    Row 0 is the top. Pieces spawn in the hidden rows.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    # ── 7-Bag Randomizer ──────────────────────────────────────────────

    def _fill_bag(self):
        """Shuffle all 7 pieces into the bag."""
        bag = list(range(NUM_PIECES))
        self.rng.shuffle(bag)
        self.bag.extend(bag)

    def _next_from_bag(self) -> int:
        """Draw the next piece index from the bag."""
        if len(self.bag) == 0:
            self._fill_bag()
        return self.bag.pop(0)

    # ── Piece Geometry ────────────────────────────────────────────────

    def _get_cells(self, piece_idx: int, rotation: int, row: int, col: int):
        """Get absolute board coordinates for a piece at (row, col)."""
        name = PIECE_NAMES[piece_idx]
        offsets = PIECES[name][rotation % 4]
        return [(row + dr, col + dc) for dr, dc in offsets]

    def _fits(self, piece_idx: int, rotation: int, row: int, col: int) -> bool:
        """Check if piece fits at position (no overlap, in bounds)."""
        for r, c in self._get_cells(piece_idx, rotation, row, col):
            if r < 0 or r >= TOTAL_ROWS or c < 0 or c >= COLS:
                return False
            if self.board[r, c]:
                return False
        return True

    def _drop_row(self, piece_idx: int, rotation: int, col: int) -> int:
        """Find the lowest row where the piece can be placed (hard drop).
        Returns -1 if the piece can't fit at all (even at the top)."""
        # Start from the top and move down
        last_valid = -1
        for row in range(TOTAL_ROWS):
            if self._fits(piece_idx, rotation, row, col):
                last_valid = row
            elif last_valid >= 0:
                break  # was valid above, now blocked
        return last_valid

    # ── Placement Enumeration ─────────────────────────────────────────

    def _compute_action_mask(self) -> np.ndarray:
        """
        Compute which of the 96 actions are valid right now.

        For hold=0: check placements for current_piece.
        For hold=1: check placements for the piece after holding
                    (only if can_hold is True).
        """
        mask = np.zeros(MAX_ACTIONS, dtype=np.float32)

        for hold in range(2):
            if hold == 1 and not self.can_hold:
                continue

            # Which piece would we be placing?
            if hold == 0:
                piece_idx = self.current_piece
            else:
                piece_idx = self.hold_piece if self.hold_piece >= 0 else self.next_pieces[0]

            for rot in range(NUM_ROTATIONS):
                for col_offset in range(NUM_COLUMNS):
                    col = col_offset - COL_OFFSET
                    drop = self._drop_row(piece_idx, rot, col)
                    if drop >= 0:
                        action = hold * 48 + rot * NUM_COLUMNS + col_offset
                        mask[action] = 1.0

        return mask

    # ── T-Spin Detection ──────────────────────────────────────────────

    def _check_tspin(self, piece_idx: int, rotation: int, row: int, col: int):
        """
        Check for T-spin after placing a T piece.

        3-corner rule: count how many of the 4 diagonal corners around
        the T's center are occupied (by walls or blocks).

        Returns: "tspin", "tspin_mini", or None
        """
        if PIECE_NAMES[piece_idx] != "T":
            return None

        center_dr, center_dc = _T_CENTER[rotation]
        center_r = row + center_dr
        center_c = col + center_dc

        # Count occupied corners
        filled = 0
        front_filled = 0
        front_indices = _T_FRONT[rotation]

        for i, (dr, dc) in enumerate(_T_CORNERS):
            r, c = center_r + dr, center_c + dc
            occupied = (r < 0 or r >= TOTAL_ROWS or c < 0 or c >= COLS
                        or self.board[r, c])
            if occupied:
                filled += 1
                if i in front_indices:
                    front_filled += 1

        if filled >= 3:
            if front_filled >= 2:
                return "tspin"
            else:
                return "tspin_mini"
        return None

    # ── Line Clearing ─────────────────────────────────────────────────

    def _clear_lines(self) -> int:
        """Clear completed lines. Returns number of lines cleared."""
        full_rows = [r for r in range(TOTAL_ROWS) if self.board[r].all()]
        if not full_rows:
            return 0

        num_cleared = len(full_rows)
        mask = np.ones(TOTAL_ROWS, dtype=bool)
        for r in full_rows:
            mask[r] = False
        self.board = np.vstack([
            np.zeros((num_cleared, COLS), dtype=np.float32),
            self.board[mask]
        ])
        self.total_lines += num_cleared
        return num_cleared

    def _is_perfect_clear(self) -> bool:
        """Check if the board is completely empty after clearing."""
        return not self.board.any()

    # ── Scoring ───────────────────────────────────────────────────────

    def _compute_reward(self, lines: int, tspin_type, perfect_clear: bool) -> float:
        """
        Compute RL reward for a placement.

        Scoring hierarchy (from most to least valuable):
          1. Perfect clear (entire board empty)
          2. T-spin triples/doubles
          3. Tetrises (4-line clears)
          4. Back-to-back bonus (consecutive "difficult" clears)
          5. Combos (consecutive line-clearing placements)
          6. Regular line clears
          7. Survival (small per-piece reward)
        """
        reward = REWARD_PER_PIECE

        # Determine if this is a "difficult" clear (Tetris or T-spin with lines)
        is_difficult = False

        if tspin_type == "tspin" and lines >= 0:
            reward += REWARD_TSPIN.get(lines, REWARD_TSPIN[3])
            if lines > 0:
                is_difficult = True
        elif tspin_type == "tspin_mini" and lines >= 0:
            reward += REWARD_TSPIN_MINI.get(lines, REWARD_TSPIN_MINI[1])
            if lines > 0:
                is_difficult = True
        elif lines > 0:
            reward += REWARD_LINES.get(lines, REWARD_LINES[4])
            if lines == 4:
                is_difficult = True

        # Back-to-back: consecutive difficult clears get a multiplier
        if lines > 0 and is_difficult:
            if self.back_to_back:
                reward *= REWARD_B2B_MULTIPLIER
            self.back_to_back = True
        elif lines > 0:
            self.back_to_back = False

        # Combo: consecutive pieces that clear lines
        if lines > 0:
            self.combo += 1
            reward += REWARD_COMBO_BONUS * self.combo
        else:
            self.combo = 0

        # Perfect clear
        if perfect_clear:
            reward += REWARD_PERFECT_CLEAR

        # Track display score (raw guideline-style points)
        self._update_score(lines, tspin_type, perfect_clear)

        return reward

    def _update_score(self, lines: int, tspin_type, perfect_clear: bool):
        """Update the display score (guideline scoring, not RL reward)."""
        level = self.total_lines // 10 + 1
        base = 0

        if tspin_type == "tspin":
            base = {0: 400, 1: 800, 2: 1200, 3: 1600}.get(lines, 1600)
        elif tspin_type == "tspin_mini":
            base = {0: 100, 1: 200}.get(lines, 200)
        elif lines > 0:
            base = {1: 100, 2: 300, 3: 500, 4: 800}.get(lines, 800)

        points = base * level
        if self.back_to_back and lines > 0:
            points = int(points * 1.5)
        if perfect_clear:
            pc_bonus = {1: 800, 2: 1200, 3: 1800, 4: 2000}.get(lines, 2000)
            points += pc_bonus * level

        self.score += points

    # ── State Representation ──────────────────────────────────────────

    def _get_state(self) -> dict:
        """
        Build the state dict fed to the Q-network.

        Returns:
            board: (1, VISIBLE_ROWS, COLS) — visible portion of the board
            piece_info: (22,) — current(7) + next(7) + hold(7) + can_hold(1)
            action_mask: (MAX_ACTIONS,) — valid actions
        """
        # Board: only the visible rows (skip hidden rows 0-1)
        board = self.board[HIDDEN_ROWS:].copy().reshape(1, VISIBLE_ROWS, COLS)

        # Piece info
        current_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        current_oh[self.current_piece] = 1.0

        next_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        next_oh[self.next_pieces[0]] = 1.0

        hold_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        if self.hold_piece >= 0:
            hold_oh[self.hold_piece] = 1.0

        can_hold = np.array([1.0 if self.can_hold else 0.0], dtype=np.float32)
        piece_info = np.concatenate([current_oh, next_oh, hold_oh, can_hold])

        # Action mask
        action_mask = self._compute_action_mask()

        return {
            "board": board,
            "piece_info": piece_info,
            "action_mask": action_mask,
        }

    # ── Core Interface ────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset the environment. Returns initial state."""
        self.board = np.zeros((TOTAL_ROWS, COLS), dtype=np.float32)
        self.score = 0
        self.total_lines = 0
        self.pieces_placed = 0
        self.combo = 0
        self.back_to_back = False
        self.done = False

        # Piece state
        self.bag = []
        self._fill_bag()
        self.current_piece = self._next_from_bag()
        # Keep a queue of upcoming pieces (for display / lookahead)
        self.next_pieces = [self._next_from_bag()]
        self.hold_piece = -1  # -1 = no held piece
        self.can_hold = True

        return self._get_state()

    def step(self, action: int) -> tuple:
        """
        Execute a placement action.

        The action encodes (hold_flag, rotation, column).
        The piece is hard-dropped to the specified column in the
        specified rotation. Lines are cleared, rewards computed,
        and the next piece is spawned.

        Returns: (state, reward, done, info)
        """
        assert not self.done, "Episode is over. Call reset()."
        assert 0 <= action < MAX_ACTIONS, f"Invalid action index: {action}"

        hold, rotation, col = action_to_tuple(action)

        # ── Hold logic ────────────────────────────────────────────────
        if hold:
            assert self.can_hold, "Cannot hold again this turn."
            if self.hold_piece < 0:
                # First hold: current → hold, draw from queue
                self.hold_piece = self.current_piece
                self.current_piece = self.next_pieces.pop(0)
                self.next_pieces.append(self._next_from_bag())
            else:
                # Swap current and hold
                self.current_piece, self.hold_piece = self.hold_piece, self.current_piece
            self.can_hold = False

        piece_idx = self.current_piece

        # ── Place the piece ───────────────────────────────────────────
        drop_row = self._drop_row(piece_idx, rotation, col)
        assert drop_row >= 0, f"Invalid placement: piece {PIECE_NAMES[piece_idx]} rot={rotation} col={col}"

        # Lock piece into board
        for r, c in self._get_cells(piece_idx, rotation, drop_row, col):
            self.board[r, c] = 1.0

        # ── T-spin check (before clearing lines) ─────────────────────
        tspin_type = self._check_tspin(piece_idx, rotation, drop_row, col)

        # ── Clear lines ───────────────────────────────────────────────
        lines = self._clear_lines()
        perfect_clear = self._is_perfect_clear() if lines > 0 else False

        # ── Reward ────────────────────────────────────────────────────
        reward = self._compute_reward(lines, tspin_type, perfect_clear)
        self.pieces_placed += 1

        # ── Spawn next piece ──────────────────────────────────────────
        self.current_piece = self.next_pieces.pop(0)
        self.next_pieces.append(self._next_from_bag())
        self.can_hold = True

        # Check game over: if no valid placements exist for the new piece
        action_mask = self._compute_action_mask()
        if not action_mask.any():
            self.done = True
            reward += REWARD_GAME_OVER

        state = self._get_state()
        info = {
            "score": self.score,
            "lines_cleared": self.total_lines,
            "pieces_placed": self.pieces_placed,
            "combo": self.combo,
            "back_to_back": self.back_to_back,
            "tspin": tspin_type,
            "lines_this_step": lines,
            "perfect_clear": perfect_clear,
        }
        return state, reward, self.done, info

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self) -> str:
        """Return a string representation of the visible board."""
        lines = []
        lines.append("+" + "-" * COLS + "+")
        for r in range(HIDDEN_ROWS, TOTAL_ROWS):
            row_str = "|"
            for c in range(COLS):
                row_str += "#" if self.board[r, c] else "."
            row_str += "|"
            lines.append(row_str)
        lines.append("+" + "-" * COLS + "+")

        piece_name = PIECE_NAMES[self.current_piece]
        next_name = PIECE_NAMES[self.next_pieces[0]]
        hold_name = PIECE_NAMES[self.hold_piece] if self.hold_piece >= 0 else "-"

        lines.append(
            f"Piece: {piece_name}  Next: {next_name}  Hold: {hold_name}  "
            f"Score: {self.score}  Lines: {self.total_lines}  "
            f"Combo: {self.combo}"
        )
        return "\n".join(lines)
