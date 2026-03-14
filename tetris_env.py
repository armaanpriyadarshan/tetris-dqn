"""
Tetris RL Environment — Guideline-compliant with engineered features

Features:
  - 10x20 visible board + 2 hidden rows above (22 total rows)
  - All 7 tetrominoes with SRS rotation and wall kicks
  - 7-bag randomizer
  - Hold piece
  - T-spin detection (3-corner rule)
  - Full scoring: T-spins, combos, back-to-back, perfect clears
  - Placement-based action space: agent picks (hold, rotation, column)

Observation space:
  - board: (1, 20, 10) — raw board for CNN
  - piece_info: (22,) — current/next/hold pieces + can_hold flag
  - board_features: (34,) — engineered features (normalized to ~[0,1]):
      column_heights (10), holes_per_col (10), wells (10),
      bumpiness (1), max_height (1), agg_height (1), total_holes (1)
  - action_mask: (96,) — which actions are valid

Reward:
  Event rewards:   exponential line clears, T-spin bonuses, combo, B2B, perfect clear
  Shaping rewards: penalties for creating holes, increasing height/bumpiness
  The shaping uses CHANGES in board features (potential-based), preserving
  the optimal policy while giving gradient signal on every step.
"""

import numpy as np
from typing import Optional

# ── Board Dimensions ──────────────────────────────────────────────────
VISIBLE_ROWS = 20
HIDDEN_ROWS = 2
TOTAL_ROWS = VISIBLE_ROWS + HIDDEN_ROWS  # 22
COLS = 10

# ── Action Space ──────────────────────────────────────────────────────
NUM_ROTATIONS = 4
NUM_COLUMNS = 12  # board columns -2 through 9
COL_OFFSET = 2
MAX_ACTIONS = 2 * NUM_ROTATIONS * NUM_COLUMNS  # 96
NUM_PIECES = 7

# ── Board Features ────────────────────────────────────────────────────
# col_heights(10) + holes_per_col(10) + wells(10) + bumpiness(1)
# + max_height(1) + agg_height(1) + total_holes(1) = 34
BOARD_FEATURES_SIZE = 34

# ── SRS Piece Definitions ────────────────────────────────────────────
PIECES = {
    "I": [
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(1, 1), (1, 2), (2, 0), (2, 1)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 2), (1, 1), (1, 2), (2, 1)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

PIECE_NAMES = list(PIECES.keys())

# ── SRS Wall Kick Data ───────────────────────────────────────────────
_JLSTZ_KICKS = {
    (0, 1): [(0, 0), (0, -1), (-1, -1), (2, 0), (2, -1)],
    (1, 0): [(0, 0), (0, 1), (1, 1), (-2, 0), (-2, 1)],
    (1, 2): [(0, 0), (0, 1), (1, 1), (-2, 0), (-2, 1)],
    (2, 1): [(0, 0), (0, -1), (-1, -1), (2, 0), (2, -1)],
    (2, 3): [(0, 0), (0, 1), (-1, 1), (2, 0), (2, 1)],
    (3, 2): [(0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)],
    (3, 0): [(0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)],
    (0, 3): [(0, 0), (0, 1), (-1, 1), (2, 0), (2, 1)],
}

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

def _get_kicks(piece, from_rot, to_rot):
    key = (from_rot % 4, to_rot % 4)
    if piece == "I":
        return _I_KICKS.get(key, [(0, 0)])
    elif piece == "O":
        return [(0, 0)]
    else:
        return _JLSTZ_KICKS.get(key, [(0, 0)])


# ── T-Spin Detection ─────────────────────────────────────────────────
_T_CENTER = {0: (1, 1), 1: (1, 1), 2: (1, 1), 3: (1, 1)}
_T_CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_T_FRONT = {
    0: [0, 1],  # T points up: top corners are front
    1: [1, 3],  # T points right
    2: [2, 3],  # T points down
    3: [0, 2],  # T points left
}


# ── Reward Constants ──────────────────────────────────────────────────
#
# Event rewards — the primary learning signal.
# Exponential line clear scaling: base^(n-1)
# Why exponential? In Tetris, clearing 4 lines at once (Tetris) is far more
# efficient than 4 singles: it clears the same rows but uses 1 piece instead
# of 4 setups. Exponential rewards encode this: a Tetris is worth 27× a single,
# strongly incentivizing the agent to stack for big clears.
#
REWARD_LINE_CLEAR = {1: 1.0, 2: 3.0, 3: 9.0, 4: 27.0}

# T-spin rewards (on top of line clear reward).
# T-spins are the most advanced Tetris technique. Rewarding them
# separately encourages the agent to learn T-spin setups.
REWARD_TSPIN = {0: 1.0, 1: 5.0, 2: 10.0, 3: 15.0}
REWARD_TSPIN_MINI = {0: 0.5, 1: 2.0}

# Combo/B2B/perfect clear
REWARD_COMBO = 0.5         # per combo count
REWARD_B2B_MULT = 1.5      # multiplier for consecutive difficult clears
REWARD_PERFECT_CLEAR = 15.0

# Shaping penalties — applied to CHANGES in board features.
# These implement potential-based reward shaping:
#   F(s, s') ≈ Φ(s') - Φ(s)  where  Φ = -w·feature
#
# Magnitudes are calibrated so that:
#   - Without line clears, the agent learns to minimize holes/height/bumps
#   - Line clear rewards dominate when available (don't prevent Tetris setups)
#   - A single hole costs 0.5 reward ≈ 50 pieces of survival reward
#
PENALTY_HOLE = -0.5        # per new hole created
PENALTY_HEIGHT = -0.1      # per row of max height increase
PENALTY_BUMP = -0.05       # per unit of bumpiness increase

# Height danger: extra penalty when board gets dangerously tall.
# This creates urgency — the agent "panics" near the top and prioritizes
# clearing over stacking. Without this, the agent might keep stacking
# because the game-over penalty is too far in the future for γ=0.99
# to propagate effectively (discounted by γ^N where N is unknown).
HEIGHT_DANGER_THRESHOLD = 15  # rows (out of 20 visible)
PENALTY_HEIGHT_DANGER = -0.5  # per row above threshold

REWARD_PER_PIECE = 0.01
REWARD_GAME_OVER = -5.0


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

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    # ── 7-Bag Randomizer ──────────────────────────────────────────────

    def _fill_bag(self):
        bag = list(range(NUM_PIECES))
        self.rng.shuffle(bag)
        self.bag.extend(bag)

    def _next_from_bag(self) -> int:
        if len(self.bag) == 0:
            self._fill_bag()
        return self.bag.pop(0)

    # ── Piece Geometry ────────────────────────────────────────────────

    def _get_cells(self, piece_idx, rotation, row, col):
        offsets = PIECES[PIECE_NAMES[piece_idx]][rotation % 4]
        return [(row + dr, col + dc) for dr, dc in offsets]

    def _fits(self, piece_idx, rotation, row, col):
        for r, c in self._get_cells(piece_idx, rotation, row, col):
            if r < 0 or r >= TOTAL_ROWS or c < 0 or c >= COLS:
                return False
            if self.board[r, c]:
                return False
        return True

    def _drop_row(self, piece_idx, rotation, col):
        last_valid = -1
        for row in range(TOTAL_ROWS):
            if self._fits(piece_idx, rotation, row, col):
                last_valid = row
            elif last_valid >= 0:
                break
        return last_valid

    # ── Action Mask ───────────────────────────────────────────────────

    def _compute_action_mask(self):
        mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
        for hold in range(2):
            if hold == 1 and not self.can_hold:
                continue
            if hold == 0:
                piece_idx = self.current_piece
            else:
                piece_idx = self.hold_piece if self.hold_piece >= 0 else self.next_pieces[0]
            for rot in range(NUM_ROTATIONS):
                for col_offset in range(NUM_COLUMNS):
                    col = col_offset - COL_OFFSET
                    if self._drop_row(piece_idx, rot, col) >= 0:
                        mask[hold * 48 + rot * NUM_COLUMNS + col_offset] = 1.0
        return mask

    # ── T-Spin Detection ──────────────────────────────────────────────

    def _check_tspin(self, piece_idx, rotation, row, col):
        if PIECE_NAMES[piece_idx] != "T":
            return None
        center_dr, center_dc = _T_CENTER[rotation]
        cr, cc = row + center_dr, col + center_dc
        filled = 0
        front_filled = 0
        front_indices = _T_FRONT[rotation]
        for i, (dr, dc) in enumerate(_T_CORNERS):
            r, c = cr + dr, cc + dc
            occ = (r < 0 or r >= TOTAL_ROWS or c < 0 or c >= COLS
                   or self.board[r, c])
            if occ:
                filled += 1
                if i in front_indices:
                    front_filled += 1
        if filled >= 3:
            return "tspin" if front_filled >= 2 else "tspin_mini"
        return None

    # ── Line Clearing ─────────────────────────────────────────────────

    def _clear_lines(self):
        full_rows = [r for r in range(TOTAL_ROWS) if self.board[r].all()]
        if not full_rows:
            return 0
        n = len(full_rows)
        mask = np.ones(TOTAL_ROWS, dtype=bool)
        for r in full_rows:
            mask[r] = False
        self.board = np.vstack([
            np.zeros((n, COLS), dtype=np.float32),
            self.board[mask]
        ])
        self.total_lines += n
        return n

    # ══════════════════════════════════════════════════════════════════
    #  BOARD FEATURES
    # ══════════════════════════════════════════════════════════════════

    def _compute_board_features_raw(self) -> dict:
        """
        Compute raw (unnormalized) board features for reward shaping.

        These features encode the strategic quality of the board:

        Column heights (10):
            The topmost filled cell in each column. A flat, low surface
            is ideal — it maximizes options for piece placement.

        Holes per column (10):
            Empty cells with a filled cell above them. Holes are the #1
            enemy in Tetris: you can't clear the row above without first
            filling the hole, creating a cascading problem. A single hole
            effectively wastes 1+ rows of board space.

        Wells (10):
            A well is a column lower than both neighbors. Wells of depth
            1-4 are useful (drop I-piece for Tetris). Wells deeper than 4
            are wasteful — you can never clear more than 4 lines at once.

            For edge columns, the wall acts as an infinitely tall neighbor,
            which correctly models that the wall prevents pieces from
            sliding off.

        Bumpiness (1):
            Sum of |h_i - h_{i+1}|. Measures surface roughness. Flat boards
            are easier to play because more pieces fit without creating holes.

        Max height (1):
            Height of the tallest column. High boards are dangerous — a few
            bad pieces can cause game over with no recovery time.

        Aggregate height (1):
            Sum of all column heights. Represents total "mass" on the board.

        Total holes (1):
            Sum of holes across all columns. The single most predictive
            feature of Tetris performance.
        """
        visible = self.board[HIDDEN_ROWS:]

        # Column heights
        col_heights = np.zeros(COLS, dtype=np.float32)
        for c in range(COLS):
            filled = np.where(visible[:, c] > 0)[0]
            if len(filled) > 0:
                col_heights[c] = VISIBLE_ROWS - filled[0]

        # Holes per column
        holes = np.zeros(COLS, dtype=np.float32)
        for c in range(COLS):
            block_found = False
            for r in range(VISIBLE_ROWS):
                if visible[r, c]:
                    block_found = True
                elif block_found:
                    holes[c] += 1

        # Wells: depth = min(left_h, right_h) - this_h, clamped >= 0
        # Walls act as tall neighbors (height = VISIBLE_ROWS)
        wells = np.zeros(COLS, dtype=np.float32)
        for c in range(COLS):
            left_h = col_heights[c - 1] if c > 0 else VISIBLE_ROWS
            right_h = col_heights[c + 1] if c < COLS - 1 else VISIBLE_ROWS
            wells[c] = max(0, min(left_h, right_h) - col_heights[c])

        bumpiness = float(np.sum(np.abs(np.diff(col_heights))))
        max_height = float(col_heights.max())
        agg_height = float(col_heights.sum())
        total_holes = float(holes.sum())

        return {
            "col_heights": col_heights,
            "holes": holes,
            "wells": wells,
            "bumpiness": bumpiness,
            "max_height": max_height,
            "agg_height": agg_height,
            "total_holes": total_holes,
        }

    def _normalize_features(self, raw: dict) -> np.ndarray:
        """
        Normalize raw features to approximately [0, 1] for the Q-network.

        Why normalize?
        - Neural networks train faster when inputs have similar scales.
        - Without normalization, col_heights (0-20) would dominate
          bumpiness (0-180) in the gradient, making the network
          struggle to learn from smaller-magnitude features.
        - We divide by the theoretical maximum of each feature.
        """
        return np.concatenate([
            raw["col_heights"] / VISIBLE_ROWS,                      # 10
            raw["holes"] / VISIBLE_ROWS,                            # 10
            raw["wells"] / VISIBLE_ROWS,                            # 10
            [raw["bumpiness"] / (VISIBLE_ROWS * (COLS - 1))],      # 1
            [raw["max_height"] / VISIBLE_ROWS],                     # 1
            [raw["agg_height"] / (VISIBLE_ROWS * COLS)],            # 1
            [raw["total_holes"] / (VISIBLE_ROWS * COLS)],           # 1
        ]).astype(np.float32)                                       # = 34

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self, lines: int, tspin_type, perfect_clear: bool,
                        raw_before: dict, raw_after: dict) -> float:
        """
        Compute the RL reward, combining event rewards and shaping.

        The reward has two components:

        1. EVENT REWARDS (sparse, large):
           Line clears, T-spins, combos, B2B, perfect clears.
           These are the TRUE objective — clear lines efficiently.

        2. SHAPING REWARDS (dense, small):
           Penalties for changes in board quality features.
           These guide the agent between line-clearing events.

           Shaping is based on CHANGES in features (Δ = after - before):
             reward += w · Δfeature

           This is approximately potential-based shaping:
             F(s, s') ≈ Φ(s') - Φ(s)
           where Φ(s) = -w_holes·holes - w_height·height - w_bump·bump

           Ng et al. (1999) proved that potential-based shaping preserves
           the set of optimal policies. Using changes (not absolutes) is
           critical: absolute penalties would create a constant drag that
           swamps the line-clearing signal.

        Magnitude calibration:
           Event rewards >> shaping penalties >> survival reward
           Tetris (27.0) >> new hole (-0.5) >> survival (0.01)

           This ensures the agent learns "clear lines" as the primary
           goal, with "avoid holes" as guidance between clears. If shaping
           penalties were too large, the agent would learn to play
           ultra-conservatively (keep board flat) and never stack for Tetrises.
        """
        reward = REWARD_PER_PIECE

        # ── Event rewards ─────────────────────────────────────────────
        is_difficult = False

        if tspin_type == "tspin":
            reward += REWARD_TSPIN.get(lines, REWARD_TSPIN[3])
            if lines > 0:
                is_difficult = True
        elif tspin_type == "tspin_mini":
            reward += REWARD_TSPIN_MINI.get(lines, REWARD_TSPIN_MINI[1])
            if lines > 0:
                is_difficult = True

        if lines > 0:
            reward += REWARD_LINE_CLEAR.get(lines, REWARD_LINE_CLEAR[4])
            if lines == 4:
                is_difficult = True

        # Back-to-back: consecutive difficult clears
        if lines > 0 and is_difficult:
            if self.back_to_back:
                reward *= REWARD_B2B_MULT
            self.back_to_back = True
        elif lines > 0:
            self.back_to_back = False

        # Combo: consecutive line-clearing placements
        if lines > 0:
            self.combo += 1
            reward += REWARD_COMBO * self.combo
        else:
            self.combo = 0

        if perfect_clear:
            reward += REWARD_PERFECT_CLEAR

        # ── Shaping rewards (potential-based) ─────────────────────────
        delta_holes = raw_after["total_holes"] - raw_before["total_holes"]
        delta_height = raw_after["max_height"] - raw_before["max_height"]
        delta_bump = raw_after["bumpiness"] - raw_before["bumpiness"]

        reward += PENALTY_HOLE * delta_holes
        reward += PENALTY_HEIGHT * delta_height
        reward += PENALTY_BUMP * delta_bump

        # Height danger: escalating penalty near the top
        h = raw_after["max_height"]
        if h > HEIGHT_DANGER_THRESHOLD:
            reward += PENALTY_HEIGHT_DANGER * (h - HEIGHT_DANGER_THRESHOLD)

        # ── Display score ─────────────────────────────────────────────
        self._update_score(lines, tspin_type, perfect_clear)

        return reward

    def _update_score(self, lines, tspin_type, perfect_clear):
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
            pc = {1: 800, 2: 1200, 3: 1800, 4: 2000}.get(lines, 2000)
            points += pc * level
        self.score += points

    # ── State ─────────────────────────────────────────────────────────

    def _get_state(self) -> dict:
        """
        Build the observation dict.

        The state has four components:
          board:          (1, 20, 10)  — raw board for CNN spatial processing
          piece_info:     (22,)       — categorical piece identities
          board_features: (34,)       — engineered features for MLP processing
          action_mask:    (96,)       — which placements are valid
        """
        board = self.board[HIDDEN_ROWS:].copy().reshape(1, VISIBLE_ROWS, COLS)

        current_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        current_oh[self.current_piece] = 1.0
        next_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        next_oh[self.next_pieces[0]] = 1.0
        hold_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        if self.hold_piece >= 0:
            hold_oh[self.hold_piece] = 1.0
        can_hold = np.array([1.0 if self.can_hold else 0.0], dtype=np.float32)
        piece_info = np.concatenate([current_oh, next_oh, hold_oh, can_hold])

        board_features = self._normalize_features(self._current_features)

        action_mask = self._compute_action_mask()

        return {
            "board": board,
            "piece_info": piece_info,
            "board_features": board_features,
            "action_mask": action_mask,
        }

    # ── Core Interface ────────────────────────────────────────────────

    def reset(self) -> dict:
        self.board = np.zeros((TOTAL_ROWS, COLS), dtype=np.float32)
        self.score = 0
        self.total_lines = 0
        self.pieces_placed = 0
        self.combo = 0
        self.back_to_back = False
        self.done = False

        self.bag = []
        self._fill_bag()
        self.current_piece = self._next_from_bag()
        self.next_pieces = [self._next_from_bag()]
        self.hold_piece = -1
        self.can_hold = True

        self._current_features = self._compute_board_features_raw()
        return self._get_state()

    def step(self, action: int) -> tuple:
        assert not self.done
        assert 0 <= action < MAX_ACTIONS

        hold, rotation, col = action_to_tuple(action)

        # ── Hold ──────────────────────────────────────────────────────
        if hold:
            assert self.can_hold
            if self.hold_piece < 0:
                self.hold_piece = self.current_piece
                self.current_piece = self.next_pieces.pop(0)
                self.next_pieces.append(self._next_from_bag())
            else:
                self.current_piece, self.hold_piece = self.hold_piece, self.current_piece
            self.can_hold = False

        piece_idx = self.current_piece

        # ── Features BEFORE placement ─────────────────────────────────
        raw_before = self._compute_board_features_raw()

        # ── Place piece ───────────────────────────────────────────────
        drop_row = self._drop_row(piece_idx, rotation, col)
        assert drop_row >= 0

        for r, c in self._get_cells(piece_idx, rotation, drop_row, col):
            self.board[r, c] = 1.0

        # ── T-spin, clear, perfect ────────────────────────────────────
        tspin_type = self._check_tspin(piece_idx, rotation, drop_row, col)
        lines = self._clear_lines()
        perfect_clear = not self.board.any() if lines > 0 else False

        # ── Features AFTER placement ──────────────────────────────────
        raw_after = self._compute_board_features_raw()
        self._current_features = raw_after

        # ── Reward ────────────────────────────────────────────────────
        reward = self._compute_reward(lines, tspin_type, perfect_clear,
                                      raw_before, raw_after)
        self.pieces_placed += 1

        # ── Next piece ────────────────────────────────────────────────
        self.current_piece = self.next_pieces.pop(0)
        self.next_pieces.append(self._next_from_bag())
        self.can_hold = True

        # ── Game over check ───────────────────────────────────────────
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
            "holes": raw_after["total_holes"],
            "max_height": raw_after["max_height"],
            "bumpiness": raw_after["bumpiness"],
        }
        return state, reward, self.done, info

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self) -> str:
        lines = []
        lines.append("+" + "-" * COLS + "+")
        for r in range(HIDDEN_ROWS, TOTAL_ROWS):
            row_str = "|"
            for c in range(COLS):
                row_str += "#" if self.board[r, c] else "."
            row_str += "|"
            lines.append(row_str)
        lines.append("+" + "-" * COLS + "+")
        piece = PIECE_NAMES[self.current_piece]
        nxt = PIECE_NAMES[self.next_pieces[0]]
        hld = PIECE_NAMES[self.hold_piece] if self.hold_piece >= 0 else "-"
        feats = self._current_features
        lines.append(
            f"Piece: {piece}  Next: {nxt}  Hold: {hld}  "
            f"Score: {self.score}  Lines: {self.total_lines}"
        )
        lines.append(
            f"Holes: {feats['total_holes']:.0f}  "
            f"Height: {feats['max_height']:.0f}  "
            f"Bump: {feats['bumpiness']:.0f}  "
            f"Combo: {self.combo}"
        )
        return "\n".join(lines)
