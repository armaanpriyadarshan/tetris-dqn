"""
DQN Agent for Tetris — Built from scratch

Components:
  1. Replay Buffer — uniform random sampling from fixed-capacity deque
  2. Q-Network — CNN backbone (board) + MLP embeddings (pieces, features) → FC head
  3. Epsilon-greedy — linear decay with action masking
  4. Target Network — frozen copy, hard-synced every K steps

Architecture:
  The Q-network processes three types of input:

    CNN path:   board (1, 20, 10)  →  Conv layers  →  3200-dim spatial features
    MLP path 1: piece_info (22,)   →  Linear+ReLU  →  64-dim piece embedding
    MLP path 2: board_features (34,) → Linear+ReLU →  64-dim feature embedding

    Combined:   [3200 + 64 + 64 = 3328]  →  FC(512) → FC(96) Q-values

  Why three separate input paths?
    - The board is SPATIAL data → CNN (weight sharing, translation equivariance)
    - Piece identity is CATEGORICAL → embedding layer (like NLP word embeddings)
    - Board features are PRE-COMPUTED STATISTICS → MLP (direct numerical reasoning)

  The board features (holes, heights, etc.) are redundant with the raw board
  in theory — the CNN could learn to compute them. In practice, the CNN
  learns these features slowly, so providing them explicitly gives the
  agent a huge head start. The CNN then focuses on learning patterns the
  hand-crafted features miss (e.g., specific piece-shaped gaps, T-spin setups).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tetris_env import MAX_ACTIONS, NUM_PIECES, BOARD_FEATURES_SIZE


# Piece info size: current(7) + next(7) + hold(7) + can_hold(1) = 22
PIECE_INFO_SIZE = NUM_PIECES * 3 + 1


# ══════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-capacity buffer with uniform random sampling.

    Stores transitions as (state_dict, action, reward, next_state_dict, done).
    At sample time, extracts and stacks each component into batch tensors.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "board": torch.FloatTensor(
                np.array([s["board"] for s in states])),
            "piece_info": torch.FloatTensor(
                np.array([s["piece_info"] for s in states])),
            "board_features": torch.FloatTensor(
                np.array([s["board_features"] for s in states])),
            "next_board": torch.FloatTensor(
                np.array([s["board"] for s in next_states])),
            "next_piece_info": torch.FloatTensor(
                np.array([s["piece_info"] for s in next_states])),
            "next_board_features": torch.FloatTensor(
                np.array([s["board_features"] for s in next_states])),
            "next_mask": torch.FloatTensor(
                np.array([s["action_mask"] for s in next_states])),
            "actions": torch.LongTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "dones": torch.FloatTensor(dones),
        }

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════
#  Q-NETWORK
# ══════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Q-value network with three input streams:

    1. CNN backbone for spatial board features
       Conv(1→32) + BN + ReLU → Conv(32→64) + BN + ReLU → Conv(64→64, stride=2) + BN + ReLU
       Output: 64 × 10 × 5 = 3200

    2. Piece embedding for categorical piece identity
       Linear(22→64) + ReLU
       Output: 64

    3. Board feature embedding for engineered statistics
       Linear(34→64) + ReLU
       Output: 64

    Combined: 3328 → Linear(512) + ReLU → Linear(96)
    """

    def __init__(self):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # (1, 20, 10) → stride-2 → (64, 10, 5) → flatten → 3200
        self._conv_out = 64 * 10 * 5

        # ── Piece embedding ───────────────────────────────────────────
        self.piece_embed = nn.Sequential(
            nn.Linear(PIECE_INFO_SIZE, 64),
            nn.ReLU(),
        )

        # ── Board features embedding ─────────────────────────────────
        self.feat_embed = nn.Sequential(
            nn.Linear(BOARD_FEATURES_SIZE, 64),
            nn.ReLU(),
        )

        # ── Decision head ─────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self._conv_out + 64 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, MAX_ACTIONS),
        )

    def forward(self, board, piece_info, board_features):
        """
        Args:
            board:          (batch, 1, 20, 10)
            piece_info:     (batch, 22)
            board_features: (batch, 34)
        Returns:
            q_values:       (batch, 96)
        """
        x = self.conv(board).view(board.size(0), -1)
        p = self.piece_embed(piece_info)
        f = self.feat_embed(board_features)
        return self.head(torch.cat([x, p, f], dim=1))


# ══════════════════════════════════════════════════════════════════════
#  DQN AGENT
# ══════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    DQN with replay buffer, target network, and masked ε-greedy.

    Epsilon schedule (linear decay):
        ε(t) = ε_start + (ε_end - ε_start) · min(1, t / T)

    Target network (hard sync every K steps):
        θ⁻ ← θ  every K gradient steps

    Training objective:
        y = r + γ(1-done) · max_{a'∈valid(s')} Q_θ⁻(s', a')
        L = Huber(Q_θ(s,a) - y)
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50_000,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update: int = 1000,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

        # Training stats for logging
        self.train_stats = {
            "loss": 0.0,
            "q_mean": 0.0,
            "q_max": 0.0,
            "grad_norm": 0.0,
        }

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.steps_done / self.epsilon_decay)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def select_action(self, state: dict) -> int:
        """Masked ε-greedy: random valid action with prob ε, else argmax Q."""
        mask = state["action_mask"]
        valid = np.where(mask > 0)[0]
        if len(valid) == 0:
            return 0

        if random.random() < self.epsilon:
            return int(valid[random.randint(0, len(valid) - 1)])

        with torch.no_grad():
            b = torch.FloatTensor(state["board"]).unsqueeze(0).to(self.device)
            p = torch.FloatTensor(state["piece_info"]).unsqueeze(0).to(self.device)
            f = torch.FloatTensor(state["board_features"]).unsqueeze(0).to(self.device)
            q = self.q_net(b, p, f).squeeze(0)
            q[torch.FloatTensor(mask).to(self.device) == 0] = -float("inf")
            return q.argmax().item()

    def train_step(self) -> float | None:
        """One gradient step. Returns loss or None if buffer too small."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        boards = batch["board"].to(self.device)
        pieces = batch["piece_info"].to(self.device)
        feats = batch["board_features"].to(self.device)
        next_boards = batch["next_board"].to(self.device)
        next_pieces = batch["next_piece_info"].to(self.device)
        next_feats = batch["next_board_features"].to(self.device)
        next_masks = batch["next_mask"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Q_θ(s, a) for taken actions
        all_q = self.q_net(boards, pieces, feats)
        q_values = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Masked Bellman target
        with torch.no_grad():
            next_q = self.target_net(next_boards, next_pieces, next_feats)
            next_q[next_masks == 0] = -float("inf")
            max_next_q = next_q.max(dim=1)[0]
            max_next_q = torch.clamp(max_next_q, min=0.0)
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Stats
        with torch.no_grad():
            valid_q = all_q[all_q > -1e6]
            self.train_stats["loss"] = loss.item()
            self.train_stats["q_mean"] = valid_q.mean().item() if len(valid_q) else 0.0
            self.train_stats["q_max"] = valid_q.max().item() if len(valid_q) else 0.0
            self.train_stats["grad_norm"] = grad_norm.item()

        return loss.item()
