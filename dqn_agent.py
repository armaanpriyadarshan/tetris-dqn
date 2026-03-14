"""
DQN Agent for Tetris

Architecture:
  - Conv layers process the 2-channel 20x10 grid (board + active piece)
  - One-hot next piece is concatenated after conv layers
  - Fully connected layers output Q(s, a) for all 6 actions

Key components:
  - Q-Network (online) and Target Network (frozen copy)
  - Experience Replay Buffer
  - Epsilon-greedy exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tetris_env import NUM_ACTIONS, NUM_PIECES, ROWS, COLS


class QNetwork(nn.Module):
    """
    Maps state -> Q-values for all actions.

    Why a CNN?
    ----------
    The board is a spatial grid. Patterns like "this column has a gap"
    or "this surface is flat" are local spatial features — exactly what
    convolutions are designed to detect. A fully connected net would
    need to independently learn these patterns for every board position.
    A CNN learns them once and applies them everywhere (weight sharing).

    Architecture choices:
    - Small kernels (3x3) — Tetris features are local
    - Few layers — the board is only 20x10, not a 224x224 image
    - Concatenate next_piece after convolutions — it's not spatial info
    """

    def __init__(self):
        super().__init__()

        # Convolutional feature extractor for the 20x10 grid
        # Input: (batch, 2, 20, 10) — 2 channels (board + active piece)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # -> (32, 20, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 20, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # -> (64, 20, 10)
            nn.ReLU(),
        )

        # After conv: flatten 64*20*10 = 12800, plus 7 for next piece
        conv_out_size = 64 * ROWS * COLS

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + NUM_PIECES, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_ACTIONS),
        )

    def forward(self, grid, next_piece):
        """
        Args:
            grid: (batch, 2, 20, 10) float tensor
            next_piece: (batch, 7) float tensor (one-hot)

        Returns:
            q_values: (batch, 6) — one Q-value per action
        """
        x = self.conv(grid)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.cat([x, next_piece], dim=1)
        return self.fc(x)


class ReplayBuffer:
    """
    Stores transitions (s, a, r, s', done) and samples random minibatches.

    Why a fixed-size buffer?
    -----------------------
    We want recent-ish experience (the agent's behavior changes over time,
    so very old transitions become less representative), but enough history
    to decorrelate samples. 100k transitions is a common sweet spot.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack state components separately
        grids = torch.FloatTensor(np.array([s["grid"] for s in states]))
        next_pieces = torch.FloatTensor(
            np.array([s["next_piece"] for s in states])
        )
        next_grids = torch.FloatTensor(
            np.array([s["grid"] for s in next_states])
        )
        next_next_pieces = torch.FloatTensor(
            np.array([s["next_piece"] for s in next_states])
        )

        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        return {
            "grid": grids,
            "next_piece": next_pieces,
            "next_grid": next_grids,
            "next_next_piece": next_next_pieces,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    The DQN agent. Brings together:
      - Online Q-network (updated every step)
      - Target Q-network (synced every target_update steps)
      - Replay buffer
      - Epsilon-greedy exploration

    Epsilon-greedy exploration
    -------------------------
    With probability epsilon, take a random action.
    With probability 1-epsilon, take argmax_a Q(s, a).

    Why? This is the exploration-exploitation tradeoff:
    - Exploitation: act on what you know (greedy)
    - Exploration: try random things to discover better strategies

    We anneal epsilon from 1.0 (fully random) to a small value (mostly
    greedy) over training. Early on, the Q-values are garbage, so random
    exploration is more useful. Later, we trust the learned Q-values.

    The math: we want the agent's behavior policy to be
        π(a|s) = (1-ε) * 1[a = argmax Q(s,a)] + ε/|A|
    This guarantees every action has nonzero probability, which is
    needed for convergence guarantees.
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

        # Networks
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        # Optimizer — Adam is standard for DQN. It adapts learning rates
        # per-parameter, which helps because different conv filters and
        # FC weights can need very different step sizes.
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # State
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon, linearly annealed."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - self.steps_done / self.epsilon_decay)

    def select_action(self, state: dict) -> int:
        """
        Epsilon-greedy action selection.

        Returns an action index (0-5).
        """
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        # Greedy: pick action with highest Q-value
        with torch.no_grad():
            grid = torch.FloatTensor(state["grid"]).unsqueeze(0).to(self.device)
            next_piece = torch.FloatTensor(
                state["next_piece"]
            ).unsqueeze(0).to(self.device)
            q_values = self.q_net(grid, next_piece)
            return q_values.argmax(dim=1).item()

    def train_step(self) -> float | None:
        """
        Sample a batch from replay buffer, compute loss, update weights.

        The core DQN update in math:

            target = r + γ * max_a' Q_θ⁻(s', a') * (1 - done)
            loss = (Q_θ(s, a) - target)²

        The (1 - done) term: if the episode ended, there's no future reward.
        The future value is 0, so the target is just r.

        Returns the loss value (for logging) or None if buffer too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # Move everything to device
        grids = batch["grid"].to(self.device)
        next_pieces = batch["next_piece"].to(self.device)
        next_grids = batch["next_grid"].to(self.device)
        next_next_pieces = batch["next_next_piece"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Q_θ(s, a) — get Q-value for the action that was actually taken
        # q_net outputs (batch, 6), we index with the action to get (batch,)
        q_values = self.q_net(grids, next_pieces)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ * max_a' Q_θ⁻(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_grids, next_next_pieces)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Huber loss (smooth L1) — less sensitive to outlier targets than MSE.
        # Early in training, max_next_q can be wildly wrong, producing huge
        # (target - prediction) values. MSE squares these, creating enormous
        # gradients. Huber is quadratic for small errors, linear for large
        # ones, so it's more robust:
        #   L(δ) = 0.5δ²       if |δ| ≤ 1
        #        = |δ| - 0.5    otherwise
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients when the
        # Q-value estimates are wildly off early in training
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Sync target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
