"""
DQN Agent for Tetris — Placement-based action space

Architecture:
  - Conv layers process the 1-channel 20x10 board
  - Piece info (22 dims) concatenated after conv layers
  - FC layers output Q(s, a) for all 96 possible placements
  - Invalid actions are masked to -inf before argmax

Key components:
  - Q-Network (online) and Target Network (frozen copy)
  - Experience Replay Buffer
  - Epsilon-greedy exploration with action masking

Action masking:
  Not all 96 placement actions are valid in every state. Invalid actions
  (piece doesn't fit, can't hold, etc.) are masked to -inf in Q-values.
  This ensures the agent never picks an impossible action, and the masked
  entries don't contribute to the max in the Bellman target.

  Mathematically, the masked Bellman target is:
      y = r + γ * max_{a' ∈ valid(s')} Q_θ⁻(s', a')
  instead of max over all actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tetris_env import MAX_ACTIONS, NUM_PIECES, VISIBLE_ROWS, COLS


# Piece info size: current(7) + next(7) + hold(7) + can_hold(1) = 22
PIECE_INFO_SIZE = NUM_PIECES * 3 + 1


class QNetwork(nn.Module):
    """
    Maps state -> Q-values for all 96 placement actions.

    Why a CNN?
    ----------
    The board is a spatial grid. Patterns like "this column has a gap"
    or "this surface is flat" are local spatial features — exactly what
    convolutions are designed to detect. A fully connected net would
    need to independently learn these patterns for every board position.
    A CNN learns them once and applies them everywhere (weight sharing).

    Architecture choices:
    - 1 input channel (just the board — no active piece in placement-based)
    - Small kernels (3x3) — Tetris features are local
    - Few layers — the board is only 20x10, not a 224x224 image
    - Piece info concatenated after conv — it's not spatial data
    """

    def __init__(self):
        super().__init__()

        # Convolutional feature extractor for the 20x10 grid
        # Input: (batch, 1, 20, 10) — 1 channel (board state)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # -> (32, 20, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 20, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # -> (64, 20, 10)
            nn.ReLU(),
        )

        conv_out_size = 64 * VISIBLE_ROWS * COLS  # 12800

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + PIECE_INFO_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, MAX_ACTIONS),
        )

    def forward(self, board, piece_info):
        """
        Args:
            board: (batch, 1, 20, 10) float tensor
            piece_info: (batch, 22) float tensor

        Returns:
            q_values: (batch, 96) — one Q-value per possible placement
        """
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, piece_info], dim=1)
        return self.fc(x)


class ReplayBuffer:
    """
    Stores transitions (s, a, r, s', done) and samples random minibatches.

    Why experience replay?
    ----------------------
    Consecutive placements are correlated (similar board states). SGD
    assumes i.i.d. samples — violating this causes unstable learning.
    The replay buffer decorrelates training data by sampling uniformly
    from a large pool of past transitions.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        boards = torch.FloatTensor(np.array([s["board"] for s in states]))
        piece_infos = torch.FloatTensor(
            np.array([s["piece_info"] for s in states])
        )
        next_boards = torch.FloatTensor(
            np.array([s["board"] for s in next_states])
        )
        next_piece_infos = torch.FloatTensor(
            np.array([s["piece_info"] for s in next_states])
        )
        next_masks = torch.FloatTensor(
            np.array([s["action_mask"] for s in next_states])
        )

        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        return {
            "board": boards,
            "piece_info": piece_infos,
            "next_board": next_boards,
            "next_piece_info": next_piece_infos,
            "next_mask": next_masks,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with action masking for placement-based Tetris.

    Epsilon-greedy exploration with masking
    ----------------------------------------
    With probability ε, pick a random VALID action (not any random action).
    With probability 1-ε, pick argmax Q(s, a) over valid actions only.

    The math: the behavior policy is
        π(a|s) = (1-ε) * 1[a = argmax_{a'∈valid} Q(s,a')] + ε/|valid(s)|
    where valid(s) is the set of valid actions in state s.

    This still guarantees every valid action has nonzero probability,
    preserving the exploration guarantees needed for convergence.
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
        self.target_net.eval()

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
        Epsilon-greedy action selection over valid actions only.

        Returns a valid action index (0-95).
        """
        mask = state["action_mask"]
        valid_actions = np.where(mask > 0)[0]
        assert len(valid_actions) > 0, "No valid actions available"

        if random.random() < self.epsilon:
            return int(self.rng_choice(valid_actions))

        # Greedy: pick valid action with highest Q-value
        with torch.no_grad():
            board = torch.FloatTensor(state["board"]).unsqueeze(0).to(self.device)
            piece_info = torch.FloatTensor(
                state["piece_info"]
            ).unsqueeze(0).to(self.device)
            q_values = self.q_net(board, piece_info).squeeze(0)

            # Mask invalid actions to -inf
            mask_tensor = torch.FloatTensor(mask).to(self.device)
            q_values[mask_tensor == 0] = -float("inf")

            return q_values.argmax().item()

    def rng_choice(self, arr):
        """Pick a random element from arr."""
        return arr[random.randint(0, len(arr) - 1)]

    def train_step(self) -> float | None:
        """
        Sample a batch, compute masked Bellman target, update weights.

        The masked DQN update:
            target = r + γ * max_{a' ∈ valid(s')} Q_θ⁻(s', a') * (1 - done)
            loss = HuberLoss(Q_θ(s, a), target)

        Action masking in the target is critical: without it, the max
        would include Q-values for impossible placements, corrupting
        the target signal.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # Move to device
        boards = batch["board"].to(self.device)
        piece_infos = batch["piece_info"].to(self.device)
        next_boards = batch["next_board"].to(self.device)
        next_piece_infos = batch["next_piece_info"].to(self.device)
        next_masks = batch["next_mask"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Q_θ(s, a) for the action that was taken
        q_values = self.q_net(boards, piece_infos)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Masked target: r + γ * max_{valid} Q_θ⁻(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_boards, next_piece_infos)
            # Mask invalid actions in next state to -inf
            next_q_values[next_masks == 0] = -float("inf")
            max_next_q = next_q_values.max(dim=1)[0]
            # If ALL actions are invalid (terminal state), max_next_q = -inf
            # Clamp to 0 for terminal states
            max_next_q = torch.clamp(max_next_q, min=-1e6)
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Huber loss — robust to outlier targets early in training
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
