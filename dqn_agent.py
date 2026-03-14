"""
Double Dueling DQN Agent for Tetris

Improvements over vanilla DQN:

1. DOUBLE DQN (van Hasselt et al., 2016)
   Fixes overestimation bias in the Bellman target.

   Vanilla DQN target:
     y = r + γ · max_a' Q_θ⁻(s', a')

   The max uses θ⁻ for BOTH selecting and evaluating the best action.
   When Q-values are noisy, max systematically picks overestimated values:
     E[max(X₁, X₂, ..., Xₙ)] ≥ max(E[X₁], E[X₂], ..., E[Xₙ])
   This is Jensen's inequality applied to the max function (convex).
   The bias compounds through the Bellman backup chain.

   Double DQN decouples selection from evaluation:
     a* = argmax_a' Q_θ(s', a')        ← online net SELECTS
     y  = r + γ · Q_θ⁻(s', a*)         ← target net EVALUATES

   Since θ and θ⁻ have independent noise (different training checkpoints),
   the overestimation bias largely cancels.

2. DUELING ARCHITECTURE (Wang et al., 2016)
   Decomposes Q(s,a) into value and advantage:

     Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

   V(s):    How good is this board state? (shared across all actions)
   A(s, a): How much better is this placement than average?

   Why this helps for Tetris:
   - Many board states have similar value regardless of action (an empty
     board is good no matter what piece you drop). V captures this.
   - The A stream focuses on what makes specific placements better,
     which is a much narrower learning problem.
   - With 96 actions (many masked), the agent sees each action rarely.
     V generalizes across all actions, so even unseen actions get
     reasonable Q-values through the shared V component.

   The mean subtraction is an identifiability constraint:
     Without it, V and A are not unique — you could add constant c to V
     and subtract c from every A(s,a). The mean subtraction forces
     mean_a A(s,a) = 0, making V the true state value E_π[return|s].

Architecture:
  CNN backbone:  board (1, 20, 10)   → 3 conv layers  → 3200-dim
  Piece embed:   piece_info (22,)    → Linear+ReLU    → 64-dim
  Feature embed: board_features (34,) → Linear+ReLU   → 64-dim

  Combined (3328) splits into:
    Value stream:     3328 → 256 → 1    (state value)
    Advantage stream: 3328 → 256 → 96   (per-action advantage)

  Q = V + A - mean(A)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tetris_env import MAX_ACTIONS, NUM_PIECES, BOARD_FEATURES_SIZE


PIECE_INFO_SIZE = NUM_PIECES * 3 + 1  # 22


# ══════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-capacity buffer with uniform random sampling."""

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
#  DUELING Q-NETWORK
# ══════════════════════════════════════════════════════════════════════

class DuelingQNetwork(nn.Module):
    """
    Dueling architecture: shared features → split into V and A streams.

    Shared backbone:
      CNN:   (1, 20, 10) → Conv+BN+ReLU ×3 (last stride=2) → 3200
      Piece: (22,) → Linear+ReLU → 64
      Feats: (34,) → Linear+ReLU → 64
      Combined: 3328

    Dueling split:
      Value stream:     3328 → 256 → ReLU → 1     →  V(s)
      Advantage stream: 3328 → 256 → ReLU → 96    →  A(s, a)

    Output:
      Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

    Why 256 per stream instead of 512 for a single head?
      Total params ≈ same (3328×256×2 ≈ 3328×512). But the split forces
      the network to explicitly separate "how good is this state" from
      "how good is this action," which is a better inductive bias for
      games where many actions have similar value.
    """

    def __init__(self):
        super().__init__()

        # ── Shared backbone ───────────────────────────────────────────
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
        self._conv_out = 64 * 10 * 5  # 3200

        self.piece_embed = nn.Sequential(
            nn.Linear(PIECE_INFO_SIZE, 64),
            nn.ReLU(),
        )

        self.feat_embed = nn.Sequential(
            nn.Linear(BOARD_FEATURES_SIZE, 64),
            nn.ReLU(),
        )

        combined_size = self._conv_out + 64 + 64  # 3328

        # ── Value stream: "how good is this state?" ───────────────────
        # Outputs a single scalar V(s).
        # This stream answers: "given this board, piece, and features,
        # what is the expected return regardless of which action I take?"
        self.value_stream = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # ── Advantage stream: "how much better is this action?" ───────
        # Outputs A(s, a) for each of 96 actions.
        # After mean subtraction, A represents the relative advantage
        # of each placement over the average placement.
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Linear(256, MAX_ACTIONS),
        )

    def forward(self, board, piece_info, board_features):
        """
        Args:
            board:          (batch, 1, 20, 10)
            piece_info:     (batch, 22)
            board_features: (batch, 34)
        Returns:
            q_values:       (batch, 96) = V + A - mean(A)
        """
        # Shared feature extraction
        x = self.conv(board).view(board.size(0), -1)
        p = self.piece_embed(piece_info)
        f = self.feat_embed(board_features)
        combined = torch.cat([x, p, f], dim=1)

        # Dueling streams
        v = self.value_stream(combined)            # (batch, 1)
        a = self.advantage_stream(combined)        # (batch, 96)

        # Q = V + (A - mean(A))
        # The mean subtraction ensures A is centered at 0.
        # This makes V ≈ E[Q(s, ·)] and A ≈ Q(s, a) - E[Q(s, ·)]
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


# ══════════════════════════════════════════════════════════════════════
#  DOUBLE DUELING DQN AGENT
# ══════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Double Dueling DQN with replay buffer and masked ε-greedy.

    The two improvements work synergistically:
    - Dueling: better Q-value estimates (especially for rarely-seen actions)
    - Double: unbiased targets from those better estimates

    Training objective (Double DQN):
        a* = argmax_{a'∈valid(s')} Q_θ(s', a')      ← online net selects
        y  = r + γ(1-done) · Q_θ⁻(s', a*)           ← target net evaluates
        L  = Huber(Q_θ(s,a) - y)

    Compare to vanilla DQN:
        y  = r + γ(1-done) · max_{a'∈valid(s')} Q_θ⁻(s', a')

    The difference is subtle but important:
    - Vanilla: target net both picks AND scores the best next action
    - Double: online net picks, target net scores

    This reduces overestimation because the selection noise (from θ)
    and evaluation noise (from θ⁻) are independent.
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

        # Both networks use the Dueling architecture
        self.q_net = DuelingQNetwork().to(self.device)
        self.target_net = DuelingQNetwork().to(self.device)
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
        """
        One Double DQN gradient step.

        Key difference from vanilla DQN (marked with ★):

        Vanilla:
          next_q = Q_θ⁻(s', ·)
          max_next_q = max_a' next_q[a']

        Double:
          ★ online_next_q = Q_θ(s', ·)              ← online net
          ★ best_actions = argmax_a' online_next_q   ← online SELECTS
          ★ next_q = Q_θ⁻(s', ·)                    ← target net
          ★ max_next_q = next_q[best_actions]        ← target EVALUATES

        Everything else (loss, optimization, target sync) is identical.
        """
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

        # ── Current Q-values ──────────────────────────────────────────
        all_q = self.q_net(boards, pieces, feats)
        q_values = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Double DQN target ─────────────────────────────────────────
        with torch.no_grad():
            # ★ Step 1: Online network SELECTS the best next action
            # We use the online net (not the target) to choose which
            # action is best. This breaks the coupling between selection
            # and evaluation that causes overestimation.
            online_next_q = self.q_net(next_boards, next_pieces, next_feats)
            online_next_q[next_masks == 0] = -float("inf")
            best_next_actions = online_next_q.argmax(dim=1)  # (batch,)

            # ★ Step 2: Target network EVALUATES that action
            # The target net provides the Q-value for the action the
            # online net selected. Since θ and θ⁻ have independent noise,
            # the overestimation bias is greatly reduced.
            target_next_q = self.target_net(
                next_boards, next_pieces, next_feats)
            max_next_q = target_next_q.gather(
                1, best_next_actions.unsqueeze(1)).squeeze(1)

            # For terminal states (all actions masked), clamp to 0
            max_next_q = torch.clamp(max_next_q, min=0.0)

            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # ── Loss and optimization ─────────────────────────────────────
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # ── Stats ─────────────────────────────────────────────────────
        with torch.no_grad():
            valid_q = all_q[all_q > -1e6]
            self.train_stats["loss"] = loss.item()
            self.train_stats["q_mean"] = valid_q.mean().item() if len(valid_q) else 0.0
            self.train_stats["q_max"] = valid_q.max().item() if len(valid_q) else 0.0
            self.train_stats["grad_norm"] = grad_norm.item()

        return loss.item()
