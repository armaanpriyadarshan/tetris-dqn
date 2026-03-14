"""
DQN Agent for Tetris — Built from scratch

No Stable-Baselines3, no RLlib. Every component is explicit:

  1. Replay Buffer — uniform random sampling from fixed-capacity deque
  2. Q-Network — CNN backbone (board) + MLP embedding (pieces) → FC head
  3. Epsilon-greedy — linear decay from ε_start to ε_end over N steps
  4. Target Network — frozen copy, hard-synced every K steps
  5. Training — minibatch SGD on Huber loss with gradient clipping

Architecture Theory
-------------------
The Q-network must map (board_state, piece_info) → Q-values for 96 actions.

Why split into CNN + MLP?
  The board is spatial data — local patterns (gaps, flat surfaces, overhangs)
  matter, and they can appear anywhere. CNNs exploit translational equivariance:
  a "gap detector" learned for one position works everywhere (weight sharing).

  Piece identity (current, next, held) is categorical, not spatial. Feeding it
  through convolutions would waste capacity. Instead, we embed it through a
  small MLP and concatenate with the CNN features before the decision head.

Why batch normalization?
  BN normalizes activations to zero mean / unit variance within each minibatch.
  This helps because:
    - Keeps activations in the useful range of ReLU (no dead neurons)
    - Allows higher learning rates (the loss landscape is smoother)
    - Acts as mild regularization (batch statistics add noise)
  In RL, BN is less universally helpful than in supervised learning (because
  the data distribution shifts as the policy changes), but for DQN with a
  replay buffer, the minibatch distribution is relatively stable.

Why strided convolution instead of pooling?
  MaxPool discards spatial information. A strided conv learns WHAT to
  downsample — it's a learnable compression. For a small board (20×10),
  we want to preserve as much spatial detail as possible in the early
  layers, then compress in the last layer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tetris_env import MAX_ACTIONS, NUM_PIECES, VISIBLE_ROWS, COLS


# ── Constants ─────────────────────────────────────────────────────────
# Piece info: current(7) + next(7) + hold(7) + can_hold(1)
PIECE_INFO_SIZE = NUM_PIECES * 3 + 1


# ══════════════════════════════════════════════════════════════════════
#  1. REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-capacity buffer with uniform random sampling.

    Why uniform sampling?
    ---------------------
    Every transition has equal probability of being sampled. This is the
    simplest approach and works well in practice. The alternative —
    Prioritized Experience Replay (PER) — samples transitions with higher
    TD error more often, which can speed up learning but adds complexity
    (importance sampling weights, sum-tree data structure).

    For a one-day project, uniform sampling is the right choice.

    Why a fixed capacity?
    ---------------------
    We want a mix of recent and older experience:
    - Too small: samples are correlated (recent transitions look similar)
    - Too large: stale data from a very different policy dominates
    - 100K is a common sweet spot for discrete-action environments

    Memory layout: each transition is stored as a tuple of numpy arrays.
    At sample time, we batch them into tensors. This is slightly less
    memory-efficient than pre-allocated arrays, but much simpler.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: dict, action: int, reward: float,
             next_state: dict, done: bool):
        """Store a transition (s, a, r, s', done)."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict:
        """
        Sample a random minibatch and stack into tensors.

        Returns dict with keys:
            board, piece_info         — current state components
            next_board, next_piece_info, next_mask — next state components
            actions, rewards, dones   — transition data
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "board": torch.FloatTensor(
                np.array([s["board"] for s in states])),
            "piece_info": torch.FloatTensor(
                np.array([s["piece_info"] for s in states])),
            "next_board": torch.FloatTensor(
                np.array([s["board"] for s in next_states])),
            "next_piece_info": torch.FloatTensor(
                np.array([s["piece_info"] for s in next_states])),
            "next_mask": torch.FloatTensor(
                np.array([s["action_mask"] for s in next_states])),
            "actions": torch.LongTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "dones": torch.FloatTensor(dones),
        }

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════
#  2. Q-NETWORK
# ══════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Q(s, a) approximator: board → CNN → features ← MLP ← piece_info
                                         ↓
                                    FC head → Q-values (96)

    Input dimensions:
        board:      (batch, 1, 20, 10)  — binary grid of locked cells
        piece_info: (batch, 22)         — one-hot pieces + can_hold flag

    Output:
        q_values:   (batch, 96)         — one Q-value per placement action

    Architecture breakdown:
        CNN backbone:
          Conv2d(1→32, 3×3, pad=1)  + BN + ReLU   → (32, 20, 10)
          Conv2d(32→64, 3×3, pad=1) + BN + ReLU   → (64, 20, 10)
          Conv2d(64→64, 3×3, stride=2, pad=1) + BN + ReLU → (64, 10, 5)
          Flatten → 3200

        Piece embedding:
          Linear(22→64) + ReLU → 64

        Decision head:
          Linear(3264→512) + ReLU
          Linear(512→96)

    Why these sizes?
    - 32/64 filters: enough to detect Tetris-relevant features (holes,
      surface roughness, column heights) without overfitting.
    - 3×3 kernels: Tetris features are local (a hole is 1-2 cells).
      Larger kernels would waste parameters.
    - Stride 2 in last conv: reduces 20×10 → 10×5, cutting the FC
      input from 12800 to 3200. Without this, the FC layer dominates
      the parameter count (12800×512 = 6.5M vs 3200×512 = 1.6M).
    - 64-dim piece embedding: piece identity is low-dimensional info
      (7 types + hold state). 64 dims gives enough capacity to learn
      piece-specific strategies without dominating the representation.
    """

    def __init__(self):
        super().__init__()

        # CNN backbone — extracts spatial features from the board
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
        # After stride-2 conv: (64, ceil(20/2), ceil(10/2)) = (64, 10, 5)
        self.conv_out_size = 64 * 10 * 5  # 3200

        # Piece info embedding — learns piece-specific features
        self.piece_embed = nn.Sequential(
            nn.Linear(PIECE_INFO_SIZE, 64),
            nn.ReLU(),
        )

        # Decision head — maps combined features to Q-values
        self.head = nn.Sequential(
            nn.Linear(self.conv_out_size + 64, 512),
            nn.ReLU(),
            nn.Linear(512, MAX_ACTIONS),
        )

    def forward(self, board: torch.Tensor,
                piece_info: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            board:      (batch, 1, 20, 10) float tensor — board state
            piece_info: (batch, 22) float tensor — piece identities + hold

        Returns:
            q_values: (batch, 96) — raw Q-values (mask externally)
        """
        # Spatial features from board
        conv_features = self.conv(board)
        conv_features = conv_features.view(conv_features.size(0), -1)

        # Categorical features from piece info
        piece_features = self.piece_embed(piece_info)

        # Combine and decide
        combined = torch.cat([conv_features, piece_features], dim=1)
        return self.head(combined)


# ══════════════════════════════════════════════════════════════════════
#  3. DQN AGENT
# ══════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Deep Q-Network agent with:
      - Online network (θ) — updated every step via gradient descent
      - Target network (θ⁻) — frozen copy, hard-synced every K steps
      - Replay buffer — uniform sampling of past transitions
      - Epsilon-greedy policy — with action masking for valid placements

    The DQN algorithm (Mnih et al., 2015):
    ────────────────────────────────────────
    Initialize Q_θ with random weights
    Initialize Q_θ⁻ ← Q_θ (copy weights)
    Initialize replay buffer D

    For each step:
        With probability ε: a = random valid action
        Otherwise:          a = argmax_{a ∈ valid(s)} Q_θ(s, a)

        Execute a, observe r, s', done
        Store (s, a, r, s', done) in D

        Sample minibatch {(sᵢ, aᵢ, rᵢ, sᵢ', doneᵢ)} from D
        Compute targets:  yᵢ = rᵢ + γ(1 - doneᵢ) max_{a'∈valid(sᵢ')} Q_θ⁻(sᵢ', a')
        Update θ by minimizing: L = (1/N) Σ Huber(Q_θ(sᵢ, aᵢ) - yᵢ)

        Every K steps: θ⁻ ← θ

    Epsilon schedule:
        ε decays linearly from ε_start to ε_end over ε_decay steps.

        Why linear (not exponential)?
        - Linear gives a predictable, controllable schedule.
        - Exponential decays too fast early on — in Tetris, early random
          exploration is valuable because the agent needs to see diverse
          board states before it can learn anything useful.
        - The decay length (50K steps = ~50K pieces placed) is chosen to
          roughly match when the replay buffer fills up. By then, the agent
          has enough diverse experience to start exploiting.

    Target network update frequency:
        Every 1000 gradient steps, we copy θ → θ⁻.

        Why not update every step (i.e., no target network)?
        - The Bellman target y = r + γ max Q_θ(s', a') depends on θ.
        - If θ changes every step, the target moves with the prediction.
        - This creates a feedback loop: Q goes up → target goes up → Q goes
          up more → divergence.
        - Freezing the target for K steps breaks this feedback loop.

        Why hard updates (copy all weights) instead of soft (Polyak averaging)?
        - Hard updates: θ⁻ ← θ every K steps
        - Soft updates: θ⁻ ← τθ + (1-τ)θ⁻ every step (τ ≈ 0.005)
        - Both work. Hard updates are simpler and standard for DQN.
          Soft updates are more common in actor-critic methods (DDPG, SAC).
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
        """
        Hyperparameter choices explained:

        lr = 1e-4:
            Adam learning rate. Lower than typical supervised learning (1e-3)
            because RL targets are noisy and non-stationary. Too high → the
            network overshoots and Q-values oscillate. Too low → painfully
            slow learning. 1e-4 is the DQN standard.

        gamma = 0.99:
            Discount factor. γ determines how far ahead the agent looks:
              effective horizon ≈ 1/(1-γ) = 100 steps
            In placement-based Tetris, each step = one piece. γ=0.99 means
            the agent cares about ~100 pieces into the future, which is
            reasonable (a typical game lasts 50-200 pieces).

        batch_size = 64:
            Minibatch size for SGD. Tradeoffs:
              - Larger: lower gradient variance, but slower per step
              - Smaller: noisier gradients, but faster iteration
            64 is a sweet spot for discrete-action DQN.

        target_update = 1000:
            Sync target network every 1000 gradient steps.
            ~1 sync per 1000 pieces placed. Faster syncing (100) would
            make the target less stable. Slower (10000) would make learning
            sluggish because the target is stale.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── Networks ──────────────────────────────────────────────────
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net: no dropout/BN training mode

        # ── Optimizer ─────────────────────────────────────────────────
        # Adam adapts per-parameter learning rates using first and second
        # moment estimates of the gradient. This is important because
        # conv filters, embeddings, and FC weights can need very different
        # step sizes. SGD with a single LR would require careful tuning.
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # ── Hyperparameters ───────────────────────────────────────────
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # ── State ─────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0  # gradient steps taken (for target sync)

        # ── Training stats (for logging) ──────────────────────────────
        self.train_stats = {
            "loss": 0.0,
            "q_mean": 0.0,
            "q_max": 0.0,
            "grad_norm": 0.0,
        }

    @property
    def epsilon(self) -> float:
        """
        Current ε, linearly annealed from ε_start to ε_end.

        ε(t) = ε_end + (ε_start - ε_end) * max(0, 1 - t / T)

        where t = steps_done (gradient steps) and T = epsilon_decay.
        """
        progress = min(1.0, self.steps_done / self.epsilon_decay)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def select_action(self, state: dict) -> int:
        """
        Masked ε-greedy action selection.

        With probability ε: sample uniformly from valid actions.
        With probability 1-ε: argmax Q(s, a) over valid actions.

        The masking ensures the agent NEVER picks an invalid placement.
        Invalid actions have their Q-values set to -∞ before the argmax.
        """
        mask = state["action_mask"]
        valid_actions = np.where(mask > 0)[0]

        if len(valid_actions) == 0:
            return 0  # shouldn't happen (game over before this)

        # Explore: random valid action
        if random.random() < self.epsilon:
            return int(valid_actions[random.randint(0, len(valid_actions) - 1)])

        # Exploit: best valid action according to Q-network
        with torch.no_grad():
            board = torch.FloatTensor(
                state["board"]).unsqueeze(0).to(self.device)
            piece_info = torch.FloatTensor(
                state["piece_info"]).unsqueeze(0).to(self.device)
            q_values = self.q_net(board, piece_info).squeeze(0)

            # Mask invalid actions to -inf
            mask_t = torch.FloatTensor(mask).to(self.device)
            q_values[mask_t == 0] = -float("inf")

            return q_values.argmax().item()

    def train_step(self) -> float | None:
        """
        One gradient step of DQN.

        The math, expanded:
        ────────────────────
        Given minibatch {(sᵢ, aᵢ, rᵢ, sᵢ', doneᵢ)}:

        1. Compute current Q-values:
             qᵢ = Q_θ(sᵢ, aᵢ)

        2. Compute targets (using frozen target network):
             yᵢ = rᵢ + γ(1 - doneᵢ) · max_{a' ∈ valid(sᵢ')} Q_θ⁻(sᵢ', a')

           The (1 - done) factor: if the episode ended, there's no future
           value. The target is just the immediate reward r.

           The max over valid actions only: without masking, the max would
           include Q-values for impossible placements, corrupting the target.

        3. Compute Huber loss:
             L = (1/N) Σᵢ Huber(qᵢ - yᵢ)

           Why Huber instead of MSE?
             MSE: L = δ²           → gradient = 2δ (grows linearly with error)
             Huber: L = |δ| - 0.5  → gradient = ±1 (constant for large errors)

           Early in training, targets yᵢ can be wildly wrong (the target net
           is also randomly initialized). MSE would produce huge gradients,
           destabilizing training. Huber clips the gradient magnitude.

        4. Gradient step with clipping:
             θ ← θ - α · clip(∇_θ L, max_norm=10)

           Gradient clipping is a safety net: even with Huber loss, the
           total gradient across all parameters could be large. Clipping
           the global norm ensures the update step is bounded.

        5. Periodically sync target:
             Every K steps: θ⁻ ← θ

        Returns the loss value for logging, or None if buffer too small.
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

        # ── Step 1: Current Q-values ──────────────────────────────────
        # Q_θ(s, a) for the actions that were actually taken.
        # q_net returns (batch, 96), gather selects the taken action → (batch,)
        all_q = self.q_net(boards, piece_infos)
        q_values = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Step 2: Compute targets ───────────────────────────────────
        with torch.no_grad():
            next_q = self.target_net(next_boards, next_piece_infos)
            # Mask invalid next-state actions to -inf
            next_q[next_masks == 0] = -float("inf")
            max_next_q = next_q.max(dim=1)[0]
            # Clamp -inf to 0 for terminal states (all actions invalid)
            max_next_q = torch.clamp(max_next_q, min=0.0)
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # ── Step 3: Huber loss ────────────────────────────────────────
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        # ── Step 4: Gradient step ─────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Step 5: Target sync ───────────────────────────────────────
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # ── Record stats for logging ─────────────────────────────────
        with torch.no_grad():
            valid_q = all_q[all_q > -1e6]  # exclude masked
            self.train_stats["loss"] = loss.item()
            self.train_stats["q_mean"] = valid_q.mean().item() if len(valid_q) > 0 else 0.0
            self.train_stats["q_max"] = valid_q.max().item() if len(valid_q) > 0 else 0.0
            self.train_stats["grad_norm"] = grad_norm.item()

        return loss.item()
