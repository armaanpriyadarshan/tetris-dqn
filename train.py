"""
Training loop for Tetris DQN.

Usage:
    python train.py

Training loop structure:
    for each episode:
        s = env.reset()
        while not done:
            a = agent.select_action(s)       # masked ε-greedy
            s', r, done, info = env.step(a)
            buffer.push(s, a, r, s', done)
            loss = agent.train_step()         # sample batch, gradient step
            s = s'
        log metrics

With placement-based actions, each step places one piece. An episode = one
full game (typically 30-200 pieces before game over). This makes episodes
much shorter than step-by-step control, so we need more episodes but each
is more informative.

Logging philosophy:
    Good RL logging tells you:
    1. Is the agent exploring? (epsilon, buffer size)
    2. Is it learning? (loss decreasing, Q-values growing)
    3. Is it improving? (reward, lines cleared trending up)
    4. Is training stable? (gradient norm, Q-value range bounded)

    We log rolling averages because individual episodes are noisy
    (Tetris has inherent randomness from piece sequence).
"""

import time
import json
import torch
import numpy as np
from collections import deque
from pathlib import Path

from tetris_env import TetrisEnv
from dqn_agent import DQNAgent


def train(
    num_episodes: int = 10_000,
    log_interval: int = 50,
    save_interval: int = 500,
    render_interval: int = 0,
    stats_file: str = "train_stats.jsonl",
):
    """
    Main training loop.

    Args:
        num_episodes: Total episodes to train.
        log_interval: Print metrics every N episodes.
        save_interval: Save checkpoint every N episodes.
        render_interval: Render a greedy eval game every N episodes (0 = off).
        stats_file: Path to write per-interval stats as JSONL (for plotting).
    """
    env = TetrisEnv()
    agent = DQNAgent()

    # ── Print config ──────────────────────────────────────────────────
    param_count = sum(p.numel() for p in agent.q_net.parameters())
    print("=" * 65)
    print("Tetris DQN Training")
    print("=" * 65)
    print(f"  Device:            {agent.device}")
    print(f"  Parameters:        {param_count:,}")
    print(f"  Batch size:        {agent.batch_size}")
    print(f"  Learning rate:     {agent.optimizer.param_groups[0]['lr']}")
    print(f"  Gamma:             {agent.gamma}")
    print(f"  Epsilon:           {agent.epsilon_start} → {agent.epsilon_end} "
          f"over {agent.epsilon_decay:,} steps")
    print(f"  Target update:     every {agent.target_update} steps")
    print(f"  Buffer capacity:   {agent.replay_buffer.buffer.maxlen:,}")
    print(f"  Episodes:          {num_episodes:,}")
    print("=" * 65)
    print()

    # ── Rolling stats ─────────────────────────────────────────────────
    window = log_interval
    ep_rewards = deque(maxlen=window)
    ep_lengths = deque(maxlen=window)   # pieces placed per episode
    ep_lines = deque(maxlen=window)
    ep_scores = deque(maxlen=window)
    recent_losses = deque(maxlen=1000)
    recent_q_means = deque(maxlen=1000)
    recent_q_maxs = deque(maxlen=1000)
    recent_grad_norms = deque(maxlen=1000)

    total_steps = 0
    best_avg_lines = -float("inf")
    start_time = time.time()

    # Clear stats file
    stats_path = Path(stats_file)
    stats_path.write_text("")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while not env.done:
            # ── Act ───────────────────────────────────────────────────
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # ── Store ─────────────────────────────────────────────────
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # ── Learn ─────────────────────────────────────────────────
            loss = agent.train_step()
            if loss is not None:
                recent_losses.append(loss)
                recent_q_means.append(agent.train_stats["q_mean"])
                recent_q_maxs.append(agent.train_stats["q_max"])
                recent_grad_norms.append(agent.train_stats["grad_norm"])

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        # Record episode stats
        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_steps)
        ep_lines.append(info["lines_cleared"])
        ep_scores.append(info["score"])

        # ── Periodic Logging ──────────────────────────────────────────
        if episode % log_interval == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(ep_rewards)
            avg_lines = np.mean(ep_lines)
            avg_length = np.mean(ep_lengths)
            avg_score = np.mean(ep_scores)
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            avg_q = np.mean(recent_q_means) if recent_q_means else 0.0
            max_q = np.mean(recent_q_maxs) if recent_q_maxs else 0.0
            avg_grad = np.mean(recent_grad_norms) if recent_grad_norms else 0.0

            # Console output: two lines per interval for readability
            print(
                f"Ep {episode:>6d} | "
                f"ε {agent.epsilon:.3f} | "
                f"Reward {avg_reward:>7.2f} | "
                f"Lines {avg_lines:>5.1f} | "
                f"Pieces {avg_length:>5.1f} | "
                f"Score {avg_score:>7.0f} | "
                f"{elapsed:>5.0f}s"
            )
            print(
                f"{'':>10s} "
                f"Loss {avg_loss:.4f} | "
                f"Q̄ {avg_q:>7.3f} | "
                f"Q_max {max_q:>7.3f} | "
                f"‖∇‖ {avg_grad:>6.3f} | "
                f"Buf {len(agent.replay_buffer):>6d} | "
                f"Steps {total_steps:>8d}"
            )

            # Save to JSONL for plotting
            stats = {
                "episode": episode,
                "total_steps": total_steps,
                "elapsed_s": round(elapsed, 1),
                "epsilon": round(agent.epsilon, 4),
                "avg_reward": round(avg_reward, 3),
                "avg_lines": round(avg_lines, 2),
                "avg_pieces": round(avg_length, 1),
                "avg_score": round(avg_score, 1),
                "avg_loss": round(avg_loss, 5),
                "avg_q": round(avg_q, 4),
                "max_q": round(max_q, 4),
                "avg_grad_norm": round(avg_grad, 4),
                "buffer_size": len(agent.replay_buffer),
            }
            with open(stats_path, "a") as f:
                f.write(json.dumps(stats) + "\n")

            # Track best model by lines cleared (the real metric)
            if avg_lines > best_avg_lines:
                best_avg_lines = avg_lines
                torch.save(agent.q_net.state_dict(), "best_model.pt")

        # ── Greedy Evaluation Render ──────────────────────────────────
        if render_interval > 0 and episode % render_interval == 0:
            eval_lines, eval_score = _greedy_eval(env, agent)
            print(f"\n--- Greedy eval: {eval_lines} lines, "
                  f"score {eval_score} ---")
            print(env.render())
            print()

        # ── Checkpoint ────────────────────────────────────────────────
        if episode % save_interval == 0:
            torch.save({
                "episode": episode,
                "total_steps": total_steps,
                "steps_done": agent.steps_done,
                "q_net_state": agent.q_net.state_dict(),
                "target_net_state": agent.target_net.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
            }, f"checkpoint_{episode}.pt")

    # ── Final save ────────────────────────────────────────────────────
    torch.save(agent.q_net.state_dict(), "final_model.pt")
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best avg lines cleared: {best_avg_lines:.2f}")
    print(f"Stats saved to: {stats_path}")


def _greedy_eval(env: TetrisEnv, agent: DQNAgent) -> tuple:
    """Play one episode with ε=0 (fully greedy). Returns (lines, score)."""
    state = env.reset()
    # Temporarily force greedy
    saved_start = agent.epsilon_start
    saved_end = agent.epsilon_end
    agent.epsilon_start = 0.0
    agent.epsilon_end = 0.0

    info = {}
    while not env.done:
        action = agent.select_action(state)
        state, _, _, info = env.step(action)

    agent.epsilon_start = saved_start
    agent.epsilon_end = saved_end
    return info.get("lines_cleared", 0), info.get("score", 0)


if __name__ == "__main__":
    train()
