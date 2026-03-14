"""
Training loop for Tetris DQN.

Usage:
    python train.py

Logs two lines per interval:
  Line 1: Episode metrics (reward, lines, pieces, score)
  Line 2: Training metrics (loss, Q-values, gradients, board quality)

Also writes JSONL stats for plotting.
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
    env = TetrisEnv()
    agent = DQNAgent()

    param_count = sum(p.numel() for p in agent.q_net.parameters())
    print("=" * 70)
    print("Tetris DQN Training")
    print("=" * 70)
    print(f"  Device:          {agent.device}")
    print(f"  Parameters:      {param_count:,}")
    print(f"  Batch size:      {agent.batch_size}")
    print(f"  LR:              {agent.optimizer.param_groups[0]['lr']}")
    print(f"  Gamma:           {agent.gamma}")
    print(f"  Epsilon:         {agent.epsilon_start} → {agent.epsilon_end} "
          f"over {agent.epsilon_decay:,} steps")
    print(f"  Target update:   every {agent.target_update} steps")
    print(f"  Buffer:          {agent.replay_buffer.buffer.maxlen:,}")
    print("=" * 70)
    print()

    # Rolling stats
    w = log_interval
    ep_rewards = deque(maxlen=w)
    ep_lengths = deque(maxlen=w)
    ep_lines = deque(maxlen=w)
    ep_scores = deque(maxlen=w)
    ep_holes = deque(maxlen=w)
    ep_heights = deque(maxlen=w)
    recent_losses = deque(maxlen=1000)
    recent_q_means = deque(maxlen=1000)
    recent_q_maxs = deque(maxlen=1000)
    recent_grads = deque(maxlen=1000)

    total_steps = 0
    best_avg_lines = -float("inf")
    start_time = time.time()

    stats_path = Path(stats_file)
    stats_path.write_text("")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                recent_losses.append(loss)
                recent_q_means.append(agent.train_stats["q_mean"])
                recent_q_maxs.append(agent.train_stats["q_max"])
                recent_grads.append(agent.train_stats["grad_norm"])

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_steps)
        ep_lines.append(info["lines_cleared"])
        ep_scores.append(info["score"])
        ep_holes.append(info["holes"])
        ep_heights.append(info["max_height"])

        # ── Logging ───────────────────────────────────────────────────
        if episode % log_interval == 0:
            elapsed = time.time() - start_time
            avg_r = np.mean(ep_rewards)
            avg_l = np.mean(ep_lines)
            avg_p = np.mean(ep_lengths)
            avg_s = np.mean(ep_scores)
            avg_h = np.mean(ep_holes)
            avg_ht = np.mean(ep_heights)
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            avg_q = np.mean(recent_q_means) if recent_q_means else 0
            max_q = np.mean(recent_q_maxs) if recent_q_maxs else 0
            avg_g = np.mean(recent_grads) if recent_grads else 0

            print(
                f"Ep {episode:>6d} | "
                f"ε {agent.epsilon:.3f} | "
                f"R {avg_r:>7.2f} | "
                f"Lines {avg_l:>5.1f} | "
                f"Pieces {avg_p:>5.0f} | "
                f"Score {avg_s:>7.0f} | "
                f"{elapsed:>5.0f}s"
            )
            print(
                f"{'':>10s} "
                f"Loss {avg_loss:.4f} | "
                f"Q̄ {avg_q:>7.3f} | "
                f"Q↑ {max_q:>7.3f} | "
                f"‖∇‖ {avg_g:>6.3f} | "
                f"Holes {avg_h:>4.1f} | "
                f"H {avg_ht:>4.1f} | "
                f"Buf {len(agent.replay_buffer):>6d}"
            )

            stats = {
                "episode": episode,
                "steps": total_steps,
                "time": round(elapsed, 1),
                "epsilon": round(agent.epsilon, 4),
                "reward": round(float(avg_r), 3),
                "lines": round(float(avg_l), 2),
                "pieces": round(float(avg_p), 1),
                "score": round(float(avg_s), 1),
                "loss": round(float(avg_loss), 5),
                "q_mean": round(float(avg_q), 4),
                "q_max": round(float(max_q), 4),
                "grad_norm": round(float(avg_g), 4),
                "holes": round(float(avg_h), 2),
                "height": round(float(avg_ht), 1),
                "buffer": len(agent.replay_buffer),
            }
            with open(stats_path, "a") as f:
                f.write(json.dumps(stats) + "\n")

            if avg_l > best_avg_lines:
                best_avg_lines = avg_l
                torch.save(agent.q_net.state_dict(), "best_model.pt")

        # ── Render ────────────────────────────────────────────────────
        if render_interval > 0 and episode % render_interval == 0:
            el, es = _greedy_eval(env, agent)
            print(f"\n--- Greedy: {el} lines, score {es} ---")
            print(env.render())
            print()

        # ── Checkpoint ────────────────────────────────────────────────
        if episode % save_interval == 0:
            torch.save({
                "episode": episode,
                "total_steps": total_steps,
                "steps_done": agent.steps_done,
                "q_net": agent.q_net.state_dict(),
                "target_net": agent.target_net.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
            }, f"checkpoint_{episode}.pt")

    torch.save(agent.q_net.state_dict(), "final_model.pt")
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s. Best avg lines: {best_avg_lines:.2f}")
    print(f"Stats: {stats_path}")


def _greedy_eval(env, agent):
    state = env.reset()
    saved = (agent.epsilon_start, agent.epsilon_end)
    agent.epsilon_start = agent.epsilon_end = 0.0
    info = {}
    while not env.done:
        action = agent.select_action(state)
        state, _, _, info = env.step(action)
    agent.epsilon_start, agent.epsilon_end = saved
    return info.get("lines_cleared", 0), info.get("score", 0)


if __name__ == "__main__":
    train()
