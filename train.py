"""
Training loop for Tetris DQN.

Usage:
    python train.py

The training loop follows the standard DQN procedure:
    1. Agent observes state s
    2. Agent picks action a via ε-greedy
    3. Environment returns (s', r, done)
    4. Store transition (s, a, r, s', done) in replay buffer
    5. Sample random minibatch from buffer, compute loss, gradient step
    6. Periodically sync target network ← online network

Why train on random minibatches instead of the latest transition?
--------------------------------------------------------------
Consecutive Tetris frames are highly correlated (the board barely changes
between steps). Training on correlated data violates the i.i.d. assumption
that SGD relies on, leading to unstable learning or divergence. The replay
buffer breaks this correlation by letting us sample uniformly from a large
pool of past experience.

This is one of the two key innovations of DQN (Mnih et al., 2015):
    1. Experience replay (decorrelates training data)
    2. Target network (stabilizes the moving target in the Bellman update)
"""

import time
import torch
import numpy as np
from collections import deque

from tetris_env import TetrisEnv
from dqn_agent import DQNAgent


def train(
    num_episodes: int = 10_000,
    log_interval: int = 50,
    save_interval: int = 500,
    render_interval: int = 0,  # set > 0 to print board every N episodes
):
    env = TetrisEnv()
    agent = DQNAgent()

    print(f"Device: {agent.device}")
    print(f"Network parameters: {sum(p.numel() for p in agent.q_net.parameters()):,}")
    print(f"Epsilon decay over {agent.epsilon_decay} steps")
    print()

    # Rolling stats for logging
    recent_rewards = deque(maxlen=log_interval)
    recent_lengths = deque(maxlen=log_interval)
    recent_lines = deque(maxlen=log_interval)
    recent_losses = deque(maxlen=1000)

    total_steps = 0
    best_avg_reward = -float("inf")
    start_time = time.time()

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

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        recent_rewards.append(episode_reward)
        recent_lengths.append(episode_steps)
        recent_lines.append(info["lines_cleared"])

        # ── Logging ───────────────────────────────────────────────────
        if episode % log_interval == 0:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            avg_lines = np.mean(recent_lines)
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            elapsed = time.time() - start_time

            print(
                f"Episode {episode:>6d} | "
                f"Steps {total_steps:>8d} | "
                f"ε {agent.epsilon:.3f} | "
                f"Reward {avg_reward:>7.2f} | "
                f"Lines {avg_lines:>5.1f} | "
                f"Length {avg_length:>6.1f} | "
                f"Loss {avg_loss:.4f} | "
                f"Time {elapsed:>6.0f}s"
            )

            # Track improvement
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.q_net.state_dict(), "best_model.pt")

        # ── Render ────────────────────────────────────────────────────
        if render_interval > 0 and episode % render_interval == 0:
            # Play one episode greedily (no exploration) and show final board
            test_state = env.reset()
            old_eps = agent.epsilon_end
            agent.epsilon_end = 0.0  # force greedy
            while not env.done:
                a = agent.select_action(test_state)
                test_state, _, _, test_info = env.step(a)
            agent.epsilon_end = old_eps
            print(f"\n--- Greedy eval: {test_info['lines_cleared']} lines ---")
            print(env.render())
            print()

        # ── Checkpoint ────────────────────────────────────────────────
        if episode % save_interval == 0:
            torch.save({
                "episode": episode,
                "total_steps": total_steps,
                "q_net_state": agent.q_net.state_dict(),
                "target_net_state": agent.target_net.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "epsilon_decay": agent.epsilon_decay,
                "steps_done": agent.steps_done,
            }, f"checkpoint_{episode}.pt")

    # Final save
    torch.save(agent.q_net.state_dict(), "final_model.pt")
    print(f"\nTraining complete. Best avg reward: {best_avg_reward:.2f}")


if __name__ == "__main__":
    train()
