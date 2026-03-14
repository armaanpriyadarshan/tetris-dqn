"""Quick sanity check for the Tetris environment."""

from tetris_env import TetrisEnv, NUM_ACTIONS, ROWS, COLS, NUM_PIECES
import numpy as np

env = TetrisEnv(seed=42)
state = env.reset()

# Check state shapes
assert state["grid"].shape == (2, ROWS, COLS), f"Grid shape: {state['grid'].shape}"
assert state["next_piece"].shape == (NUM_PIECES,), f"Next piece shape: {state['next_piece'].shape}"
assert state["next_piece"].sum() == 1.0, "Next piece should be one-hot"
print("✓ State shapes correct")

# Play a random game
total_reward = 0
steps = 0
while not env.done:
    action = np.random.randint(NUM_ACTIONS)
    state, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1

print(f"✓ Random game completed in {steps} steps")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Lines cleared: {info['lines_cleared']}")
print(f"  Pieces placed: {info['pieces_placed']}")
print(f"\nFinal board:")
print(env.render())

# Play several games to check stability
for i in range(100):
    env.reset()
    while not env.done:
        env.step(np.random.randint(NUM_ACTIONS))
print(f"\n✓ 100 random games completed without errors")
