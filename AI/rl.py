import gymnasium as gym

# Create the environment
# map_name: "4x4" or "8x8"
# is_slippery: If True, the agent might move in a different direction than intended
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

observation, info = env.reset()

for _ in range(20):
    # Sample a random action (0: Left, 1: Down, 2: Right, 3: Up)
    action = env.action_space.sample() 
    
    # Apply the action
    observation, reward, terminated, truncated, info = env.step(action)

    # If the agent falls in a hole or reaches the goal, reset
    if terminated or truncated:
        observation, info = env.reset()

env.close()

import numpy as np

# 1. Setup Environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n

# 2. Initialize Q-table with zeros
q_table = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.8
gamma = 0.95           # Discount factor
epsilon = 1.0          # Exploration rate
epsilon_decay = 0.001
episodes = 2000

# 3. Training Loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    
    for _ in range(100): # Limit steps per episode
        # Exploration vs Exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)

        # Update Q-table using the Bellman Equation
        # Q(s,a) = Q(s,a) + lr * [R + gamma * max(Q(s',a')) - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state
        if terminated or truncated:
            break
    
    # Reduce epsilon to explore less over time
    epsilon = max(0.01, epsilon - epsilon_decay)

print("Training finished. Q-Table learned!")

# 4. Watch the Trained Agent
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
state, _ = env.reset()
for _ in range(10):
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()