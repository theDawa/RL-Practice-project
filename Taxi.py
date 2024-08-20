import numpy as np
import gymnasium as gym
import random

class bcolors:
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    RESET = '\u001b[0m'

# Create Taxi environment
env = gym.make('Taxi-v3', render_mode="human")

# Initialize Q-table
state_size = env.observation_space.n
action_size = env.action_space.n
qtable = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate = 0.005

# Training variables
num_episodes = 2000
max_steps = 99  # per episode

print("AGENT IS TRAINING...")

for episode in range(num_episodes):
    # Reset the environment
    state, info = env.reset()
    done = False

    for step in range(max_steps):
        # Exploration-exploitation tradeoff
        if random.uniform(0, 1) < epsilon:
            # Explore
            action = env.action_space.sample()
        else:
            # Exploit
            action = np.argmax(qtable[state, :])

        # Take an action and observe the reward
        new_state, reward, done, info, _ = env.step(action)

        # Q-learning algorithm
        qtable[state, action] = qtable[state, action] + learning_rate * (
            reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
        )

        # Update to our new state
        state = new_state

        # If done, finish episode
        if done:
            break

    # Decrease epsilon (exploration rate)
    epsilon = np.exp(-decay_rate * episode)

# Display Q-table
print(f"Our Q-table: {qtable}")
print(f"Training completed over {num_episodes} episodes")
input("Press Enter to see our trained taxi agent")

# Get ready to watch our trained agent
episodes_to_preview = 3
num_steps = 99
for episode in range(episodes_to_preview):
    # Reset the environment
    state, info = env.reset()
    done = False
    episode_rewards = 0

    for step in range(num_steps):
        print(f"TRAINED AGENT")
        print(f"+++++EPISODE {episode + 1}+++++")
        print(f"Step {step + 1}")

        # Exploit
        action = np.argmax(qtable[state, :])

        # Take an action and observe the reward
        new_state, reward, done, info, _ = env.step(action)

        # Accumulate our rewards
        episode_rewards += reward

        env.render()

        if episode_rewards < 0:
            print(f"Score: {bcolors.RED}{episode_rewards}{bcolors.RESET}")
        else:
            print(f"Score: {bcolors.GREEN}{episode_rewards}{bcolors.RESET}")

        # Update to our new state
        state = new_state

        # If done, finish episode
        if done:
            break

# Close the Taxi environment
env.close()
