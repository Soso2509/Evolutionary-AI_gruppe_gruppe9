import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3', render_mode="rgb_array") # , render_mode="human"
env.action_space.seed(42)

# Hyperparameters
alpha = 0.1 # Learning rate
gamma = 0.99 # Discount factor
epsilon = 1 # Exploration

# Q-table init
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))
rewards = []

# Training loop
episodes = 4000

for episode in range(episodes):
    observation, info = env.reset() # Make sure environment is reset before every run
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample() # Take a random action aka explore
        else:
            action = np.argmax(q_table[observation]) # Take the best action aka exploit

        next_observation, reward, terminated, truncated, info = env.step(action)

        # Updating the Q-table (state is observation)
        best_next_action = np.max(q_table[next_observation])
        q_table[observation, action] = q_table[observation, action] + \
            alpha * (reward + gamma * best_next_action - q_table[observation, action])
        
        # Update observation
        observation = next_observation

        total_reward += reward
        rewards.append(total_reward)

        epsilon = max(0.1, epsilon * 0.995)

        if episode % 1000 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')

        done = terminated or truncated

plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes')
plt.show()


env.close()