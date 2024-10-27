import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3', render_mode="rgb_array") # , render_mode="human"
env.action_space.seed(42)

# Hyperparameters
alpha = 0.3 # Learning rate
gamma = 0.99 # Discount factor
epsilon = 0.5 # Exploration

# Q-table init
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))
rewards = []

# Training loop
episodes = 20000

for episode in range(episodes):
    observation, info = env.reset() # Make sure environment is reset before every run
    done = False
    total_reward = 0

    # Initial action
    action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[observation])

    while not done:

        next_observation, reward, terminated, truncated, info = env.step(action)

        next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[next_observation])

        

        # Updating the Q-table (state is observation)
        q_table[observation, action] = q_table[observation, action] + \
            alpha * (reward + gamma * q_table[next_observation, next_action] - q_table[observation, action])
        
        # Update observation
        observation, action = next_observation, next_action

        total_reward += reward
        

        epsilon = max(0.1, epsilon * 0.995)

        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')

        done = terminated or truncated
    
    rewards.append(total_reward)

plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes')
plt.show()


env.close()