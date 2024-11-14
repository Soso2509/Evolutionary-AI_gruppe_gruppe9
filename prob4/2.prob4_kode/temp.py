import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from gym.wrappers import RecordVideo

# Initialize environment
env = gym.make('Taxi-v3', render_mode="rgb_array") # , render_mode="human"
env.action_space.seed(42)

# Initialize metrics
average_rewards = []
episode_lengths = []
success_rates = []
rewards = []
memory = {}

# Records a video every 200 episodes
env = RecordVideo(env, video_folder='prob4/3.prob4_output/Heuristic_Policy', episode_trigger=lambda episode_id: episode_id % 1 == 0)

# Hyperparameters
episodes = 10 # Amount of episodes

def heuristic_policy(observation, memory):
    taxi_row, taxi_col, passenger_location, destination = env.unwrapped.decode(observation)

    # Locate both passenger and destination
    locations = [(0,0), (0,4), (4,0), (3,4)]

    # Determine the target based on whether passenger is in the taxi
    if passenger_location == 4:  # Passenger is in the taxi
        target_row, target_col = locations[destination]
        print("Going to final destination")
    else:
        target_row, target_col = locations[passenger_location]

    if taxi_col == target_col and taxi_row == target_row:
        if passenger_location == 4:
            print("At dropoff - attemping dropff")
            return 5
        else:
            print("At pickup - attempting pickup")
            return 4

    # Choose the primary movement direction towards the target
    if taxi_row < target_row:
        return 0  # Move down
    elif taxi_row > target_row:
        return 1  # Move up
    elif taxi_col < target_col:
        return 2  # Move right
    elif taxi_col > target_col:
        return 3  # Move left
    
    # Check if we're repeating states to prevent getting stuck
    current_state = (taxi_row, taxi_col, passenger_location)
    if memory.get("last_state") == current_state:
        action = random.randint(0, 3)
        print("random")
    # Update memory with current state
    memory["last_state"] = current_state

    return action

# Training loop
for episode in range(episodes):
    observation, info = env.reset() # Make sure environment is reset before every run
    memory["last_state"] = None

    # Initializing variables
    total_reward = 0
    steps = 0
    success = 0
    done = False

     # Looping through the episodes
    while not done:
        action = heuristic_policy(observation, memory) # Choose the action based on the heuristic policy
        next_observation, reward, terminated, truncated, info = env.step(action)

        # For metrics
        steps += 1 
        total_reward += reward
        if terminated:
            success = 1
        done = terminated or truncated

    # Tracking metrics
    rewards.append(total_reward)
    episode_lengths.append(steps)
    success_rates.append(success)

    if episode % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        avg_length = np.mean(episode_lengths[-100:])
        avg_success_rate = np.mean(success_rates[-100:]) * 100

        average_rewards.append(avg_reward)
        print(f'Episode {episode}, Average Reward (last 100): {avg_reward}, Average Success Rate: {avg_success_rate}%')
        
print(f'Average reward: {avg_reward}')
print(f'Average length: {avg_length}')
print(f'Average success rate: {avg_success_rate}%')
        
# Plotting the total reward
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes')
plt.show()

# Plotting the average reward
plt.plot(average_rewards)
plt.xlabel('Episodes (x100)')
plt.ylabel('Average Reward')
plt.title('Average Reward of Time')
plt.show()

env.close()