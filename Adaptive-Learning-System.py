import pandas as pd
from adaptive_env import AdaptiveLearningEnv
from dqn_agent import DQNAgent
import numpy as np
import os

# Load dataset safely
data_path = 'data/processed/preprocessed_student_data.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Error: Data file not found at {data_path}")

data = pd.read_csv(data_path)

# Check if 'learning_path' exists
if 'learning_path' not in data.columns:
    raise KeyError("Error: 'learning_path' column is missing from dataset.")

# Ensure 'learning_path' contains valid values
unique_learning_paths = data['learning_path'].dropna().unique()
if len(unique_learning_paths) == 0:
    raise ValueError("Error: No valid learning paths found in dataset.")

# Map unique learning paths to integer actions
action_mapping = {path: i for i, path in enumerate(unique_learning_paths)}
data['learning_path'] = data['learning_path'].map(action_mapping)

# Ensure proper mapping
if data['learning_path'].isna().sum() > 0:
    raise ValueError("Error: Some 'learning_path' values could not be mapped.")

# Model parameters
state_size = len(data.columns) - 1  # Assuming 'learning_path' is the last column
action_size = len(unique_learning_paths)  # Number of unique learning paths
batch_size = 32
episodes = 200  # Increased episodes for better training

# Initialize environment and agent
try:
    env = AdaptiveLearningEnv(data, num_actions=action_size)
    agent = DQNAgent(state_size, action_size)
except Exception as e:
    raise RuntimeError(f"Error initializing environment or agent: {e}")

# Training loop
for e in range(episodes):
    state = env.reset()
    
    if state is None:
        print(f"Skipping episode {e+1}: Environment returned None state.")
        continue

    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        try:
            action = agent.act(state)  # Choose an action
            next_state, reward, done, info = env.step(action)

            if next_state is not None:
                next_state = np.reshape(next_state, [1, state_size])
            else:
                next_state = np.zeros((1, state_size))  # Handle terminal state

            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)  # Train the model

            state = next_state
            total_reward += reward

        except Exception as err:
            print(f"Error during training in episode {e+1}: {err}")
            break  # Skip this episode if an error occurs

    print(f"Episode: {e+1}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon}")

    # Save model checkpoint every 10 episodes
    if e % 10 == 0:
        model_path = f"models/dqn_model_{e}.h5"
        os.makedirs("models", exist_ok=True)
        try:
            agent.save(model_path)
            print(f"Model saved: {model_path}")
        except Exception as save_err:
            print(f"Error saving model at episode {e}: {save_err}")
