from argparse import Action
import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.preprocessing import LabelEncoder

class AdaptiveLearningEnv(gym.Env):
    def __init__(self, data, num_actions=3):
        super(AdaptiveLearningEnv, self).__init__()
        self.data = data
        self.num_actions = num_actions  # Number of possible learning paths
        self.state_size = len(data.columns) - 1  # Assuming last column is reward/target
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.current_step = 0

        # Encode categorical columns
        self.categorical_columns = ['LearningStyle', 'PreferredContent']
        self.label_encoders = {col: LabelEncoder().fit(self.data[col]) for col in self.categorical_columns}
        self.data[self.categorical_columns] = self.data[self.categorical_columns].apply(lambda col: self.label_encoders[col.name].transform(col))

    def reset(self):
        self.current_step = 0
        state = self._next_observation()
        return state

    def _next_observation(self):
        if self.current_step < len(self.data):
            obs = self.data.iloc[self.current_step, :-1].values.astype(np.float32)
            return obs
        else:
            return None

    def step(self, action):
        if self.current_step < len(self.data):
            reward = self._get_reward(action)
            self.current_step += 1
            next_state = self._next_observation()
            done = next_state is None
            info = {"action_taken": action, "step": self.current_step}
            return next_state, reward, done, info
        else:
            return None, 0, True, {}

    def _get_reward(self, action):
        # Improved reward function using student performance trends
        target = self.data.iloc[self.current_step, -1]  # Actual target value
        performance_trend = np.mean(self.data.iloc[max(0, self.current_step - 5):self.current_step, -1])
        
        if action == target:
            return 1 + performance_trend  # Reward increases if recent performance is good
        else:
            return -1 * (1 - performance_trend)  # Penalize but adjust based on past trend

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State: {self._next_observation()}, Action: {Action}")
