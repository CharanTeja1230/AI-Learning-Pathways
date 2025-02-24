import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearningAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
    
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()
    
    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        
        target = reward_tensor
        if not done:
            target += self.gamma * torch.max(self.model(next_state_tensor)).detach()
        
        prediction = self.model(state_tensor)[action_tensor]
        loss = self.criterion(prediction, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Example usage
if __name__ == "__main__":
    agent = DeepQLearningAgent(state_dim=3, action_dim=5)
    sample_state = np.array([0.5, 25, 0.7])
    action = agent.select_action(sample_state)
    print(f"Selected Action: {action}")
