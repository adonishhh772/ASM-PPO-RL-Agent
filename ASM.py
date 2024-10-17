import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Define ASM parameters with power discount factors
ASM_PARAMETERS = {
    0: {'latency': 0, 'power_discount': 1.0},
    1: {'latency': 1, 'power_discount': 0.69},
    2: {'latency': 10, 'power_discount': 0.5},
    3: {'latency': 100, 'power_discount': 0.29}
}

class ASMSleepEnvironment:
    def __init__(self, num_bs):
        self.num_bs = num_bs
        self.state = np.zeros(self.num_bs, dtype=int)  # ASM mode for each BS

    def step(self, actions):
        # Apply actions (ASM level selection for each BS)
        self.state = np.array(actions)
        reward = self._calculate_reward()
        return self.state, reward

    def _calculate_reward(self):
        # Calculate power consumption based on ASM levels
        power_consumption = np.sum([ASM_PARAMETERS[asm]['power_discount'] for asm in self.state])
        # Negative reward since we aim to minimize power consumption
        return -power_consumption

class ASMPolicyNetwork(nn.Module):
    def __init__(self, num_bs):
        super(ASMPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_bs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_bs * 4)  # 4 possible ASM levels for each BS

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x).view(-1, 4), dim=-1)  # Outputs probabilities for ASM levels per BS

def select_actions(agent, state, epsilon=0.1):
    state_tensor = torch.tensor(state, dtype=torch.float)
    probs = agent(state_tensor).detach().numpy()
    actions = []
    for prob in probs:
        if random.random() < epsilon:  # Exploration
            action = random.randint(0, 3)  # Random ASM level
        else:  # Exploitation
            action = np.argmax(prob)  # Select ASM level with highest probability
        actions.append(action)
    return actions

def train_agent(env, agent, episodes=1000, lr=1e-4, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    rewards = []  # List to store total reward for each episode
    
    for episode in range(episodes):
        state = env.state
        actions = select_actions(agent, state, epsilon)
        next_state, reward = env.step(actions)
        
        # Track reward for plotting
        rewards.append(reward)
        
        # Update epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Convert reward to a torch tensor for backpropagation
        reward_tensor = torch.tensor(reward, dtype=torch.float, requires_grad=True)
        loss = -reward_tensor  # Minimize negative reward to maximize reward
        
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update state
        state = next_state
    
    # Plot the rewards over episodes and save it
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Progression Over Episodes')
    plt.savefig("reward_progression.png")  # Save plot as a PNG file
    plt.close()  # Close the plot to avoid displaying

# Initialize environment and agent
num_bs = 7  # Example with 7 base stations
env = ASMSleepEnvironment(num_bs)
agent = ASMPolicyNetwork(num_bs)

# Train the agent and save the reward plot
train_agent(env, agent, episodes=1000)
