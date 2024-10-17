import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# ASM parameters
ASM_PARAMETERS = {0: {'latency': 0, 'power_discount': 1.0},
                  1: {'latency': 1, 'power_discount': 0.69},
                  2: {'latency': 10, 'power_discount': 0.5},
                  3: {'latency': 100, 'power_discount': 0.29}}

class ASMSleepEnvironment:
    def __init__(self, num_bs):
        self.num_bs = num_bs
        self.state = np.zeros(num_bs, dtype=int)
        self.ue_count = np.full(num_bs, 10)  # Set as needed
        self.average_power = np.zeros(num_bs)
        self.throughput_ul = np.zeros(num_bs)
        self.throughput_dl = np.zeros(num_bs)
        self.latency_ul = np.zeros(num_bs)
        self.latency_dl = np.zeros(num_bs)

    def step(self, actions):
        self.state = np.array(actions)
        self.calculate_throughput()
        self.calculate_latency()
        self.calculate_power()

        total_throughput = np.sum(self.throughput_ul + self.throughput_dl)
        total_power = np.sum(self.average_power)
        energy_efficiency = total_throughput / total_power if total_power > 0 else 0

        qos_penalty = self.calculate_qos_penalty()
        correlation_penalty = self.calculate_correlation_penalty()
        reward = self.calculate_reward(energy_efficiency, qos_penalty, correlation_penalty)
        return self.state, reward

    def calculate_throughput(self):
        for i in range(self.num_bs):
            asm_factor = ASM_PARAMETERS[self.state[i]]['power_discount']
            max_ul = 15
            max_dl = 20
            self.throughput_ul[i] = max_ul * asm_factor * (50 / (self.ue_count[i] + 1))
            self.throughput_dl[i] = max_dl * asm_factor * (50 / (self.ue_count[i] + 1))

    def calculate_latency(self):
        for i in range(self.num_bs):
            asm_latency_factor = ASM_PARAMETERS[self.state[i]]['latency']
            base_latency_ul = 20
            base_latency_dl = 20
            load_factor = self.ue_count[i] / 50
            self.latency_ul[i] = (base_latency_ul * (1 + load_factor)) + asm_latency_factor
            self.latency_dl[i] = (base_latency_dl * (1 + load_factor)) + asm_latency_factor

    def calculate_power(self):
        for i in range(self.num_bs):
            base_power = 10
            asm_factor = ASM_PARAMETERS[self.state[i]]['power_discount']
            self.average_power[i] = base_power * asm_factor

    def calculate_qos_penalty(self):
        qos_penalty = 0
        for i in range(self.num_bs):
            load_factor = self.ue_count[i] / 20

            # Increase penalties for high UE count in ASM mode 3
            if self.ue_count[i] >= 15 and self.state[i] == 3:
                qos_penalty += 300 * load_factor
            elif self.ue_count[i] >= 10 and self.state[i] == 3:
                qos_penalty += 200 * load_factor
            elif self.ue_count[i] >= 5 and self.state[i] == 3:
                qos_penalty += 100 * load_factor

            # Penalty for low throughput and high latency
            if self.latency_ul[i] > 50 or self.latency_dl[i] > 50:
                qos_penalty += 200 * load_factor
            if self.throughput_ul[i] < 10 or self.throughput_dl[i] < 10:
                qos_penalty += 150 * load_factor

            # Baseline penalties for ASM modes 0-2 under certain load conditions
            if self.state[i] == 2 and self.ue_count[i] >= 15:
                qos_penalty += 100 * load_factor
            elif self.state[i] == 1 and self.ue_count[i] >= 10:
                qos_penalty += 50 * load_factor

        return qos_penalty

    def calculate_correlation_penalty(self):
        # Additional penalty for inconsistencies in latency and throughput
        correlation_penalty = 0
        for i in range(self.num_bs):
            if self.throughput_ul[i] > 30 and self.latency_ul[i] > 50:
                correlation_penalty += 50
            if self.throughput_dl[i] > 40 and self.latency_dl[i] > 50:
                correlation_penalty += 50
        return correlation_penalty

    def calculate_reward(self, energy_efficiency, qos_penalty, correlation_penalty):
        qos_importance = 1200  # Stronger focus on QoS
        energy_importance = 200  # Reduced importance on power savings

        reward = (qos_importance - qos_penalty - correlation_penalty) + (energy_efficiency + energy_importance)
        return reward
class ActorCritic(nn.Module):
    def __init__(self, num_bs):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_bs, 128)
        self.policy_head = nn.Linear(128, num_bs * 4)
        self.value_head = nn.Linear(128, num_bs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        policy = torch.softmax(self.policy_head(x).view(-1, 4), dim=-1)
        value = self.value_head(x).squeeze(-1)
        return policy, value

def compute_ppo_loss(old_log_probs, states, actions, advantages, returns, clip_param):
    new_probs, new_values = model(states)
    new_probs = new_probs.view(-1, 4)
    actions_flat = actions.view(-1, 1)
    selected_new_probs = new_probs.gather(1, actions_flat).squeeze(1)

    old_log_probs = old_log_probs.view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)
    new_values = new_values.view(-1)

    ratio = torch.exp(selected_new_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = (returns - new_values).pow(2).mean()
    
    return policy_loss + 0.5 * value_loss

# def select_action(policy, env, epsilon):
#     actions = torch.zeros(policy.shape[0], dtype=torch.long)
#     for i in range(policy.shape[0]):
#         ue_count = env.ue_count[i]
#         # Define preferred ASM modes based on fine-grained UE count conditions
#         if 15 <= ue_count <= 20:
#             preferred_modes = [0, 1]
#         elif ue_count > 20:
#             preferred_modes = [0]
#         elif 10 <= ue_count < 15:
#             preferred_modes = [1, 2]
#         elif 5 <= ue_count < 10:
#             preferred_modes = [2, 3]
#         else:  # ue_count < 5
#             preferred_modes = [3]

#         # Select action with exploration or based on policy
#         if np.random.rand() < epsilon:
#             actions[i] = torch.tensor(np.random.choice(actions))
#         else:
#             # Select based on policy, but only over preferred modes
#             dist = torch.distributions.Categorical(policy[i][preferred_modes])
#             selected_mode = dist.sample().item()
#             actions[i] = preferred_modes[selected_mode]
            
#     return actions
def select_action(policy, epsilon):
    # Epsilon-greedy action selection to allow free exploration of all ASM modes
    if np.random.rand() < epsilon:
        return torch.tensor(np.random.randint(0, 4, size=(policy.shape[0],)), dtype=torch.long)
    else:
        dist = torch.distributions.Categorical(policy)
        return dist.sample()


def train_ppo(env, model, episodes=1000, lr=1e-3, gamma=0.99, clip_param=0.2, epsilon=0.1, epsilon_decay=0.995):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_rewards = []
    all_losses = []
    metrics_data = []

    for episode in range(episodes):
        state = torch.tensor(env.state, dtype=torch.float)
        log_probs, values, rewards, states, actions = [], [], [], [], []
        episode_data = []

        policy, value = model(state)
        action = select_action(policy,epsilon)
        epsilon = max(0.01, epsilon * epsilon_decay)

        dist = torch.distributions.Categorical(policy)
        log_prob = dist.log_prob(action)

        log_probs.append(log_prob)
        values.append(value)
        actions.append(action)
        states.append(state)

        state, reward = env.step(action.detach().numpy())
        state = torch.tensor(state, dtype=torch.float)
        rewards.append(reward)

        for bs in range(env.num_bs):
            episode_data.append({
                "Episode": episode,
                "BaseStation": bs,
                "ASM_Mode": action[bs].item(),
                "Throughput_UL": env.throughput_ul[bs],
                "Throughput_DL": env.throughput_dl[bs],
                "Power": env.average_power[bs],
                "Latency_UL": env.latency_ul[bs],
                "Latency_DL": env.latency_dl[bs],
                "Cell_Load": env.ue_count[bs]
            })

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).repeat(env.num_bs, 1).view(-1)

        values = torch.cat(values).view(-1)
        advantages = (returns - values).detach()

        old_log_probs = torch.stack(log_probs).view(-1)
        states = torch.stack(states)
        actions = torch.stack(actions).view(-1, 1)

        optimizer.zero_grad()
        loss = compute_ppo_loss(old_log_probs, states, actions, advantages, returns, clip_param)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        all_rewards.append(sum(rewards))
        all_losses.append(loss.item())
        metrics_data.extend(episode_data)

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {all_rewards[-1]}, Loss: {all_losses[-1]}")

    with open('asm_metrics.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Episode", "BaseStation", "ASM_Mode", "Throughput_UL", "Throughput_DL", "Power", "Latency_UL", "Latency_DL", "Cell_Load"])
        writer.writeheader()
        writer.writerows(metrics_data)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.plot(all_rewards, label='Total Reward')
    plt.plot(pd.Series(all_rewards).rolling(50).mean(), color='red',label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Over Episodes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Over Episodes')

    plt.tight_layout()
    plt.savefig("ppo_training_monitor.png")
    plt.close()

# Initialize the environment and model
env = ASMSleepEnvironment(num_bs=2)
model = ActorCritic(num_bs=2)
train_ppo(env, model, episodes=5000)
