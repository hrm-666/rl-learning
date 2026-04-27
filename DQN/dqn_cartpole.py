import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. Q 网络
# =========================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 2. 经验回放池
# =========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =========================
# 3. ε-greedy 选动作
# =========================
def select_action(state, policy_net, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state)

    return q_values.argmax(dim=1).item()


# =========================
# 4. Soft Update
# =========================
def soft_update(policy_net, target_net, tau):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            tau * policy_param.data + (1.0 - tau) * target_param.data
        )


# =========================
# 5. 一次 DQN 更新
# =========================
def optimize_model(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    # 当前 Q(s, a)
    q_values = policy_net(states).gather(1, actions)

    # DQN target: r + gamma * max_a Q_target(s', a)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Huber loss 比 MSE 更稳
    loss = F.smooth_l1_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪，防止梯度突然爆掉
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)

    optimizer.step()

    return loss.item()


# =========================
# 6. 独立评估函数
# =========================
def evaluate(policy_net, env_name, device, episodes=20):
    env = gym.make(env_name)
    rewards = []

    for _ in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = policy_net(state_tensor)

            action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        rewards.append(total_reward)

    env.close()

    return np.mean(rewards), np.std(rewards)


# =========================
# 7. 主训练函数
# =========================
def train():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    replay_buffer = ReplayBuffer(capacity=50000)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-4)

    gamma = 0.99
    batch_size = 128

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    num_episodes = 1000
    tau = 0.005

    reward_history = []
    best_eval_reward = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        last_loss = None

        while not done:
            action = select_action(
                state,
                policy_net,
                epsilon,
                action_dim,
                device
            )

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            loss = optimize_model(
                policy_net,
                target_net,
                replay_buffer,
                optimizer,
                batch_size,
                gamma,
                device
            )

            if loss is not None:
                last_loss = loss

            # Soft update target network
            soft_update(policy_net, target_net, tau)

            state = next_state
            total_reward += reward

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        reward_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_train_reward = np.mean(reward_history[-10:])
            loss_text = f"{last_loss:.4f}" if last_loss is not None else "N/A"

            print(
                f"Episode {episode + 1}, "
                f"Train Avg Reward: {avg_train_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Loss: {loss_text}"
            )

        # 每 20 轮做一次独立评估
        if (episode + 1) % 20 == 0:
            eval_mean, eval_std = evaluate(policy_net, env_name, device, episodes=20)

            print(
                f"==> Eval at Episode {episode + 1}: "
                f"Mean Reward: {eval_mean:.2f}, "
                f"Std: {eval_std:.2f}"
            )

            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                torch.save(policy_net.state_dict(), "best_dqn_cartpole.pth")
                print(f"保存最好模型: best_dqn_cartpole.pth, Eval Reward = {best_eval_reward:.2f}")

    env.close()

    torch.save(policy_net.state_dict(), "last_dqn_cartpole.pth")
    print("训练结束")
    print("最后模型保存为 last_dqn_cartpole.pth")
    print("最好模型保存为 best_dqn_cartpole.pth")


if __name__ == "__main__":
    train()