import sys
from datetime import datetime
from pathlib import Path
import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 让脚本在 DQN_new 目录下直接运行时，也能导入项目根目录中的 rl_utils.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import rl_utils


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    """DQN算法"""

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor([state], dtype=torch.float32, device=self.device)
        return self.q_net(state).argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            transition_dict['actions'], dtype=torch.long, device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(
            transition_dict['rewards'], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.tensor(
            transition_dict['next_states'], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            transition_dict['dones'], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        return dqn_loss.item()


def create_experiment_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_dir / "experiments" / f"run_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_training_artifacts(experiment_dir, agent, return_list, moving_avg):
    torch.save(agent.q_net.state_dict(), experiment_dir / "q_net_final.pth")

    plt.figure(figsize=(10, 6))
    episodes_list = list(range(1, len(return_list) + 1))
    plt.plot(episodes_list, return_list, alpha=0.4, label="return")
    plt.plot(episodes_list, moving_avg, linewidth=2, label="moving average")
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(experiment_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_hyperparameters(experiment_dir, config):
    with open(experiment_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main():
    lr = 2e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.1
    target_update = 20
    buffer_size = 50000
    minimal_size = 1000
    batch_size = 128
    random_seed = 0
    env_name = "CartPole-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    env = gym.make(env_name)
    env.reset(seed=random_seed)
    try:
        env.action_space.seed(random_seed)
    except AttributeError:
        pass

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    script_dir = Path(__file__).resolve().parent
    experiment_dir = create_experiment_dir(script_dir)

    return_list = []
    loss_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                episode_losses = []
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d
                        }
                        loss = agent.update(transition_dict)
                        episode_losses.append(loss)
                return_list.append(episode_return)
                loss_list.append(float(np.mean(episode_losses)) if episode_losses else None)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                        "return": "%.3f" % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    env.close()

    moving_avg = rl_utils.moving_average(return_list, 9)
    config = {
        "env_name": env_name,
        "learning_rate": lr,
        "num_episodes": num_episodes,
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "epsilon": epsilon,
        "target_update": target_update,
        "buffer_size": buffer_size,
        "minimal_size": minimal_size,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "device": str(device),
    }
    save_training_artifacts(experiment_dir, agent, return_list, moving_avg)
    save_hyperparameters(experiment_dir, config)

    print(f"\n实验结果已保存到: {experiment_dir}")
    print(f"最终权重: {experiment_dir / 'q_net_final.pth'}")
    print(f"训练曲线: {experiment_dir / 'training_curve.png'}")
    print(f"超参数: {experiment_dir / 'hyperparameters.json'}")


if __name__ == "__main__":
    main()
