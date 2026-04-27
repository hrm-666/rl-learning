from pathlib import Path
from datetime import datetime
import gymnasium as gym
import torch
import torch.nn as nn


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


def test():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = Path(__file__).resolve().parent / "cartpole_videos" / timestamp
    video_dir.mkdir(exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: episode_id < 2,
        name_prefix="dqn_cartpole_inference"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = QNetwork(state_dim, action_dim).to(device)
    model_path = Path(__file__).resolve().parent / "dqn_cartpole.pth"
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    for episode in range(1):
        state, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)

            action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated
            state = next_state
            step += 1

            print(
                f"episode={episode + 1}, step={step}, "
                f"Q values: {q_values.cpu().numpy()}, action: {action}, reward: {reward}"
            )

        print(f"第 {episode + 1} 局结束, 总奖励: {total_reward}")

    print("测试结束")
    print(f"视频已保存到: {video_dir}")

    env.close()


if __name__ == "__main__":
    test()
