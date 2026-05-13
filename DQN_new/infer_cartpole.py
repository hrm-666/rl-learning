import argparse
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    """与训练脚本保持一致的Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def find_latest_experiment(experiments_dir):
    candidates = [p for p in experiments_dir.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("未找到任何实验目录，请先运行 DQN.py 完成训练。")
    return sorted(candidates)[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的DQN权重推理 CartPole")
    parser.add_argument(
        "--experiment",
        type=str,
        default="latest",
        help="实验目录名，例如 run_20260507_160000；默认使用 latest"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="推理回合数"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="是否录制视频到实验目录下的 inference_videos"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    experiments_dir = script_dir / "experiments"
    env_name = "CartPole-v1"
    hidden_dim = 128
    random_seed = 0

    if args.experiment == "latest":
        experiment_dir = find_latest_experiment(experiments_dir)
    else:
        experiment_dir = experiments_dir / args.experiment
        if not experiment_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    weight_path = experiment_dir / "q_net_final.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"缺少权重文件: {weight_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_mode = "rgb_array" if args.record else "human"
    env = gym.make(env_name, render_mode=render_mode)
    if args.record:
        video_dir = experiment_dir / "inference_videos"
        video_dir.mkdir(exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: True,
            name_prefix="cartpole_inference"
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
    q_net.load_state_dict(torch.load(weight_path, map_location=device))
    q_net.eval()

    print(f"加载实验目录: {experiment_dir}")
    print(f"加载权重文件: {weight_path}")

    for episode in range(args.episodes):
        state, _ = env.reset(seed=random_seed + episode)
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            state_tensor = torch.tensor(
                [state], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action = q_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step += 1

        print(f"episode={episode + 1}, steps={step}, total_reward={total_reward}")

    env.close()
    if args.record:
        print(f"推理视频已保存到: {experiment_dir / 'inference_videos'}")


if __name__ == "__main__":
    main()
