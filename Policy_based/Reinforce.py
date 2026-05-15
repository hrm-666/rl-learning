import random
import os
import numpy as np
import time
from dataclasses import dataclass
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

@dataclass 
class Args:
    #实验类型参数
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = True

    #算法具体参数
    env_id: str = "CartPole-v1"
    total_timesteps: int = 200000
    learning_rate: float = 5e-4
    num_envs: int = 1
    gamma: float = 0.99

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env
    return thunk


class PolicyNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.single_action_space.n),
        )

    def forward(self, x):
        logits = self.network(x)
        return logits
    
def evaluate(
    model_path,
    make_env,
    env_id,
    eval_episodes,
    run_name,
    Model,
    device=torch.device("cpu"),
    capture_video=False,
):

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed=0, idx=0, capture_video=capture_video, run_name=run_name)]
    )
    try:
        model = Model(envs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        obs, _ = envs.reset()
        episodic_returns = []

        while len(episodic_returns) < eval_episodes:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            logits = model(obs_tensor)
            action = torch.argmax(logits, dim=1).cpu().numpy()

            next_obs, reward, termination, truncation, infos = envs.step(action)

            if "episode" in infos and infos["_episode"][0]:
                episodic_return = infos["episode"]["r"][0]
                print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
                episodic_returns.append(episodic_return)

            obs = next_obs

        return episodic_returns
    finally:
        envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "目前只支持单环境训练"

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )   

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    policy_network = PolicyNetwork(envs).to(device)
    optimizer = optim.Adam(policy_network.parameters(), lr=args.learning_rate)

    global_step = 0
    episode = 0
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)

    while global_step < args.total_timesteps:
        log_probs = []
        rewards = []
        episodic_return = 0
        done = False

        while not done and global_step < args.total_timesteps:
            obs_tensor = torch.Tensor(obs).to(device)
            logits = policy_network(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, termination, truncation, infos = envs.step(action.cpu().numpy())

            log_probs.append(log_prob.squeeze())
            rewards.append(reward[0])
            episodic_return += reward[0]

            obs = next_obs
            done = termination[0] or truncation[0]
            global_step += 1

            if "episode" in infos and infos["_episode"][0]:
                episodic_length = infos["episode"]["l"][0]
                print(f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}")
                writer.add_scalar("train/episodic_return", episodic_return, global_step)
                writer.add_scalar("train/episodic_length", episodic_length, global_step)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("losses/policy_loss", loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if done:
            obs, _ = envs.reset()
            episode += 1

    if args.save_model:
        model_path = f"runs/{run_name}/policy_gradient_model.pt"
        torch.save(policy_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=1000,
            run_name=f"{run_name}-eval",
            Model=PolicyNetwork,
            device=device,
            capture_video=False,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
