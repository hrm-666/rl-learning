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
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers import ReplayBuffer



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
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10
    dqn_type: str = "dueling" #可选项：vanilla, double,dueling


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

class QNetwork(nn.Module):
    def __init__(self, env, dqn_type="vanilla"):
        super().__init__()
        self.dqn_type = dqn_type
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = env.single_action_space.n

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        if dqn_type == "dueling":
            self.value_stream = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
        else:
            self.q_head = nn.Linear(128, action_dim)
    def forward(self, x):
        features = self.feature(x)

        if self.dqn_type == "dueling":
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_head(features)
        return q_values

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1,"vectorized envs are not supported at the moment"

    run_name = f"{args.env_id}__{args.dqn_type}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            name=run_name, 
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )

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

    #启动环境
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, dqn_type=args.dqn_type).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_q_network = QNetwork(envs, dqn_type=args.dqn_type).to(device)
    target_q_network.load_state_dict(q_network.state_dict())

    #经验回放缓冲区
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False
    )
    start_time = time.time()

    #正式开始训练
    obs, _ =envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        #epsilon-greedy action selection
        #TODO:可以适当修改这里的动作更新方法，进行对比分析
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(action)

        if "episode" in infos and infos["_episode"][0]:
            episodic_return = infos["episode"]["r"][0]
            episodic_length = infos["episode"]["l"][0]
            print(f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}")
            writer.add_scalar("train/episodic_return", episodic_return, global_step)
            writer.add_scalar("train/episodic_length", episodic_length, global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_obs" in infos:
                    real_next_obs[idx] = infos["final_obs"][idx]
                elif "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]

        
        replay_buffer.add(obs, real_next_obs, action, rewards, terminations, infos)

        obs = next_obs

        #TODO:可以适当修改这里的训练更新方法，进行对比分析
        if global_step > args.learning_starts:#当经验回放缓冲区中积累了足够的经验后，才开始训练
            if global_step % args.train_frequency == 0:
                data = replay_buffer.sample(args.batch_size)
                with torch.no_grad():
                    if args.dqn_type == "double":
                        next_q_values = q_network(data.next_observations)
                        next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
                        target_max = target_q_network(data.next_observations).gather(1, next_actions).squeeze(1)
                    else: 
                        target_max, _ = target_q_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    target_max_mean = target_max.mean().item()
                    target_q_mean = td_target.mean().item()
                old_val = q_network(data.observations).gather(1, data.actions.long()).squeeze(1)
                loss = F.mse_loss(old_val, td_target)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    #print(f"SPS", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("compare/target_q_mean", target_q_mean, global_step)
                    writer.add_scalar("compare/target_max_mean", target_max_mean, global_step)
                    writer.add_scalar("compare/pred_q_mean", old_val.mean().item(), global_step)
                    writer.add_scalar("compare/q_target_gap", old_val.mean().item() - td_target.mean().item(), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_q_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
    if args.save_model:
        model_path = f"runs/{run_name}/{args.dqn_type}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=1000,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
            dqn_type=args.dqn_type,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
