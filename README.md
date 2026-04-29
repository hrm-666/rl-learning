# RL Learning Notes and Experiments

这是一个我在初学强化学习过程中逐步整理出来的小工程，主要用于记录和实践从表格方法到深度强化学习的入门内容。

目前仓库包含三部分：

- `MAB/`：多臂老虎机问题的代码实验、图像结果与学习笔记
- `Q-learning/`：基于 `FrozenLake` 的 Q-learning 练习与笔记
- `DQN/`：基于 `CartPole` 的 DQN 练习、推理脚本与笔记

这个仓库的定位更偏向“学习记录 + 小型实验”，所以代码和笔记会一起保留，方便后续回顾自己当时是怎么理解这些算法的。

## Project Structure

```text
RL/
├─ MAB/
│  ├─ mab.py
│  ├─ MAB学习笔记.md
│  ├─ MAB学习笔记.pdf
│  └─ figures/
│     ├─ comparison.png
│     ├─ epsilon_greedy.png
│     ├─ decay_epsilon_greedy.png
│     ├─ ucb.png
│     └─ thompson_sampling.png
├─ Q-learning/
│  ├─ FrozenLake.py
│  ├─ FrozenLake_inference.py
│  ├─ cartpole.py
│  └─ Q-learning笔记.md
├─ DQN/
│  ├─ dqn_cartpole.py
│  ├─ DQN_CartPole_interface.py
│  └─ DQN笔记.md
├─ README.md
└─ requirements.txt
```

## Included Content

### 1. Multi-Armed Bandit

`MAB/` 目录主要包含：

- `mab.py`
  实现 Bernoulli Bandit 环境，以及 `epsilon-greedy`、衰减 `epsilon-greedy`、`UCB`、`Thompson Sampling` 四种策略
- `MAB学习笔记.md`
  记录自己对多臂老虎机问题的系统梳理，包括：
  - 多臂老虎机的背景与探索/利用问题
  - 奖励估计值的递推更新公式推导
  - 累计懊悔值的定义与意义
  - 四种经典 bandit 算法的核心思想
  - 代码实现和实验结果分析
- `MAB学习笔记.pdf`
  Markdown 笔记导出的 PDF 版本，便于归档和阅读
- `figures/`
  保存实验生成的累计懊悔曲线图，包括单算法曲线和总对比图

### 2. Q-learning

`Q-learning/` 目录主要包含：

- `FrozenLake.py`
  使用 `FrozenLake-v1` 训练 Q 表
- `FrozenLake_inference.py`
  使用训练好的 Q 表进行策略推理与可视化
- `Q-learning笔记.md`
  记录自己对 Q-learning 核心概念的理解，包括：
  - `Q(s, a)` 的含义
  - 更新公式为什么要看 `next_state`
  - 价值为什么会呈现“从后往前传递”的感觉
  - `alpha`、`gamma`、`epsilon` 的作用

### 3. DQN

`DQN/` 目录主要包含：

- `dqn_cartpole.py`
  使用 DQN 训练 `CartPole-v1`
- `DQN_CartPole_interface.py`
  加载训练好的模型进行推理，并支持录制视频
- `DQN笔记.md`
  记录自己对 DQN 的理解，包括：
  - 为什么 Q 表不能直接处理连续状态
  - DQN 如何用神经网络近似 Q 函数
  - replay buffer 和 target network 的作用
  - `done`、`.gather()`、`torch.no_grad()` 等训练细节

## Environment

推荐使用 Python 3.11 左右版本，并通过 Conda 或 venv 创建独立环境。

安装依赖：

```bash
pip install -r requirements.txt
```

如果你只想运行其中一部分代码，也可以按需安装：

- `gymnasium`
- `numpy`
- `torch`
- `moviepy`

## How to Run

### Multi-Armed Bandit

运行多臂老虎机实验：

```bash
python MAB/mab.py
```

该脚本会：

- 随机生成一个 `10` 臂 Bernoulli Bandit
- 分别运行 `epsilon-greedy`、衰减 `epsilon-greedy`、`UCB`、`Thompson Sampling`
- 输出各策略的累计懊悔值
- 绘制累计懊悔曲线

### Q-learning

训练：

```bash
python Q-learning/FrozenLake.py
```

推理：

```bash
python Q-learning/FrozenLake_inference.py
```

### DQN

训练：

```bash
python DQN/dqn_cartpole.py
```

推理与录制：

```bash
python DQN/DQN_CartPole_interface.py
```

## Notes

- 这个仓库主要服务于个人学习过程，因此更强调“逐步理解”和“可回顾性”，而不是工程化封装。
- 一些模型权重、录制视频、缓存文件等运行产物默认不会纳入 Git 跟踪。
- 目前内容已经覆盖多臂老虎机、Q-learning 和 DQN，整体学习路径也更完整了：先从 bandit 理解探索与利用，再过渡到带状态转移的强化学习。

## Future Work

后面准备继续补充的方向：

- contextual bandit
- Double DQN
- Dueling DQN
- PPO
- 更系统的实验记录与结果对比
