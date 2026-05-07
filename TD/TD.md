# SARSA、Q-learning 与 n-step SARSA 学习笔记

---

# 1. SARSA 算法的核心思想

SARSA 是一种：

```text
On-Policy
Model-Free
Temporal Difference (TD)
```

强化学习算法。

SARSA 名字来源于：

```text
S → A → R → S' → A'
```

即：

| 符号 | 含义   |
| -- | ---- |
| S  | 当前状态 |
| A  | 当前动作 |
| R  | 当前奖励 |
| S' | 下一状态 |
| A' | 下一动作 |

---

# 2. SARSA 更新公式

SARSA 的更新公式：

Q(s,a) \leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)]

其中：

* α：学习率
* γ：折扣因子
* Q(s',a')：下一状态下，真实选择动作后的 Q 值

---

# 3. 如何理解“当前动作”

SARSA 中：

```python
state = env.reset()
action = agent.take_action(state)
```

这里：

```python
action
```

就是当前动作。

---

在循环中：

```python
next_state, reward, done = env.step(action)
```

执行当前动作。

然后：

```python
next_action = agent.take_action(next_state)
```

得到下一动作。

最后更新：

```python
agent.update(state, action, reward, next_state, next_action)
```

因此：

```text
当前动作 = 上一轮已经选好的动作
```

而：

```python
action = next_action
```

又会让：

```text
下一动作 → 变成新的当前动作
```

---

# 4. ε-greedy 与 SARSA 的关系

代码中：

```python
if np.random.random() < self.epsilon:
    action = np.random.randint(self.n_action)
else:
    action = np.argmax(self.Q_table[state])
```

说明：

* 大部分时间选择 Q 最大动作
* 小概率随机探索

---

很多人误以为：

```text
SARSA 不应该使用 argmax
```

实际上：

```text
SARSA 与 Q-learning 都可以使用 ε-greedy 选动作
```

真正区别不在：

```text
怎么选动作
```

而在：

```text
如何更新 Q 值
```

---

# 5. SARSA 与 Q-learning 的根本区别

---

## SARSA

更新目标：

r+\gamma Q(s',a')

其中：

```text
a'
```

是真实按照 ε-greedy 选出来的动作。

因此：

```text
SARSA 学习的是“当前真实执行策略”
```

属于：

```text
On-Policy
```

---

## Q-learning

更新目标：

r+\gamma \max_{a'}Q(s',a')

即：

```text
永远假设未来选择最优动作
```

因此：

```text
Q-learning 学习的是“理论最优策略”
```

属于：

```text
Off-Policy
```

---

# 6. On-Policy 与 Off-Policy

---

# On-Policy

核心思想：

```text
按照什么策略行动，就学习什么策略
```

例如：

* SARSA
* PPO
* A2C

特点：

* 更稳定
* 更保守
* 更符合真实执行行为

---

# Off-Policy

核心思想：

```text
可以用别的策略产生数据，但学习目标策略
```

例如：

* Q-learning
* DQN
* SAC
* DDPG

特点：

* 样本利用率高
* 可使用 Replay Buffer
* 更激进

---

# 7. 为什么 SARSA 更保守

经典例子：

```text
悬崖行走 Cliff Walking
```

Q-learning：

```text
假设未来永远最优
```

因此：

```text
喜欢贴着悬崖边走
```

因为路径最短。

---

但：

由于 ε-greedy 仍然存在探索：

```text
未来仍可能随机掉下悬崖
```

---

SARSA 会考虑：

```text
未来真实还会探索
```

因此：

```text
会主动离悬崖远一点
```

所以：

| 算法         | 风格      |
| ---------- | ------- |
| SARSA      | 保守、安全   |
| Q-learning | 激进、理论最优 |

---

# 8. SARSA 与 Q-learning 的本质区别

实际上：

```text
二者环境交互流程几乎完全一样
```

最大区别在于：

# TD Target（更新目标）

---

## SARSA Target

r+\gamma Q(s',a')

---

## Q-learning Target

r+\gamma \max Q(s')

---

强化学习中很多算法本质上都可以理解为：

# 不同 Target 的设计

---

# 9. n-step SARSA

普通 SARSA：

```text
一步更新
```

即：

```text
只看一步奖励
```

---

n-step SARSA：

```text
看未来 n 步真实奖励
```

---

## n-step SARSA 更新目标

G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots+\gamma^n Q(s_{t+n},a_{t+n})

即：

* 前 n 步使用真实奖励
* 第 n 步之后使用 Q 估计

---

# 10. 为什么要使用 n-step

普通 SARSA：

```text
奖励传播慢
```

例如：

```text
掉悬崖的惩罚只能一步一步往前传播
```

---

n-step SARSA：

```text
一次更新会影响更前面的状态
```

因此：

```text
奖励传播更快
```

---

# 11. n-step SARSA 代码核心思想

代码中：

```python
self.state_list
self.action_list
self.reward_list
```

用于保存最近 n 步轨迹。

---

每一步：

```python
append()
```

加入缓冲区。

---

当：

```python
len(state_list) == n
```

时：

开始进行 n 步更新。

---

更新核心：

```python
G = gamma * G + reward
```

倒序递推：

```text
不断向前累积折扣奖励
```

最终得到：

```text
n-step return
```

---

# 12. n-step 的本质位置

可以理解为：

```text
TD 与 Monte Carlo 的折中
```

---

## 1-step SARSA

特点：

```text
稳定
但传播慢
```

---

## Monte Carlo

特点：

```text
传播快
但方差大
```

---

## n-step SARSA

特点：

```text
兼顾稳定性与传播速度
```

---

# 13. 整体强化学习理解

可以把很多 RL 算法统一理解为：

Q \leftarrow Q+\alpha(\text{Target}-Q)

区别核心：

```text
Target 如何构造
```

---

例如：

| 算法           | Target             |
| ------------ | ------------------ |
| SARSA        | r + γQ(s',a')      |
| Q-learning   | r + γmaxQ(s')      |
| n-step SARSA | 多步真实奖励 + Bootstrap |
| DQN          | Q-learning + 神经网络  |
| Double DQN   | 分离动作选择与评估          |

---

# 14. 最终总结

---

# SARSA

```text
学习当前真实执行策略
考虑未来探索风险
更保守、更稳定
```

---

# Q-learning

```text
学习理论最优策略
假设未来永远最优
更激进、更高效
```

---

# n-step SARSA

```text
利用多步真实奖励
加快奖励传播
是 TD 与 Monte Carlo 的折中
```

---

# 强化学习很多算法的本质：

# 不同 Bellman Target 的设计。
