# DQN 学习笔记

这篇笔记主要是给自己复习用，重点把下面几件事讲清楚：

- 为什么 Q-learning 不能直接处理连续状态
- DQN 到底是在学什么
- 神经网络在 DQN 里扮演什么角色
- 为什么 DQN 必须依赖 replay buffer 和 target network
- 训练时的几个关键细节为什么不能省

## 1. DQN 要解决什么问题

Q-learning 在表格环境里很好理解，因为我们可以直接维护一张 Q 表：

- 行表示状态 `state`
- 列表示动作 `action`
- 每个格子存 `Q(s, a)`

但是当状态空间变大，尤其变成连续状态时，Q 表就很难用了。

例如 CartPole 的状态是：

```python
[位置, 速度, 杆子角度, 杆子角速度]
```

这 4 个量都是连续值，所以状态几乎是无限多的。

这时就没法像 FrozenLake 那样，直接开一张有限大小的表，把每个状态动作对的值都存下来。

所以 DQN 的核心任务可以理解成：

`当 Q 表存不下时，用神经网络去近似 Q 函数`

## 2. DQN 在学什么

DQN 本质上还是在学动作价值函数：

`Q(s, a)`

它学的东西和 Q-learning 没变，变的只是表达方式。

所以我应该把 DQN 理解成：

- 不是换了一个强化学习目标
- 而是把“表格存 Q 值”换成了“网络估计 Q 值”

也就是说，DQN 仍然是在回答这个问题：

`在状态 s 下，如果我采取动作 a，未来大概能拿到多少累计回报`

## 3. DQN 和 Q-learning 的关系

DQN 不是脱离 Q-learning 的新东西，它其实是 Q-learning 在连续状态上的延伸。

Q-learning 的核心更新思想是：

```python
Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s',a') - Q(s,a))
```

而 DQN 的核心思想是：

- 不再显式维护 `Q[s, a]`
- 改成让网络输出 `Q(s, a; θ)`
- 用神经网络参数 `θ` 去逼近真实的 Q 函数

所以可以把两者关系记成：

- Q-learning：用表存 Q 值
- DQN：用网络算 Q 值

## 4. DQN 的网络到底在做什么

在 DQN 里，网络输入是当前状态 `state`，输出是“当前状态下每个动作的 Q 值”。

例如在 CartPole 里，动作只有两个：

- 向左
- 向右

那么网络输出就可以理解成：

```python
[Q(s, 左), Q(s, 右)]
```

这意味着网络不是直接输出“该选哪个动作”，而是在做：

`给每个动作打分`

然后再由我们自己选分数最高的动作。

## 5. 为什么网络输出维度是 action_dim

这是一个很容易卡住的点。

如果环境有 `action_dim` 个离散动作，那么网络最后一层通常就输出 `action_dim` 个值。

原因是：

- 每个输出对应一个动作的价值估计
- 网络一次前向传播，就把当前状态下所有动作的分数都算出来

所以：

- 不是输出一个动作编号
- 而是输出“所有动作各值多少钱”

真正决策时才会做：

```python
action = argmax(Q(s, a))
```

## 6. DQN 仍然是 value-based

这一点要记牢。

DQN 学的是 Q 值，不是动作概率。

也就是说：

- 它不是 policy-based 方法
- 它不是直接学 `π(a|s)`
- 它还是 value-based 方法

所以 DQN 的决策逻辑仍然非常直接：

- 先估计每个动作的 Q 值
- 再选择 Q 值最大的动作

## 7. DQN 的训练目标是什么

虽然 DQN 用了神经网络，但目标依然和 Q-learning 一脉相承。

核心目标值还是：

```python
target = reward + gamma * max Q(next_state, a')
```

如果 `done=True`，那么后续没有未来价值，一般写成：

```python
target = reward
```

或者统一写成：

```python
target = reward + gamma * next_q * (1 - done)
```

其中 `done=1` 时，后半项自动归零。

所以 DQN 的本质仍然是：

`让当前 Q 估计，朝着 当前奖励 + 未来最优收益 这个目标靠近`

## 8. DQN 为什么会不稳定

这里是 DQN 和表格 Q-learning 最大的不同之一。

表格 Q-learning 里，我们直接更新某个表格元素；
但 DQN 里，我们在更新一个神经网络。

这会带来两个问题：

- 神经网络近似本身会有误差
- 目标值里又用了网络自己的估计

也就是说，DQN 经常是在做这种事：

- 用当前网络去算预测值
- 又用网络去构造目标值
- 再拿这个目标反过来训练网络

这就会出现“自己给自己出答案”的现象。

所以 DQN 容易不稳定，本质原因是：

- 目标会变
- 数据相关性强
- 函数逼近误差会传播

## 9. 为什么需要 target network

target network 的作用就是让训练目标稳定一点。

通常 DQN 里会有两个网络：

- `policy_net`：正在学习的网络
- `target_net`：专门用来计算目标值的网络

更新时会写成类似：

```python
next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
target_q_values = rewards + gamma * next_q_values * (1 - dones)
```

这样做的意义是：

- 预测值来自 `policy_net`
- 目标值来自相对更稳定的 `target_net`

如果不用 target network，而是所有东西都用同一个网络算，那么网络参数一变，目标也跟着一起变，训练会更抖。

## 10. Hard Update 和 Soft Update

更新 target network 常见有两种方式。

### 10.1 Hard Update

每隔若干轮，直接复制一次参数：

```python
target_net.load_state_dict(policy_net.state_dict())
```

优点是简单。

缺点是：

- 目标网络会突然跳一下
- 稳定性一般

### 10.2 Soft Update

每次只更新一点点：

```python
target = tau * policy + (1 - tau) * target
```

优点是：

- 更新更平滑
- 通常更稳定

你现在的代码用的是 Hard Update，这对入门完全没问题。

## 11. 为什么需要 Replay Buffer

Replay Buffer 用来存经验：

```python
(state, action, reward, next_state, done)
```

它的核心作用不是“缓存起来好看”，而是解决在线数据太相关的问题。

如果智能体刚连续经历了一串相近状态，然后立刻用这些连续样本训练，那么：

- 数据相关性太强
- 更新方向容易偏
- 网络训练不稳定

Replay Buffer 的好处主要有三个：

- 打破样本之间的时间相关性
- 提高样本利用率
- 让训练更稳定，不容易遗忘旧经验

所以它不是可有可无的附属品，而是 DQN 稳定训练的关键组件。

## 12. `.gather()` 到底在做什么

这一句非常常见：

```python
q_values = policy_net(states).gather(1, actions)
```

它的作用是：

- 网络会输出每个状态下所有动作的 Q 值
- 但我们真正想更新的，不是所有动作
- 而是“这次样本里实际执行过的那个动作”的 Q 值

例如网络输出：

```python
[Q(s,左), Q(s,右)]
```

如果当前样本实际执行的是“右”，那就只应该取出 `Q(s,右)` 来参与 loss 计算。

所以 `.gather()` 的本质就是：

`从所有动作值里，挑出本次实际动作对应的那一个`

## 13. `done` 为什么这么重要

在终止状态下，未来已经结束了，所以不应该再把未来 Q 值接上去。

因此：

- 如果 `done=False`，目标值里有未来收益
- 如果 `done=True`，目标值里只保留当前奖励

这就是为什么代码里经常写：

```python
target = reward + gamma * next_q * (1 - done)
```

这里要注意：

- `done` 最后参与运算时，通常要转成数值型
- 如果是 PyTorch，常见做法是转成 `float32`

否则就容易出现布尔张量不能直接做减法的报错。

## 14. `torch.no_grad()` 为什么不能省

在计算 target 的时候，经常会写：

```python
with torch.no_grad():
    next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
```

它的意义是：

- 目标值只是“训练标签”
- 这部分不应该参与反向传播

如果不加 `no_grad()`，就相当于：

- 目标分支也被拉进计算图
- 梯度会沿着不该传播的方向传播
- 训练更乱，也更浪费显存和计算

所以这里不是“可加可不加的小优化”，而是概念上就应该加。

## 15. DQN 里的探索和利用

DQN 训练时通常也会用 `epsilon-greedy`：

- 以 `epsilon` 的概率随机探索
- 否则选择当前 Q 值最大的动作

这和表格 Q-learning 是一样的思路。

原因也一样：

- 网络刚开始时 Q 值估计很不准
- 如果一开始就只贪心，容易过早卡在不好的策略上
- 需要随机探索来收集更多样的经验

而在测试或推理时，一般会直接用：

```python
argmax(Q)
```

因为这时候目标是评估当前学到的策略，而不是继续探索。

## 16. 为什么训练时表现不错，测试时却可能不稳定

这是强化学习里很常见的现象。

可能原因包括：

- 训练阶段还有 `epsilon`，动作带随机性
- 神经网络估计本身有波动
- 测试局数太少，结果受随机性影响
- 当前保存的模型不一定是训练过程中最好的那个

所以在 DQN 里，通常不能只看一两局表现，而更适合看：

- 多局平均 reward
- 最佳模型表现
- 训练过程中的整体趋势

## 17. DQN 的一个经典问题：Q 值高估

标准 DQN 里，目标值常写成：

```python
max Q(next_state, a')
```

问题在于：

- `max` 很容易偏向那些“被偶然高估”的动作
- 久而久之，会让整体 Q 值偏高

这叫做：

`overestimation`

也就是 Q 值高估问题。

这也是后面 Double DQN 出现的重要原因之一。

## 18. 结合自己的代码理解一遍

在 `dqn_cartpole.py` 里，一次训练大致是：

1. 环境重置，得到初始状态
2. 用 `epsilon-greedy` 选动作
3. 执行动作，得到 `next_state`、`reward`、`done`
4. 把经验存进 replay buffer
5. 从 buffer 随机采样一批经验
6. 用 `policy_net` 算当前动作的 Q 值
7. 用 `target_net` 算目标值
8. 计算 loss，更新 `policy_net`
9. 定期同步 `target_net`

对应的关键流程可以概括成：

```python
action = select_action(...)
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
```

这个过程和 Q-learning 的核心精神其实是一致的：

- 先和环境交互
- 再根据结果修正当前估计

只是这里把“更新表格元素”换成了“更新网络参数”。

## 19. 这阶段我最该记住的几句话

- DQN 解决的是 Q 表无法处理大规模或连续状态的问题
- DQN 学的仍然是 `Q(s, a)`，不是动作概率
- 网络输入是状态，输出是当前状态下每个动作的 Q 值
- 决策时不是网络直接输出动作，而是对输出做 `argmax`
- Replay Buffer 用来打破样本相关性，Target Network 用来稳定目标值
- `done` 表示是否还有未来价值，终止状态下目标值通常只剩 `reward`
- DQN 会有 Q 值高估问题，这也是 Double DQN 的动机

## 20. 我目前最容易记错的点

### 错误 1：DQN 学的是动作概率

不对。

正确说法是：

DQN 学的是动作价值 `Q(s, a)`，属于 value-based 方法。

### 错误 2：网络直接输出最终动作

不对。

正确说法是：

网络输出的是每个动作的 Q 值，动作是通过 `argmax` 选出来的。

### 错误 3：Replay Buffer 只是为了存数据

不对。

正确说法是：

Replay Buffer 的关键作用是打破时间相关性、提高样本利用率、增强训练稳定性。

### 错误 4：Target Network 可有可无

不对。

正确说法是：

Target Network 是 DQN 稳定训练的核心机制之一，用来避免目标值跟着学习网络一起剧烈变化。

### 错误 5：`done` 只是表示一局结束，对更新没什么影响

不对。

正确说法是：

`done` 会直接决定目标值里是否保留未来收益，因此对更新非常重要。

## 21. 之后还可以继续补充的内容

等我后面继续学强化学习时，可以把这篇笔记继续补成：

- Double DQN 为什么能缓解 Q 值高估
- Dueling DQN 在结构上改了什么
- Prioritized Replay 为什么能进一步提升采样效率
- 从 DQN 过渡到 policy-based 和 actor-critic


