import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

state_size = env.observation_space.n
action_size = env.action_space.n

Q = np.zeros((state_size, action_size))

alpha = 0.1             # 学习率
gamma = 0.99            # 折扣因子
epsilon = 1.0           # 初始探索率，训练初期更多地探索环境
epsilon_min = 0.01      # 最小探索率，确保在训练后期仍有一定的探索
epsilon_decay = 0.995   # 探索率衰减，每完成一个回合后，探索率会逐渐降低
episodes = 20000        # 训练的总回合数

success_count = 0

def greedy_action(Q, state):
    max_value = np.max(Q[state])
    max_actions = np.where(Q[state] == max_value)[0]
    return np.random.choice(max_actions)

for episode in range(episodes):
    state, info = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = greedy_action(Q, state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        old_value = Q[state, action]
        target = reward + gamma * np.max(Q[next_state])
        Q[state, action] = old_value + alpha * (target - old_value)

        state = next_state

        if reward == 1:
            success_count += 1

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("训练完成后的 Q 表：")
print(Q)

print(f"\n成功到达终点的次数: {success_count}/{episodes}")

print("\n每个状态下选择的最优动作:")
for s in range(state_size):
    print(f"state {s}: best action = {np.argmax(Q[s])}")