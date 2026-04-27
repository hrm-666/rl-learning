import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# 把你训练好的 Q 表复制到这里
Q = np.array([
    [0.93629555, 0.95099005, 0.92806781, 0.93585494],
    [0.94103457, 0.,         0.,         0.        ],
    [0.,         0.,         0.,         0.        ],
    [0.,         0.,         0.,         0.        ],
    [0.94800393, 0.96059601, 0.,         0.93814698],
    [0.,         0.,         0.,         0.        ],
    [0.,         0.0546296,  0.,         0.        ],
    [0.,         0.,         0.,         0.        ],
    [0.95883431, 0.,         0.970299,   0.94941275],
    [0.94905072, 0.9801,     0.97587444, 0.        ],
    [0.,         0.98998956, 0.,         0.00246654],
    [0.,         0.,         0.,         0.        ],
    [0.,         0.,         0.,         0.        ],
    [0.,         0.97737746, 0.99,       0.96949256],
    [0.97631043, 0.9847757,  1.,         0.96942807],
    [0.,         0.,         0.,         0.        ]
])

state, info = env.reset()
done = False
total_reward = 0

action_map = {
    0: "左",
    1: "下",
    2: "右",
    3: "上"
}

print("开始测试训练好的策略...\n")

while not done:
    action = np.argmax(Q[state])
    print(f"当前状态: {state}, 选择动作: {action}({action_map[action]})")

    next_state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    print(f"到达状态: {next_state}, reward = {reward}\n")

    state = next_state
    time.sleep(0.5)

print("测试结束")
print("总奖励:", total_reward)

env.close()