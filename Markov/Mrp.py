import numpy as np
np.random.seed(0)
# 这个过程既是忽略过程中的动作选择，也就是每一步的价值仅取决于当前状态，而与动作无关，因此是一个马尔可夫奖励过程（MRP）。
# 定义状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = np.array([-1, -2, -2, 10, 1, 0]) # 定义奖励函数
gamma = 0.5

def compute_return(start_index, chain, gamma):
    """计算从start_index开始的回报"""
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = rewards[chain[i] - 1] + gamma * G
    return G

def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数 '''
    rewards = np.array(rewards).reshape((-1, 1))  #将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards)
    return value



# 一个状态序列,s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到回报为：%s。" % G)

V = compute(P, rewards, gamma, states_num=6)
print("MRP中每个状态价值分别为\n", V)