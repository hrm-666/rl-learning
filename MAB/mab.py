import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)      # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.k = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(bandit.k) #每根拉杆的尝试次数
        self.regret = 0 #当前步的累计懊悔值
        self.actions = [] #每一步选择的动作
        self.regrets = [] #每一步的累计懊悔值

    def update_regret(self, k):
        #计算累计懊悔值，k为本次动作选择的拉杆编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)




class EpsilonGreedy(Solver):
    """ε-贪婪算法,继承Solver类"""
    def __init__(self, bandit, epsilon=0.01, init_prob = 1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * bandit.k) #每根拉杆的奖励估计值,初始值为init_prob

    def  run_one_step(self):
        if np.random.rand() <self.epsilon:
            k = np.random.randint(0,self.bandit.k) #以ε的概率随机选择一个拉杆
        else:
            k = np.argmax(self.estimates) #以1-ε的概率选择奖励估计值最大的拉杆
        r= self.bandit.step(k) #本轮拉动选择的拉杆,获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) #更新奖励估计值
        return k
    
class DecayEpsilonGreedy(Solver):
    """衰减ε-贪婪算法,继承Solver类,具体衰减形式为反比例衰减"""
    def __init__(self, bandit,  init_prob = 1.0):
        super(DecayEpsilonGreedy,self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.k) #每根拉杆的奖励估计值,初始值为init_prob
        self.total_count =0

    def run_one_step(self):
        self.total_count += 1
        epsilon = 1. / self.total_count #反比例衰减
        if np.random.random() < epsilon:
            k = np.random.randint(0,self.bandit.k) #以ε的概率随机选择一个拉杆
        else:
            k = np.argmax(self.estimates) #以1-ε的概率选择奖励估计值最大的拉杆
        r= self.bandit.step(k) #本轮拉动选择的拉杆,获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) #更新奖励估计值
        return k

class UCB(Solver):
    """上置信界算法，又称UCB算法"""
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.k) #每根拉杆的奖励估计值,初始值为init_prob
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1))) #计算每根拉杆的上置信界值
        k = np.argmax(ucb) #选择上置信界值最大的拉杆
        r = self.bandit.step(k) #本轮拉动选择的拉杆,获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) #更新奖励估计值
        return k

class ThompsonSampling(Solver):
    """汤普森采样算法"""
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.a = np.ones(bandit.k) #每根拉杆的成功次数
        self.b = np.ones(bandit.k) #每根拉杆的失败次数

    def run_one_step(self):
        samples = np.random.beta(self.a, self.b) #从每根拉杆的Beta分布中采样一个值
        k = np.argmax(samples) #选择采样值最大的拉杆
        r = self.bandit.step(k) #本轮拉动选择的拉杆,获得奖励

        self.a[k] += r #更新成功次数
        self.b[k] += 1 - r #更新失败次数
        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])

    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title('%d-armed bandit' % solvers[0].bandit.k)
    plt.legend()
    plt.show()



np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
# print("随机生成了一个%d臂的Bernoulli Bandit" % K)
# print("获奖概率最大的臂是%d，获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-greedy算法的累计懊悔值为%.4f' % epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ['Epsilon-Greedy'])

decay_epsilon_greedy_solver = DecayEpsilonGreedy(bandit_10_arm)
decay_epsilon_greedy_solver.run(5000)
print('decay epsilon-greedy算法的累计懊悔值为%.4f' % decay_epsilon_greedy_solver.regret)
plot_results([decay_epsilon_greedy_solver], ['Decay Epsilon-Greedy'])

coef = 1.0
ucb_solver = UCB(bandit_10_arm, coef=coef)
ucb_solver.run(5000)
print('UCB算法的累计懊悔值为%.4f' % ucb_solver.regret)
plot_results([ucb_solver], ['UCB'])

thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)  
print('Thompson Sampling算法的累计懊悔值为%.4f' % thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ['Thompson Sampling'])
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [
#     EpsilonGreedy(bandit_10_arm, epsilon=eps) for eps in epsilons
# ]
# epsilon_greedy_solver_names = ["epsilon={}".format(eps) for eps in epsilons]
# for solver in epsilon_greedy_solver_list:
#     solver.run(5000)

# plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
# 根据图标观察可以很明显的看出来，累计懊悔值都是线性增长的

