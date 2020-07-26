"""
使用Sarsa解决悬崖问题，找到绕过悬崖通往终点的路径
Author:  Haruki
Time: 2020.07.26
"""

import gym
import numpy as np


class SarsaAgent(object):
    """
    Agent是交互的主体
    """
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n  # 动作维度，也就是有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

 # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        """
        在predict方法基础上使用e-greedy增加搜索
        """
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  #根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)   #有一定概率随机探索选取一个动作
        return action

    def predict(self, obs):
        """
        输入观察值observation/state 输出动作值
        """
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """
        输入训练数据，完成一轮Q表格的更新
            on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    def save(self):
        # 保存Q表格数据到文件
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./q_table.npy'):
         # 从文件中读取Q值到Q表格中
        self.Q = np.load(npy_file)
        print(npy_file + ' load.')


def run_episode(env, agent, render=False):
    # agent在一个episode中训练的过程，使用agent.sample()与环境交互，使用agent.learn()训练Q表格。
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    action = agent.sample(obs)  # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        next_action = agent.sample(next_obs)   # 根据算法选择一个动作

        # train Sarsa algorithm
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps +=1
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    # agent在一个episode中测试效果，评估目前的agent能在一个episode中拿到多少总reward。
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        # time.sleep(0.5)
        # env.render()
        if done:
            break
    return total_reward


# 使用gym创建悬崖环境
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

# 创建一个agent实例，输入超参数
agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)


# 训练500个episode，打印每个episode的分数
for episode in range(500):
    ep_reward, ep_steps = run_episode(env, agent, False)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# 全部训练结束，查看算法效果
test_reward = test_episode(env, agent)
print('test reward = %.1f' % test_reward)
