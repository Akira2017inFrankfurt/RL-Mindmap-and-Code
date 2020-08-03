import gym
import time
import numpy as np
from gridworld import FrozenLakeWapper


class Agent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1, tag="Sarsa"):
        self.act_n = act_n  # 动作维度，也就是有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))  # Q表格
        self.flag = tag  # 判断使用那种算法，Sara还是Qlearning

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        """
        在predict方法基础上使用e-greedy增加搜索
        """
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
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

    def learn(self, obs, action, reward, next_obs, done, next_action=None):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            if next_action is not None:
                target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
            else:
                target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])  # Q_learning
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    def save(self):
        # 保存Q表格数据到文件
        npy_file = './xq_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./xq_table.npy'):
        # 从文件中读取Q值到Q表格中
        self.Q = np.load(npy_file)
        print(npy_file + ' load.')


def run_episode(env, agent, is_render, flag):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    if flag == "Sarsa":
        action = agent.sample(obs)  # 根据算法选择一个动作

    while True:
        if flag == "Sarsa":
            # train Sarsa algorithm
            next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
            next_action = agent.sample(next_obs)  # 根据算法选择一个动作
            agent.learn(obs, action, reward, next_obs, done, next_action)
            action = next_action
        else:
            # train Qlearning algorithm
            action = agent.sample(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1
        if is_render:
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            break
    return total_reward


env = gym.make("FrozenLake-v0", is_slippery=False)
env = FrozenLakeWapper(env)
env.reset()
agent = Agent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    learning_rate=0.1,
    gamma=0.9,
    e_greed=0.1
)

flag = input("Sarsa or Qlearning? Give me your choice, S stands for Sarsa and Q for Qlearning: ")
if flag == "Q":
    flag = "Qlearning"
elif flag == "S":
    flag = "Sarsa"
else:
    print("Sorry Sir, choose S or Q please~ ")

print("Your algorithm is: ", flag)
is_render = False
for episode in range(500):
    ep_reward, ep_steps = run_episode(env, agent, is_render, flag)
    print('Episode %s: step = %s, reward = %.1f' % (episode, ep_steps, ep_reward))
    if episode % 20 == 0:
        is_render = True
    else:
        is_render = False

test_reward = test_episode(env, agent)
print('test reward = %.1f' % test_reward)
env.close()
