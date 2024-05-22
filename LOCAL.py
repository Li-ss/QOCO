import numpy as np


class LocalModel():
    def __init__(self, n_time):

        super(LocalModel, self).__init__()
        self.n_time = n_time  # 时间片
        self.reward_store = list()  # 记录奖励
        self.action_store = list()  # 记录动作
        self.delay_store = list()   # 记录延迟
        self.energy_store = list()  # 记录能量

    def choose_action(self, observation):
        action = 0
        return action

    def do_store_reward(self, episode, time, reward):
        # 记录奖励
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        # 记录动作
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        # 记录延迟
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
        # 记录总能量
        fog_energy = 0
        for i in range(len(energy3)):
            if energy3[i] != 0:
                fog_energy = energy3[i]

        idle_energy = 0
        for i in range(len(energy4)):
            if energy4[i] != 0:
                idle_energy = energy4[i]

        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy


