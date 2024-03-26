import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DuelingDoubleDeepQNetwork(nn.Module):

    def __init__(self, n_actions, n_features, n_lstm_features, n_time, learning_rate=0.01,
                 reward_decay=0.9, e_greedy=0.99, replace_target_iter=200, memory_size=500,
                 batch_size=32, e_greedy_increment=0.00025, n_lstm_step=10, dueling=True,
                 double_q=True, hidden_units_l1=20, N_lstm=20):

        super(DuelingDoubleDeepQNetwork, self).__init__()

        self.n_actions = n_actions  # 动作数量
        self.n_features = n_features  # 网络特征数
        self.n_time = n_time  # 时间片
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 折扣率
        self.epsilon_max = e_greedy  # 最大利用概率
        self.replace_target_iter = replace_target_iter  # target network 更新间隔
        self.memory_size = memory_size  # 内存最大容量
        self.batch_size = batch_size  # batch容量
        self.epsilon_increment = e_greedy_increment  # 利用概率增长率
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 利用概率
        self.dueling = dueling  # 是否用dueling network
        self.double_q = double_q  # 是否用double q network
        self.learn_step_counter = 0  # 学习步
        self.hidden_units_l1 = hidden_units_l1  # 隐藏层维度

        # lstm
        self.N_lstm = N_lstm  # lstm隐藏层维度
        self.n_lstm_step = n_lstm_step  # lstm步长
        self.n_lstm_state = n_lstm_features  # lstm状态数

        # 回放缓存
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                + self.n_features + self.n_lstm_state + self.n_lstm_state))

        self._build_net()  # 构建网络

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)  # 优化器
        self.loss_func = nn.MSELoss()  # 损失函数

        self.reward_store = list()  # 记录奖励
        self.action_store = list()  # 记录动作
        self.delay_store = list()   # 记录延迟
        self.energy_store = list()  # 记录能量

        self.lstm_history = deque(maxlen=self.n_lstm_step)  # 历史lstm，最大长度为lstm步长的双向队列
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()  # 记录q值

    def _build_net(self):
        # Build the neural network model 构建神经网络模型
        hidden_units_l1 = self.hidden_units_l1
        N_lstm = self.N_lstm

        # LSTM layer for load levels 构建lstm层来导入
        self.lstm_dnn = nn.LSTM(self.n_lstm_state, N_lstm, batch_first=True)

        # Common layers 全连接层
        self.fc1 = nn.Linear(N_lstm + self.n_features, hidden_units_l1)
        self.fc2 = nn.Linear(hidden_units_l1, hidden_units_l1)

        if self.dueling:
            # Dueling DQN
            # Value stream 价值网络
            self.value = nn.Linear(hidden_units_l1, 1)
            # Advantage stream 优势网络
            self.advantage = nn.Linear(hidden_units_l1, self.n_actions)
        else:
            self.q = nn.Linear(hidden_units_l1, self.n_actions)

    def forward(self, s, lstm_s):
        # Forward pass of the network

        lstm_output, _ = self.lstm_dnn(lstm_s)
        lstm_output_reduced = lstm_output[:, -1, :]

        x = torch.cat((lstm_output_reduced, s), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        if self.dueling:
            value = self.value(x)
            advantage = self.advantage(x)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q = self.q(x)

        return q

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)

        if np.random.uniform() < self.epsilon:
            lstm_observation = torch.tensor(np.array(self.lstm_history), dtype=torch.float).unsqueeze(0)
            actions_value = self.forward(observation, lstm_observation)
            self.store_q_value.append({'observation': observation, 'q_value': actions_value})

            action = torch.argmax(actions_value, dim=1).item()
        else:
            if np.random.randint(0, 100) < 25:
                action = np.random.randint(1, self.n_actions)
            else:
                action = 0

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            # No target network in PyTorch, this step can be omitted.
            print('\ntarget_params_replaced')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])

        for i in range(len(sample_index)):
            for j in range(self.n_lstm_step):
                lstm_batch_memory[i, j, :] = self.memory[sample_index[i] + j, self.n_features + 1 + 1 + self.n_features:]

        batch_memory = torch.tensor(batch_memory, dtype=torch.float)
        lstm_batch_memory = torch.tensor(lstm_batch_memory, dtype=torch.float)

        q_next, q_eval4next = (self.forward(batch_memory[:, -self.n_features:], lstm_batch_memory[:, :, self.n_lstm_state:]),
                               self.forward(batch_memory[:, -self.n_features:], lstm_batch_memory[:, :, self.n_lstm_state:]))
        q_eval = self.forward(batch_memory[:, :self.n_features], lstm_batch_memory[:, :, :self.n_lstm_state])

        q_target = q_eval.clone().detach()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].long()
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = torch.argmax(q_eval4next, dim=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next, _ = torch.max(q_next, dim=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
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

    def Initialize(self, iot):
        latest_model_path = f"./models/500/{iot}_X_model.pth"
        self.load_state_dict(torch.load(latest_model_path))

    def save_model(self, iot):
        model_path = f"./models/500/{iot}_X_model.pth"
        torch.save(self.state_dict(), model_path)
