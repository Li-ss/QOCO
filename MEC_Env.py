from Config import Config
import numpy as np
import random
import math
import queue

class MEC:
    def __init__(self, num_ue, num_edge, num_time, num_component, max_delay):
        # Initialize variables
        # 初始化变量
        self.n_ue          = num_ue           # 用户数量
        self.n_edge        = num_edge         # 边缘节点数量
        self.n_time        = num_time         # 时间片
        self.n_component   = num_component    # 组件数量
        self.max_delay     = max_delay        # 最大延迟
        self.duration      = Config.DURATION  # 持续时间
        self.ue_p_comp     = Config.UE_COMP_ENERGY    # 用户计算功率
        self.ue_p_tran     = Config.UE_TRAN_ENERGY    # 用户和边缘间传输功率
        self.ue_p_idle     = Config.UE_IDLE_ENERGY    # 用户待机功率
        self.edge_p_comp   = Config.EDGE_COMP_ENERGY  # 边缘节点计算功率

        self.time_count      = 0  # 时间步
        self.task_count_ue   = 0  # 用户任务数
        self.task_count_edge = 0  # 边缘节点任务数
        self.n_actions       = self.n_component + 1     # 动作数量
        self.n_features      = 1 + 1 + 1 + self.n_edge  # 网络特征数
        self.n_lstm_state    = self.n_edge              # lstm状态数
  
        # Computation and transmission capacities
        # 计算和传输容量
        self.comp_cap_ue   = Config.UE_COMP_CAP * np.ones(self.n_ue) * self.duration                 # 用户计算容量
        self.comp_cap_edge = Config.EDGE_COMP_CAP * np.ones([self.n_edge]) * self.duration           # 边缘节点计算容量
        self.tran_cap_ue   = Config.UE_TRAN_CAP * np.ones([self.n_ue, self.n_edge]) * self.duration  # 用户和边缘间传输容量
        self.comp_density  = Config.TASK_COMP_DENS * np.ones([self.n_ue])                            # 计算任务密度

        self.n_cycle = 1  # 周期数量
        self.task_arrive_prob = Config.TASK_ARRIVE_PROB  # 任务到达概率
        self.max_arrive_size   = Config.TASK_MAX_SIZE    # 最大任务数量
        self.min_arrive_size   = Config.TASK_MIN_SIZE    # 最小任务数量
        self.arrive_task_set    = np.arange(self.min_arrive_size, self.max_arrive_size, 0.1)  # 到达任务集
        self.arrive_task        = np.zeros([self.n_time, self.n_ue])  # 到达任务
        self.n_task = int(self.n_time * self.task_arrive_prob)  # 任务数量

        # Task delay and energy-related arrays
        # 任务延迟和能量相关数组
        self.process_delay = np.zeros([self.n_time, self.n_ue])  # 处理延迟
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])                 # 用户处理bit
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])  # 边缘节点处理bit
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])               # 用户传输bit
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])                 # 用户计算能量
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])  # 边缘节点计算能量
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])    # 用户待机能量
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])                 # 用户和边缘间传输能量
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])  # 未完成任务标记
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])  # 传输等待延迟
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])  # 边缘丢弃bit

        # Queue information initialization
        # 队列信息初始化
        self.t_ue_comp = -np.ones([self.n_ue])  # 用户下一任务开始计算时间
        self.t_ue_tran = -np.ones([self.n_ue])  # 用户下一任务开始传输时间
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])  # 边缘计算队列剩余的bit

        # Queue initialization
        # 队列初始化
        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]   # 用户计算队列
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]  # 用户传输队列
        self.edge_computation_queue = [[queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)]  # 边缘计算队列
        self.edge_ue_m = np.zeros(self.n_edge)  # 下一时间步活跃计算队列数量
        self.edge_ue_m_observe = np.zeros(self.n_edge)  # 观测到的活跃计算队列数量

        # Task indicator initialization 任务指示器初始化
        # 本地处理任务
        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        # 本地传输任务
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        # 边缘处理任务
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        # 历史任务
        self.task_history = [[] for _ in range(self.n_ue)]

    def reset(self, arrive_task):
        # Reset variables and queues
        # 重置变量和队列
        self.task_history = [[] for _ in range(self.n_ue)]
        self.UE_TASK = [-1] * self.n_ue
        self.drop_edge_count = 0
        self.arrive_task = arrive_task
        self.time_count = 0
        self.local_process_task = []
        self.local_transmit_task = []
        self.edge_process_task = []
        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])
        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.edge_computation_queue = [[queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)]
        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])
        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        # Initial observation and LSTM state
        # 初始化观测状态变量
        UEs_OBS = np.zeros([self.n_ue, self.n_features])
        for ue_index in range(self.n_ue):
            if self.arrive_task[self.time_count, ue_index] != 0:
                UEs_OBS[ue_index, :] = np.hstack([
                    self.arrive_task[self.time_count, ue_index], self.t_ue_comp[ue_index],
                    self.t_ue_tran[ue_index],
                    np.squeeze(self.b_edge_comp[ue_index, :])])

        UEs_lstm_state = np.zeros([self.n_ue, self.n_lstm_state])

        return UEs_OBS, UEs_lstm_state

    # perform action, observe state and delay (several steps later)
    # 执行动作，观测状态变量和延迟（几步之后）
    def step(self, action):
    
        # EXTRACT ACTION FOR EACH ue
        # 提取各个用户的动作
        ue_action_local = np.zeros([self.n_ue], np.int32)
        ue_action_offload = np.zeros([self.n_ue], np.int32)
        # ue_action_edge = np.zeros([self.n_ue], np.int32)
        ue_action_component = np.zeros([self.n_ue], np.int32)-1

        random_list  = []
        for i in range(self.n_component):
            random_list.append(i)

        # UE QUEUES UPDATE
        # 更新用户队列
        for ue_index in range(self.n_ue):
            component_list = np.zeros([self.n_component], np.int32) - 1
            state_list = np.zeros([self.n_component], np.int32)
            ue_action = action[ue_index]
            
            if ue_action == 0:
                # 本地计算
                ue_action_local[ue_index] = 1
            else:
                # 卸载到边缘
                ue_action_offload[ue_index] = 1
                sample = random.sample(random_list, int(ue_action))
                for i in range(len(sample)):
                    component_list[sample[i]] = np.random.randint(0, self.n_edge)
            
            ue_action_component[ue_index] = action[ue_index]
            ue_comp_cap = np.squeeze(self.comp_cap_ue[ue_index])
            ue_comp_density = self.comp_density[ue_index]
            ue_tran_cap = np.squeeze(self.tran_cap_ue[ue_index, :])[1] / self.n_cycle
            ue_arrive_task = np.squeeze(self.arrive_task[self.time_count, ue_index])
            
            if ue_arrive_task > 0:
                # 将任务信息记录到历史任务
                self.UE_TASK[ue_index] += 1
                task_dic = {
                    'UE_ID': ue_index,
                    'TASK_ID': self.UE_TASK[ue_index],
                    'SIZE': ue_arrive_task,
                    'TIME': self.time_count,
                    'EDGE': component_list,
                    'd_state': state_list,
                    'state': np.nan
                }
                self.task_history[ue_index].append(task_dic)
            
            for component in range(self.n_component):
                # 将任务分成component份，每份都可能本地计算或卸载到边缘，action的值代表卸载到边缘的份数
                temp_dic = {
                    'DIV': component,
                    'UE_ID': ue_index,
                    'TASK_ID': self.UE_TASK[ue_index],
                    'SIZE': ue_arrive_task / self.n_component,
                    'TIME': self.time_count,
                    'EDGE': component_list[component],
                    'd_state': state_list[component]
                }
                
                if component_list[component] > -1:
                    self.ue_transmission_queue[ue_index].put(temp_dic)
                else:
                    self.ue_computation_queue[ue_index].put(temp_dic)
            
            for cycle in range(self.n_cycle):
                # 每一步迭代cycle轮
                ue_comp_cap = np.squeeze(self.comp_cap_ue[ue_index]) / self.n_cycle

                if ((math.isnan(self.local_process_task[ue_index]['REMAIN']) and (not self.ue_computation_queue[ue_index].empty())) or
                    (math.isnan(self.local_transmit_task[ue_index]['REMAIN']) and (not self.ue_transmission_queue[ue_index].empty()))):

                    # Process UE computation queue
                    # 处理用户计算队列
                    if not self.ue_computation_queue[ue_index].empty():
                        # 计算队列非空，尝试取出需要计算的任务到本地计算
                        while not self.ue_computation_queue[ue_index].empty():
                            task = self.ue_computation_queue[ue_index].get()
                            if task['SIZE'] != 0:
                                if self.time_count - task['TIME'] + 1 <= self.max_delay:
                                    # 取出任务等待时间小于最大延迟，进行本地处理
                                    self.local_process_task[ue_index].update({
                                        'UE_ID': task['UE_ID'],
                                        'TASK_ID': task['TASK_ID'],
                                        'SIZE': task['SIZE'],
                                        'TIME': task['TIME'],
                                        'REMAIN': task['SIZE'],
                                        'DIV': task['DIV'],
                                    })
                                    break
                                else:
                                    self.task_history[ue_index][task['TASK_ID']]['d_state'][task['DIV']] = -1
                                    self.process_delay[task['TIME'], ue_index] = self.max_delay
                                    self.unfinish_task[task['TIME'], ue_index] = 1
                    
                    # Process UE transmission queue
                    # 处理用户传输队列
                    if not self.ue_transmission_queue[ue_index].empty():
                        # 传输队列非空，尝试取出需要传输的任务传输到边缘
                        while not self.ue_transmission_queue[ue_index].empty():
                            task = self.ue_transmission_queue[ue_index].get()
                            if task['SIZE'] != 0:
                                if self.time_count - task['TIME'] + 1 <= self.max_delay:
                                    # 取出任务等待时间小于最大延迟，进行传输
                                    self.local_transmit_task[ue_index].update({
                                        'UE_ID': task['UE_ID'],
                                        'TASK_ID': task['TASK_ID'],
                                        'SIZE': task['SIZE'],
                                        'TIME': task['TIME'],
                                        'EDGE': int(task['EDGE']),
                                        'REMAIN': self.local_transmit_task[ue_index]['SIZE'],
                                        'DIV': task['DIV'],
                                    })
                                    break
                                else:
                                    # 取出任务等待时间超过最大延迟，记录状态和延迟
                                    self.task_history[task['UE_ID']][task['TASK_ID']]['d_state'][task['DIV']] = -1
                                    self.process_delay[task['TIME'], ue_index] = self.max_delay
                                    self.unfinish_task[task['TIME'], ue_index] = 1

                # PROCESS
                # 处理本地计算的任务，更新用户处理的总bit和能量，本地任务减去处理的bit数，
                if self.local_process_task[ue_index]['REMAIN'] > 0:
                    if self.local_process_task[ue_index]['REMAIN'] >= (ue_comp_cap / ue_comp_density):
                        self.ue_bit_processed[self.local_process_task[ue_index]['TIME'], ue_index] += ue_comp_cap / ue_comp_density
                        self.ue_comp_energy[self.local_process_task[ue_index]['TIME'], ue_index] += (
                            (ue_comp_cap / ue_comp_density) * self.ue_p_comp * ue_comp_density
                        ) / (self.comp_cap_ue[ue_index])
                    else:
                        self.ue_bit_processed[self.local_process_task[ue_index]['TIME'], ue_index] += self.local_process_task[ue_index]['REMAIN']
                        self.ue_comp_energy[self.local_process_task[ue_index]['TIME'], ue_index] += (
                            self.local_process_task[ue_index]['REMAIN'] * self.ue_p_comp * ue_comp_density
                        ) / (self.comp_cap_ue[ue_index])

                    self.local_process_task[ue_index]['REMAIN'] = self.local_process_task[ue_index]['REMAIN'] - (ue_comp_cap / ue_comp_density)
                
                    # if no remain or reach the max delay, compute and record processing delay
                    # 如果任务完成或超过最大延迟，计算并记录延迟
                    if self.local_process_task[ue_index]['REMAIN'] <= 0:
                        self.task_history[ue_index][self.local_process_task[ue_index]['TASK_ID']]['d_state'][self.local_process_task[ue_index]['DIV']] = 1
                        self.local_process_task[ue_index]['REMAIN'] = np.nan
                        if sum(self.task_history[ue_index][self.local_process_task[ue_index]['TASK_ID']]['d_state']) > self.n_component - 1:
                            self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] = self.time_count - self.local_process_task[ue_index]['TIME'] + 1
                    elif self.time_count - self.local_process_task[ue_index]['TIME'] + 1 == self.max_delay:
                        self.task_history[ue_index][self.local_process_task[ue_index]['TASK_ID']]['d_state'][self.local_process_task[ue_index]['DIV']] = -1
                        self.local_process_task[ue_index]['REMAIN'] = np.nan
                        self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] = self.max_delay
                        self.unfinish_task[self.local_process_task[ue_index]['TIME'], ue_index] = 1
                
                # 处理传输的任务，更新用户传输的总bit和能量，传输任务减去传输的bit数
                if self.local_transmit_task[ue_index]['REMAIN'] > 0:
                    if self.local_transmit_task[ue_index]['REMAIN'] >= ue_tran_cap:
                        self.ue_bit_transmitted[self.local_transmit_task[ue_index]['TIME'], ue_index] += ue_tran_cap
                        self.ue_tran_energy[self.local_transmit_task[ue_index]['TIME'], ue_index] += (self.local_transmit_task[ue_index]['REMAIN'] * self.ue_p_tran) / self.tran_cap_ue[0][0]
                    else:
                        self.ue_bit_transmitted[self.local_transmit_task[ue_index]['TIME'], ue_index] += self.local_transmit_task[ue_index]['REMAIN']
                        self.ue_tran_energy[self.local_transmit_task[ue_index]['TIME'], ue_index] += (self.local_transmit_task[ue_index]['REMAIN'] * self.ue_p_tran) / self.tran_cap_ue[0][0]
                    self.local_transmit_task[ue_index]['REMAIN'] = self.local_transmit_task[ue_index]['REMAIN'] - ue_tran_cap 

                    # UPDATE edge QUEUE
                    # 更新边缘队列
                    if self.local_transmit_task[ue_index]['REMAIN'] <= 0:
                        # 任务完成传输，将任务信息放入边缘计算队列中，更新边缘计算队列bit，计算任务传输延迟
                        tmp_dict = {
                            'UE_ID': self.local_transmit_task[ue_index]['UE_ID'],
                            'TASK_ID': self.local_transmit_task[ue_index]['TASK_ID'],
                            'SIZE': self.local_transmit_task[ue_index]['SIZE'],
                            'TIME': self.local_transmit_task[ue_index]['TIME'],
                            'EDGE': self.local_transmit_task[ue_index]['EDGE'],
                            'DIV': self.local_transmit_task[ue_index]['DIV']
                        }
                        self.edge_computation_queue[ue_index][self.local_transmit_task[ue_index]['EDGE']].put(tmp_dict)
                        self.task_count_edge = self.task_count_edge + 1
                        edge_index = self.local_transmit_task[ue_index]['EDGE']
                        self.b_edge_comp[ue_index, edge_index] = self.b_edge_comp[ue_index, edge_index] + self.local_transmit_task[ue_index]['SIZE']
                        self.process_delay_trans[self.local_transmit_task[ue_index]['TIME'], ue_index] = self.time_count - self.local_transmit_task[ue_index]['TIME'] + 1
                        self.local_transmit_task[ue_index]['REMAIN'] = np.nan
                    elif self.time_count - self.local_transmit_task[ue_index]['TIME'] + 1 == self.max_delay:
                        # 如果任务超过最大延迟，计算并记录延迟
                        self.task_history[self.local_transmit_task[ue_index]['UE_ID']][self.local_transmit_task[ue_index]['TASK_ID']]['d_state'][self.local_transmit_task[ue_index]['DIV']] = -1
                        self.local_transmit_task[ue_index]['REMAIN'] = np.nan
                        self.process_delay[self.local_transmit_task[ue_index]['TIME'], ue_index] = self.max_delay
                        self.unfinish_task[self.local_transmit_task[ue_index]['TIME'], ue_index] = 1
            
            if ue_arrive_task != 0:
                # 计算前等待时间
                tmp_tilde_t_ue_comp = max(self.t_ue_comp[ue_index] + 1, self.time_count)
                comp_time = math.ceil(ue_arrive_task * ue_action_local[ue_index] / (np.squeeze(self.comp_cap_ue[ue_index]) / ue_comp_density))
                # 执行（计算或丢弃）时间
                self.t_ue_comp[ue_index] = min(tmp_tilde_t_ue_comp + comp_time - 1, self.time_count + self.max_delay - 1)
                # 传输前等待时间
                tmp_tilde_t_ue_tran = max(self.t_ue_tran[ue_index] + 1, self.time_count)
                tran_time = math.ceil(ue_arrive_task * (1 - ue_action_local[ue_index]) / np.squeeze(self.tran_cap_ue[ue_index,:])[1])
                # 执行（传输或丢弃）时间
                self.t_ue_tran[ue_index] = min(tmp_tilde_t_ue_tran + tran_time - 1, self.time_count + self.max_delay - 1)

        # EDGE QUEUES UPDATE
        # 更新边缘队列
        for ue_index in range(self.n_ue):
            ue_comp_density = self.comp_density[ue_index]
            for edge_index in range(self.n_edge):
                edge_cap = self.comp_cap_edge[edge_index] / self.n_cycle
                for cycle in range(self.n_cycle): 
                    # TASK ON PROCESS
                    # 处理边缘计算队列
                    if math.isnan(self.edge_process_task[ue_index][edge_index]['REMAIN']) and (not self.edge_computation_queue[ue_index][edge_index].empty()):
                        # 边缘计算队列非空，尝试取出需要计算的任务到边缘计算
                        while not self.edge_computation_queue[ue_index][edge_index].empty():
                            task = self.edge_computation_queue[ue_index][edge_index].get()
                            if self.time_count - task['TIME'] + 1 <= self.max_delay:
                                # 取出任务等待时候小于最大延迟，进行边缘处理
                                self.edge_process_task[ue_index][edge_index].update({
                                    'UE_ID': task['UE_ID'],
                                    'TASK_ID': task['TASK_ID'],
                                    'SIZE': task['SIZE'],
                                    'TIME': task['TIME'],
                                    'REMAIN': self.edge_process_task[ue_index][edge_index]['SIZE'],
                                    'DIV': task['DIV'],
                                })
                                break
                            else:
                                # 取出任务等待时间大于最大延迟，记录状态和延迟
                                self.task_history[task['UE_ID']][task['TASK_ID']]['d_state'][task['DIV']] = -1
                                self.process_delay[task['TIME'], ue_index] = self.max_delay
                                self.unfinish_task[task['TIME'], ue_index] = 1

                    # PROCESS
                    # 处理边缘计算任务
                    self.edge_drop[ue_index, edge_index] = 0  # 记录边缘本次丢弃任务的bit
                    remaining_task = self.edge_process_task[ue_index][edge_index]['REMAIN']
                    if remaining_task > 0:
                        # 更新边缘处理的总bit和能量，计算任务减去处理的bit数
                        processed_amount = min(remaining_task, edge_cap / ue_comp_density / self.edge_ue_m[edge_index])
                        self.edge_bit_processed[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += processed_amount
                        comp_energy = (self.edge_p_comp * processed_amount * ue_comp_density * pow(10, 9)) / (edge_cap * 10 * pow(10, 9))
                        idle_energy = (self.ue_p_idle * processed_amount * ue_comp_density * pow(10, 9)) / (edge_cap * 10 * pow(10, 9) / self.edge_ue_m[edge_index])
                        self.edge_comp_energy[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += comp_energy
                        self.ue_idle_energy[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += idle_energy
                        self.edge_process_task[ue_index][edge_index]['REMAIN'] -= processed_amount
                        if self.edge_process_task[ue_index][edge_index]['REMAIN'] <= 0:
                            # if no remain, compute processing delay
                            # 如果任务完成，更新任务状态，计算并记录延迟
                            task_id = self.edge_process_task[ue_index][edge_index]['TASK_ID']
                            task_history = self.task_history[self.edge_process_task[ue_index][edge_index]['UE_ID']][task_id]
                            task_history['d_state'][self.edge_process_task[ue_index][edge_index]['DIV']] = 1
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan
                            if sum(task_history['d_state']) > self.n_component - 1:
                                self.process_delay[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = self.time_count - self.edge_process_task[ue_index][edge_index]['TIME'] + 1
                        elif self.time_count - self.edge_process_task[ue_index][edge_index]['TIME'] + 1 == self.max_delay:
                            # 如果任务超过最大延迟，更新任务状态，计算并处理延迟
                            task_id = self.edge_process_task[ue_index][edge_index]['TASK_ID']
                            task_history = self.task_history[self.edge_process_task[ue_index][edge_index]['UE_ID']][task_id]
                            task_history['d_state'][self.edge_process_task[ue_index][edge_index]['DIV']] = -1
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan
                            self.edge_drop[ue_index, edge_index] = remaining_task
                            self.process_delay[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = self.max_delay
                            self.unfinish_task[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = 1

                # OTHER INFO
                if self.edge_ue_m[edge_index] != 0:
                    # 计算环境转移后，当前edge中还有多少bit需要计算处理
                    b_edge_comp_value = self.b_edge_comp[ue_index, edge_index]
                    b_edge_comp_value -= (edge_cap / ue_comp_density / self.edge_ue_m[edge_index] + self.edge_drop[ue_index, edge_index])
                    self.b_edge_comp[ue_index, edge_index] = max(b_edge_comp_value, 0)

        # COMPUTE CONGESTION (FOR NEXT TIME SLOT)
        # 活跃队列
        self.edge_ue_m_observe = self.edge_ue_m
        self.edge_ue_m = np.zeros(self.n_edge)
        for edge_index in range(self.n_edge):
            for ue_index in range(self.n_ue):
                if (not self.edge_computation_queue[ue_index][edge_index].empty()) \
                        or self.edge_process_task[ue_index][edge_index]['REMAIN'] > 0:
                    # 记录下一时间步每个edge分别有多少个用户要使用
                    self.edge_ue_m[edge_index] += 1

        # TIME UPDATE
        # 更新时间步
        self.time_count = self.time_count + 1
        done = False
        if self.time_count >= self.n_time:
            # 若时间步超过额定时间，本轮结束
            done = True
            # set all the tasks' processing delay and unfinished indicator
            # 设置所有的任务处理延迟和未完成指示器
            for time_index in range(self.n_time):
                for ue_index in range(self.n_ue):
                    if self.process_delay[time_index, ue_index] == 0 and self.arrive_task[time_index, ue_index] != 0:
                        self.process_delay[time_index, ue_index] = (self.time_count - 1) - time_index + 1
                        self.unfinish_task[time_index, ue_index] = 1
 
        # OBSERVATION
        # 观测状态变量
        UEs_OBS_ = np.zeros([self.n_ue, self.n_features])
        UEs_lstm_state_ = np.zeros([self.n_ue, self.n_lstm_state])
        if not done:
            for ue_index in range(self.n_ue):
                # observation is zero if there is no task arrival
                # 如果没有任务到达，observation为0
                if self.arrive_task[self.time_count, ue_index] != 0:
                    # state [A, B^{comp}, B^{tran}, [B^{edge}]]
                    # 非0时为到达任务bit大小，计算等待时间，传输等待时间，边缘队列中剩余需要处理的bit
                    UEs_OBS_[ue_index, :] = np.hstack([
                        self.arrive_task[self.time_count, ue_index],
                        self.t_ue_comp[ue_index] - self.time_count + 1,
                        self.t_ue_tran[ue_index] - self.time_count + 1,
                        self.b_edge_comp[ue_index, :]])

                UEs_lstm_state_[ue_index, :] = np.hstack(self.edge_ue_m_observe)

        return UEs_OBS_, UEs_lstm_state_, done