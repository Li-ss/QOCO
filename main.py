from MEC_Env import MEC
from DDQN_torch import DuelingDoubleDeepQNetwork
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
# from tensorboard.writer import SummaryWriter


'''
if not os.path.exists("models"):
    os.mkdir("models")
else:
    shutil.rmtree("models")
    os.mkdir("models")
'''

def reward_fun(ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy, delay, max_delay, unfinish_task):
    # 根据各个能量和延迟计算奖励
    edge_energy  = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    penalty     = -max_delay*4
    if unfinish_task == 1:
        reward = penalty
    else:
        reward = 0
    reward = reward - (ue_comp_energy + ue_trans_energy + edge_energy + idle_energy)
    return reward

def monitor_reward(ue_RL_list, episode):
    # 根据记录信息求每轮平均奖励
    episode_sum_reward = sum(sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list)
    avg_episode_sum_reward = episode_sum_reward / len(ue_RL_list)
    print(f"reward: {avg_episode_sum_reward}")
    return avg_episode_sum_reward

def monitor_delay(ue_RL_list, episode):
    # 根据记录信息求每轮平均延迟
    delay_ue_list = [sum(ue_RL.delay_store[episode]) for ue_RL in ue_RL_list]
    avg_delay_in_episode = sum(delay_ue_list) / len(delay_ue_list)
    print(f"delay: {avg_delay_in_episode}")
    return avg_delay_in_episode

def monitor_energy(ue_RL_list, episode):
    # 根据记录信息求每轮平均能量
    energy_ue_list = [sum(ue_RL.energy_store[episode]) for ue_RL in ue_RL_list]
    avg_energy_in_episode = sum(energy_ue_list) / len(energy_ue_list)
    print(f"energy: {avg_energy_in_episode}")
    return avg_energy_in_episode

def cal_reward(ue_RL_list):
    total_sum_reward = 0
    num_episodes = 0
    for ue_num, ue_RL in enumerate(ue_RL_list):
        print("________________________")
        print("ue_num:", ue_num)
        print("________________________")
        for episode, reward in enumerate(ue_RL.reward_store):
            print("episode:", episode)
            reward_sum = sum(reward)
            print(reward_sum)
            total_sum_reward += reward_sum
            num_episodes += 1
    avg_reward = total_sum_reward / num_episodes
    print(total_sum_reward, avg_reward)


def train(ue_RL_list, NUM_EPISODE):
    avg_reward_list = []
    avg_reward_list_2 = []
    avg_delay_list_in_episode = []
    avg_energy_list_in_episode = []
    num_task_drop_list_in_episode = []
    RL_step = 0
    a = 1

    writer = SummaryWriter('log_1000')
    for episode in range(NUM_EPISODE):
        print("episode  :", episode)
        print("epsilon  :", ue_RL_list[0].epsilon)

        # BITRATE ARRIVAL
        # 随机生成到达的bit
        bitarrive = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])

        # OBSERVATION MATRIX SETTING
        # 设置观测矩阵，用于记录历史状态和对应动作
        history = list()


        for time_index in range(env.n_time):
            history.append(list())
            for ue_index in range(env.n_ue):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_ue])

        # INITIALIZE OBSERVATION
        # 将环境初始化，并观测当前状态
        observation_all, lstm_state_all = env.reset(bitarrive)

        # TRAIN DRL
        # 训练各用户
        while True:

            # PERFORM ACTION
            # 各用户根据观测的当前状态执行动作
            action_all = np.zeros([env.n_ue])
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if np.sum(observation) == 0:
                    # if there is no task and ready to compute and transmit, action = 0 (also need to be stored)
                    # 如果各状态参数都为0（无任务且可立即计算和传输），action = 0（本地计算），也需要记录
                    action_all[ue_index] = 0
                else:
                    # 用户对应的神经网络根据当前状态选择动作
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    if observation[0] != 0:
                        # 有任务到达时记录动作
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            # 环境转移，观测下一状态的参数、执行延迟和奖励，本轮是否完成
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # update the lstm state
            # 更新lstm状态
            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index,:])

            # STORE HISTORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            # 记录历史状态和对应动作，如果有任务延迟刚更新，意味着任务处理完或被drop，记录transition
            for ue_index in range(env.n_ue):
                obs = observation_all[ue_index, :]
                lstm = np.squeeze(lstm_state_all[ue_index, :])
                action = action_all[ue_index]
                obs_ = observation_all_[ue_index]
                lstm_ = np.squeeze(lstm_state_all_[ue_index,:])
                history[env.time_count - 1][ue_index].update({
                    'observation': obs,
                    'lstm': lstm,
                    'action': action,
                    'observation_': obs_,
                    'lstm_': lstm_
                })

                update_index = np.where((1 - reward_indicator[:,ue_index]) *env.process_delay[:,ue_index] > 0)[0]
                if len(update_index) != 0:
                    for time_index in update_index:
                        reward = reward_fun(
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy [time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index],
                            env.process_delay[time_index, ue_index],
                            env.max_delay,
                            env.unfinish_task[time_index, ue_index]
                        )
                        ue_RL_list[ue_index].store_transition(
                            history[time_index][ue_index]['observation'],
                            history[time_index][ue_index]['lstm'],
                            history[time_index][ue_index]['action'],
                            reward,
                            history[time_index][ue_index]['observation_'],
                            history[time_index][ue_index]['lstm_']
                        )
                        ue_RL_list[ue_index].do_store_reward(
                            episode,
                            time_index,
                            reward
                        )
                        ue_RL_list[ue_index].do_store_delay(
                            episode,
                            time_index,
                            env.process_delay[time_index, ue_index]
                        )
                        ue_RL_list[ue_index].do_store_energy(
                            episode,
                            time_index,
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy [time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index]
                        )
                        reward_indicator[time_index, ue_index] = 1
                        writer.add_scalar("reward(QoE)  :", reward, episode)
                        writer.add_scalar("reward:", episode, episode)


            # ADD STEP (one step does not mean one store)
            # 学习步加一，一步不代表只有一个记录（可能没有或多个）
            RL_step += 1

            # UPDATE OBSERVATION
            # 更新观测的状态变量
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY
            # 如学习步大于200，每个10步更新一次神经网络
            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn(RL_step, ue)  # 添加一个参数记录RL_step

            # GAME ENDS
            # 若本轮结束，统计任务丢弃数量、完成数量、丢弃率、
            if done:
                for task in env.task_history:
                    cmpl = drp = 0
                    for t in task:
                        d_states = t['d_state']
                        if any(d < 0 for d in d_states):
                            t['state'] = 'D'
                            drp += 1
                        elif all(d > 0 for d in d_states):
                            t['state'] = 'C'
                            cmpl += 1
                full_complete_task = 0
                full_drop_task = 0
                complete_task = 0
                drop_task = 0
                for history in env.task_history:
                    for task in history:
                        if task['state'] == 'C':
                            full_complete_task += 1
                        elif task['state'] == 'D':
                            full_drop_task += 1
                        for component_state in task['d_state']:
                            if component_state == 1:
                                complete_task += 1
                            elif component_state == -1:
                                drop_task += 1
                cnt = len(env.task_history) * len(env.task_history[0]) * env.n_component

                writer.add_scalar("drop_rate   : ", full_drop_task/(cnt/env.n_component), episode)
                writer.add_scalar("full_drop   : ", full_drop_task, episode)
                writer.add_scalar("full_complete: ", full_complete_task, episode)
                writer.add_scalar("complete_task: ", complete_task, episode)
                writer.add_scalar("drop_task:     ", drop_task, episode)
                # print("++++++++++++++++++++++")
                # print("drop_rate   : ", full_drop_task/(cnt/env.n_component))
                # print("full_drop   : ", full_drop_task)
                # print("full_complete: ", full_complete_task)
                # print("complete_task: ", complete_task)
                # print("drop_task:     ", drop_task)
                # print("++++++++++++++++++++++")

                # if episode % 9 == 0 and episode != 0:
                #     os.mkdir("models1" + "/" + str(episode))
                #     for ue in range(env.n_ue):
                #         ue_RL_list[ue].save_model(str(ue)) 
                #         ue_RL_list[ue].saver.save(ue_RL_list[ue].sess, "models1/" + str(episode) +'/'+ str(ue) + "_X_model" +'/model.ckpt', global_step=episode)

                avg_reward_list.append((monitor_reward(ue_RL_list, episode)))  # 求奖励时各增加了1000，此处去掉了负号
                if episode % 10 == 0:
                    # 每隔10轮记录平均奖励、平均延迟、平均能量和丢弃任务数量
                    avg_reward_list_2.append(sum(avg_reward_list[episode-10:episode])/10)
                    avg_delay_list_in_episode.append(monitor_delay(ue_RL_list, episode))
                    avg_energy_list_in_episode.append(monitor_energy(ue_RL_list, episode))
                    total_drop = full_drop_task
                    num_task_drop_list_in_episode.append(total_drop)

                    # Plotting and saving figures
                    # '''
                if episode % 99 == 0 and episode != 0:
                    fig, axs = plt.subplots(5, 1, figsize=(8, 16))
                    axs[0].plot(avg_reward_list, '-')
                    axs[0].set_ylabel('LSTM')
                    axs[1].plot(avg_delay_list_in_episode, '-')
                    axs[1].set_ylabel('r')
                    axs[2].plot(avg_energy_list_in_episode, '-')
                    axs[2].set_ylabel('r')
                    axs[3].plot(num_task_drop_list_in_episode, '-')
                    axs[3].set_ylabel('r')
                    axs[4].plot(ue_RL_list[0].loss_store, '-')
                    axs[4].set_ylabel('loss')
                    plt.savefig('figures.png')
                    # '''

                    # Writing data to files
                    '''
                    data = [avg_reward_list, avg_delay_list_in_episode, avg_energy_list_in_episode, num_task_drop_list_in_episode]
                    filenames = ['reward.txt', 'delay.txt', 'energy.txt', 'drop.txt']
                    for i in range(len(data)):
                        with open(filenames[i], 'w') as f:
                            f.write('\n'.join(str(x) for x in data[i]))
                    '''
                # Process energy
                # 处理能量
                ue_bit_processed = sum(sum(env.ue_bit_processed))
                ue_comp_energy = sum(sum(env.ue_comp_energy))

                # Transmission energy
                # 传输能量
                ue_bit_transmitted = sum(sum(env.ue_bit_transmitted))
                ue_tran_energy = sum(sum(env.ue_tran_energy))

                # edge energy
                # 边缘能量
                edge_bit_processed = sum(sum(env.edge_bit_processed))
                edge_comp_energy = sum(sum(env.edge_comp_energy))
                ue_idle_energy = sum(sum(env.ue_idle_energy))

                # Print results
                # 结果打印
                writer.add_scalars("local", {'ubp': int(ue_bit_processed), 'uce': ue_comp_energy},  episode)
                writer.add_scalars("trans", {'ubt': int(ue_bit_transmitted), 'ute': ue_tran_energy}, episode)
                writer.add_scalars("edge", {'ebp': int(sum(edge_bit_processed)), 'ece': sum(edge_comp_energy), 'uie': sum(ue_idle_energy)}, episode)
                # print(int(ue_bit_processed), ue_comp_energy, "local")
                # print(int(ue_bit_transmitted), ue_tran_energy, "trans")
                # print(int(sum(edge_bit_processed)),sum(edge_comp_energy), sum(ue_idle_energy), "edge")
                # print("_________________________________________________")

                break # Training Finished
    writer.close()



if __name__ == "__main__":

    # GENERATE ENVIRONMENT
    # 生成环境，给定用户数量、边缘节点数量、时间数量、组件数量、最大延迟
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    # 生成多个强化学习类神经网络代表各用户
    ue_RL_list = list()
    for ue in range(Config.N_UE):
        ue_RL_list.append(DuelingDoubleDeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                                    learning_rate       = Config.LEARNING_RATE,
                                                    reward_decay        = Config.REWARD_DECAY,
                                                    e_greedy            = Config.E_GREEDY,
                                                    replace_target_iter = Config.N_NETWORK_UPDATE,  # each 200 steps, update target net
                                                    memory_size         = Config.MEMORY_SIZE,  # maximum of memory
                                                    ))

    # LOAD MODEL
    '''
    for ue in range(Config.N_UE):
        ue_RL_list[ue].Initialize(ue_RL_list[ue].sess, ue)
    '''

    # TRAIN THE SYSTEM
    # 训练
    train(ue_RL_list, Config.N_EPISODE)


