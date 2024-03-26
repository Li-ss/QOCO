

class Config(object):
    N_UE        = 10   # 用户数量
    N_EDGE      = 2    # 边缘节点数量
    N_COMPONENT = 1    # 组件数量
    N_EPISODE = 10     # 回合数量
    # N_EPISODE   = 1000
    N_TIME_SLOT = 100  # 时间片数量
    MAX_DELAY   = 10   # 最大延迟
    N_TIME      = N_TIME_SLOT + MAX_DELAY  # 时间数量

    UE_COMP_ENERGY   = 2    # 用户计算功率
    UE_TRAN_ENERGY   = 2.3  # 用户和边缘间传输功率
    UE_IDLE_ENERGY   = 0.1  # 用户待机功率
    EDGE_COMP_ENERGY = 5    # 边缘节点计算功率

    DURATION         = 0.1   # 持续时间
    UE_COMP_CAP      = 2.5   # 用户计算容量
    UE_TRAN_CAP      = 14    # 用户和边缘间传输容量
    EDGE_COMP_CAP    = 41.8  # 边缘节点计算容量

    TASK_COMP_DENS   = 0.297  # 计算任务密度
    TASK_ARRIVE_PROB = 0.3    # 任务到达概率
    TASK_MIN_SIZE    = 2      # 最小任务数量
    TASK_MAX_SIZE    = 5      # 最大任务数量

    LEARNING_RATE    = 0.01  # 学习率
    REWARD_DECAY    = 0.9   # 折扣率
    E_GREEDY         = 0.99  # 最大利用概率
    N_NETWORK_UPDATE = 200   # target network 更新间隔
    MEMORY_SIZE      = 500   # 内存最大容量

