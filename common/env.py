import numpy as np
import random
import math
from .buffer import Trajectory
from .utils import calculate_prediction_error
import statistics
import matplotlib.pyplot as plt



class I2Vchannel:
    def __init__(self, RSU_position, RSU_trans_power, noise_power):
        self.h_bs = 25                               # RSU高度
        self.h_ms = 1.5                              # 车辆高度
        self.Decorrelation_distance = 50
        self.RSU_position = [RSU_position, 0]             # 路口处，假设单条公路场景，将监控范围边缘作为起点0，则RSU坐标为监测范围，此处取平行公路间距的一半
        self.RSU_trans_power = RSU_trans_power          # RSU的发射功率（dBm）
        self.shadow_std = 8                             # 8
        self.noise_power = noise_power                  # 噪声功率（dBm）

    def get_path_loss(self, position, position_orth):
        d1 = abs(position - self.RSU_position[0])
        d2 = abs(position_orth - self.RSU_position[1])
        distance = math.hypot(d1, d2)
        # 自由空间路径损耗，Pathloss=128.1+37.6*lg(d) d单位为km,Pathloss单位为dB
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, previous_shadowing):
        return (np.exp(-1 * (delta_distance / self.Decorrelation_distance)) * previous_shadowing) + \
               (np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, self.shadow_std))

    def get_channel_gain(self, position, position_orth, delta_distance, previous_shadowing):
        path_loss = self.get_path_loss(position, position_orth)      # 计算路径损耗（单位：dB）
        shadow = self.get_shadowing(delta_distance, previous_shadowing)     # 计算阴影衰落（单位：dB）
        P_received_dBm = self.RSU_trans_power - (path_loss + shadow)        # 单位：dBm
        # 计算信道增益
        channel_gain_dB = P_received_dBm - self.noise_power     # 单位：dB
        channel_gain = 10 ** (channel_gain_dB / 10)
        return channel_gain, shadow

    def get_channel_capacity(self, position, position_orth, delta_distance, previous_shadowing, bandwidth):
        channel_gain, shadow = self.get_channel_gain(position, position_orth, delta_distance, previous_shadowing)
        RSU_trans_power = 10 ** ((self.RSU_trans_power - 30) / 10)  # 将发射功率从 dBm 转换为瓦
        noise_power = 10 ** ((self.noise_power - 30) / 10)          # 将噪声功率从 dBm 转换为瓦
        snr = RSU_trans_power * channel_gain / noise_power
        capacity = bandwidth * np.log2(1 + snr)         # 单位：bps
        # print('传输速率bps：', capacity)
        return capacity, shadow



class Vehicle:
    def __init__(self, info):
        self.arrival_timeslot = info['arrival_timeslot']
        self.f = info['f']                          # 计算能力(GHz)
        self.velocity = info['velocity']
        self.anytime_alpha = None
        self.code_rate = None          # 码率
        self.departure_timeslot = info['departure_timeslot']
        self.is_warm_up = info['is_warm_up']
        self.I2V_channel = info['I2V_channel']
        self.position_orth = info['position_orth']

        self.position = info['position']
        self.time_remained = info['time_remained']  # 剩余时间(s)
        self.bits_in_queue = 0
        self.received_data = 0                  # 已经接收的数据量(bit)
        self.is_computed = False
        self.is_left = False
        self.trans_rate = None
        self.shadowing = np.random.normal(0, self.I2V_channel.shadow_std)
        self.traj = Trajectory()

    def compute_step_reward(self, args):
        ''' 计算当车辆选择计算时的reward '''

        FF = self.time_remained * self.f / args.G0          # f和G0的单位都是 G级别
        F = max([x for x in args.diff_step_set if x <= FF], default=min(args.diff_step_set))  # 选择扩散步数
        # print('选择的扩散步数: ', F)
        comp_delay = args.G0 * F / self.f                  # 实际计算时延

        received_frame_num = int(self.received_data / (args.frame_size * 1024 * 8 * self.code_rate))
        prediction_error = calculate_prediction_error(received_frame_num, self.anytime_alpha, self.time_remained, F, args.time_slot)

        received_dataa = self.received_data / 8 / 1024

        reward = - (prediction_error + args.lamda * received_dataa) - args.r_T * max(comp_delay - self.time_remained, 0)   # 0.5x -- 1.xx
        # print('====== reward ======')
        # print('预测误差: ', prediction_error)
        # print('接收数据量: ', received_dataa)
        # print('超出时间: ', comp_delay - self.time_remained, ', comp_delay: ', comp_delay, ', time_remained: ', self.time_remained)
        # print('reward: ', reward)

        return reward

    def update_trans_rate(self, timeslot, bandwidth):
        self.trans_rate, self.shadowing = self.I2V_channel.get_channel_capacity(
                                                 self.position, self.position_orth, self.velocity * timeslot, self.shadowing, bandwidth)





class Env:
    def __init__(self, args):
        self.monitor_range = args.monitor_range                # RSU监控范围(m)
        self.RSU_position = args.monitor_range      # RSU位置设为监控范围边缘，即车辆进入时位置为0
        self.time_slot = args.time_slot                        # 时隙长度(s)
        self.warm_up_vehicles = args.warm_up_vehicles
        self.I2V_channel = I2Vchannel(self.RSU_position, args.RSU_trans_power, args.noise_power)
        self.orth_pos_list = [3,6,9]            # 车辆与RSU在垂直方向的距离，类似车道宽度

        # 车辆参数
        self.min_velocity = args.v_min  # 最小车速(m/s)
        self.max_velocity = args.v_max  # 最大车速(m/s)
        self.vehicle_arrival_rate = args.vehicle_arrival_rate     # 车辆泊松到达过程中，平均每时隙到达率(veh/slot)
        self.position_variance = 10  # 初始位置的随机波动范围(m)
        self.f_min = args.f_min     # 最小计算能力(GHz)
        self.f_max = args.f_max
        self.diffusion_steps_range = args.diff_step_set  # 可选扩散步数

        # RSU传输参数
        self.data_rates = args.rates_set  # 可选码率级别
        self.alpha_set = args.any_alpha_set   # 码率对应的anytime因子alpha的集合
        self.base_bandwidth = args.bandwidth             # 总带宽 (Hz)
        self.tx_power = args.RSU_trans_power           # 发射功率(W)
        self.frame_size = args.frame_size          # 原始特征图大小(KB)
        self.beta = args.beta              # anytime误码率参数
        self.a = args.a                  # 计算车辆紧迫性权重的参数
        self.gamma = args.gamma            # 计算码率倾向因子的放缩参数

        # 环境状态
        self.current_timeslot = 0
        self.vehicles = {}         # 车辆字典：{id: {状态信息}} / {id: obj}
        self.active_vehicles = []  # 当前活跃车辆ID列表 / 当前时隙在监控范围内的车辆ID列表
        self.history_vehicles = []  # 历史车辆ID列表
        self.next_vehicle_id = 0  # 下一个车辆ID (可用于记录总共有过多少车辆，包括初始化的和预热的车辆)
        self.vehicle_counter = 0  # 到达车辆计数器（用于预热）

        # RSU带宽历史记录
        self.bandwidth_history_len = 5  # 保存的带宽历史长度
        self.bandwidth_history = []    # 每个时隙的平均单车可用带宽

        # 初始化RSU带宽历史
        for _ in range(self.bandwidth_history_len):
            self.bandwidth_history.append(self.base_bandwidth / 10)     # 除以平均车辆数

        # 初始化场景
        self._initialize_traffic_scene()


    def reset(self):
        """重置整个环境，开始新的模拟"""
        self.current_timeslot = 0
        self.vehicles = {}
        self.active_vehicles = []
        self.history_vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_counter = 0

        # 保留带宽历史记录，因为这是RSU的长期状态
        # 但为避免极端值影响，重置为基础带宽的平均值
        avg_bandwidth = sum(self.bandwidth_history) / len(self.bandwidth_history)
        self.bandwidth_history = [avg_bandwidth] * self.bandwidth_history_len

        # 初始化场景
        self._initialize_traffic_scene()


    def _initialize_traffic_scene(self):
        """
        初始化交通场景，创建更加真实的车流
        在监控区域内的不同位置生成一些车辆，模拟已经在路上行驶的车辆
        现在场景有效区域为[0, monitor_range]，车辆从0开始向monitor_range移动
        """
        # 生成分布在监控区域内的车辆
        num_initial_vehicles = random.randint(8, 12)

        # 在监控区域内均匀分布车辆
        for i in range(num_initial_vehicles):
            # 计算初始位置 - 在监控区域内均匀分布
            position_factor = i / num_initial_vehicles  # 0到1之间的值
            base_position = self.monitor_range * position_factor

            # 添加一些随机波动，避免完全均匀
            position_noise = np.random.uniform(-self.position_variance, self.position_variance)
            position = base_position + position_noise

            # 超出监控区域的不管它
            if position >= 0 and position < self.monitor_range:
                self._add_vehicle_at_position(position)     # 添加车辆


    def _add_vehicle_at_position(self, position):
        """在指定位置添加新车辆"""
        vehicle_id = self.next_vehicle_id       # 从0开始编号
        self.next_vehicle_id += 1

        # 随机生成车辆速度
        velocity = np.random.uniform(self.min_velocity, self.max_velocity)

        # 距离到达路口的时隙跨度（xx个时隙）
        total_monitor_time = (self.RSU_position - position) / velocity / self.time_slot

        # 选择码率(在集合中的序号)
        data_rate_index = random.randrange(0, len(self.data_rates))

        # 初始化车辆状态
        self.vehicles[vehicle_id] = Vehicle({
            'arrival_timeslot': self.current_timeslot,
            'position': position,
            'position_orth': random.choice(self.orth_pos_list),
            'velocity': velocity,
            'time_remained': total_monitor_time * self.time_slot,         # time_to_intersection
            'departure_timeslot': self.current_timeslot + total_monitor_time,
            'f': random.randint(self.f_min, self.f_max),
            'I2V_channel': self.I2V_channel,
            'is_warm_up': True                        # 初始化时添加的车辆属于预热车辆，不统计其动作轨迹
        })
        self.vehicles[vehicle_id].code_rate = self.data_rates[data_rate_index]
        self.vehicles[vehicle_id].anytime_alpha = self.alpha_set[data_rate_index]
        # 已接收数据量：随机系数 * 进入以来的时隙数 * 一帧实际传输大小
        self.vehicles[vehicle_id].received_data = random.uniform(0.6, 1) * int(position / velocity / self.time_slot) * self.frame_size * 1024 * 8 * self.vehicles[vehicle_id].code_rate
        self.vehicles[vehicle_id].bits_in_queue = random.randrange(0, int(self.frame_size * 1024 * 8 * self.vehicles[vehicle_id].code_rate * 1.2))
        self.vehicles[vehicle_id].update_trans_rate(self.time_slot, self.bandwidth_history[-1])
        self.vehicles[vehicle_id].is_computed = random.uniform(0, 1) <= (position/self.monitor_range)

        # 添加车辆id到当前场景车辆列表
        self.active_vehicles.append(vehicle_id)


    def arrive_new_vehicles(self):
        """
        单个时隙按泊松过程随机到达新的车辆
        """
        self.new_vehicles = []       # 记录该时隙新到达的车辆的id
        k = np.random.poisson(self.vehicle_arrival_rate)
        for _ in range(k):
            self._add_new_vehicle()        # 添加具体车辆

        return self.vehicles, self.active_vehicles, self.new_vehicles

    def _add_new_vehicle(self):
        """
        向场景中添加新进入的具体车辆
        """
        vehicle_id = self.next_vehicle_id
        self.new_vehicles.append(vehicle_id)
        self.next_vehicle_id += 1
        self.vehicle_counter += 1  # 增加车辆计数器

        # 随机生成车辆速度
        velocity = np.random.uniform(self.min_velocity, self.max_velocity)

        # 在起点附近添加随机波动
        position = 0 + np.random.uniform(0, self.position_variance)
        position = abs(position)  # 确保不低于0

        # 距离到达路口的时隙跨度（xx个时隙）
        total_monitor_time = (self.RSU_position - position) / velocity / self.time_slot

        # 判断是否为预热车辆（前WARM_UP_VEHICLES个新到达的车辆为预热车辆）
        is_warm_up = self.vehicle_counter <= self.warm_up_vehicles

        # 初始化车辆状态
        self.vehicles[vehicle_id] = Vehicle({
            'arrival_timeslot': self.current_timeslot,
            'f': random.randint(self.f_min, self.f_max),
            'position': position,
            'position_orth': random.choice(self.orth_pos_list),  # 车辆与RSU在垂直方向的距离，类似车道宽度
            'velocity': velocity,
            'time_remained': total_monitor_time * self.time_slot,  # time_to_intersection
            'departure_timeslot': self.current_timeslot + total_monitor_time,
            'I2V_channel': self.I2V_channel,
            'is_warm_up': is_warm_up                    #
        })

        # 添加车辆id到当前场景车辆列表
        self.active_vehicles.append(vehicle_id)


    def select_data_rate(self, vehicle_id, selected_method=2):
        """ 为某车辆选择传输码率 """
        vehicle = self.vehicles[vehicle_id]
        pre_timeslots = int(vehicle.time_remained / self.time_slot)       # 剩余时隙数
        pre_rates = [vehicle.trans_rate]                            # 记录未来每个时隙的预测传输速率(bps)，初始值为当前时隙速率，可观测

        # 使用调和平均值，依次预测未来各时隙的带宽，并计算各时隙的速率
        shadoww = vehicle.shadowing
        bands = self.bandwidth_history.copy()
        for i in range(1,pre_timeslots):
            inv_bandwidths = [1 / bw for bw in bands]
            predicted_bandwidth = len(inv_bandwidths) / sum(inv_bandwidths)         # 调和平均值 = n / (1/x1 + 1/x2 + ... + 1/xn)
            bands.append(predicted_bandwidth)
            bands.pop(0)
            predicted_rate, shadoww = self.I2V_channel.get_channel_capacity(vehicle.position+vehicle.velocity*self.time_slot*i,
                                                                   vehicle.position_orth, vehicle.velocity*self.time_slot,
                                                                   shadoww, predicted_bandwidth)
            pre_rates.append(predicted_rate)

        # 估计平均传输速率
        avg_pre_rate = sum(pre_rates) / len(pre_rates)

        # 计算紧迫性权重
        urgency_weight = np.exp(-self.a * vehicle.time_remained)
        # print('紧迫性权重u: ', urgency_weight)


        #### 方法 1 ####
        if selected_method == 1:
            l_tt = avg_pre_rate * self.time_slot / (self.frame_size * 1024 * 8)
            index = 0
            min_diff = abs(self.data_rates[0] - urgency_weight * l_tt)
            for i in range(len(self.data_rates)):
                diff = abs(self.data_rates[i] - urgency_weight * l_tt)
                if diff < min_diff:
                    min_diff = diff
                    index = i

        #### 方法 2 ####
        elif selected_method == 2:
            l_mid = statistics.median(self.data_rates)
            e = avg_pre_rate * self.time_slot - self.frame_size * 1024 * 8 * l_mid
            e = e * 1e-7
            m = urgency_weight / (1 + np.exp(-self.gamma * e))          # 根据平均速率和紧迫性权重计算码率倾向因子
            # print('码率倾向因子m: ', m)
            index = round(m * (len(self.data_rates) - 1))               # 将码率倾向因子映射到具体码率的序号

        return index


    def update_bandwidth_history(self):
        """
        计算根据当前时隙需要通信的车辆数量，并据此更新车均带宽
        """
        # 计算当前需要接收数据的车辆数
        to_trans_vehicles = [id for id in self.active_vehicles
                              if not self.vehicles[id].is_computed]

        # 均分带宽给所有活跃且正在接收数据的车辆
        if len(to_trans_vehicles) != 0:
            bandwidth_per_vehicle = self.base_bandwidth / len(to_trans_vehicles)
        else:
            bandwidth_per_vehicle = self.base_bandwidth

        # 将当前时隙的车均带宽放入RSU的带宽历史记录中
        self.bandwidth_history.append(bandwidth_per_vehicle)
        if len(self.bandwidth_history) > self.bandwidth_history_len:
            self.bandwidth_history.pop(0)  # 移除最旧的记录

        return bandwidth_per_vehicle, to_trans_vehicles


    def update_vehicles(self):
        """ 更新上一时隙场景中的车辆在此时隙开始时的状态 """
        for vehicle_id in self.active_vehicles:
            vehicle = self.vehicles[vehicle_id]
            vehicle.position += vehicle.velocity * self.time_slot       # 更新位置
            vehicle.time_remained -= self.time_slot                     # 更新到达路口的剩余时间
            if vehicle.time_remained <= 0:                              # 如果车辆已到达路口，标记为离开
                vehicle.is_left = True
                self.active_vehicles.remove(vehicle_id)
            if not vehicle.is_computed:                     # 如果车辆上一时隙未计算，更新上一时隙的传输数据量。此时vehicle信息中还是上一时隙的传输速率。如果上一时隙计算了，则上一时隙未传输，无新增传输量
                transmitted_bits_last_slot = min(vehicle.trans_rate * self.time_slot, vehicle.bits_in_queue)    # 上一时隙传输的比特数
                vehicle.bits_in_queue -= transmitted_bits_last_slot         # 更新传输队列中的剩余的比特数
                vehicle.received_data += transmitted_bits_last_slot         # 更新已接收数据量


    def _calculate_avg_remaining_time(self, exclude_id=None):
        """ 计算除了指定ID外其他还未计算的车辆的平均剩余时间 """
        remaining_times = []
        for vid in self.active_vehicles:
            if vid != exclude_id and not self.vehicles[vid].is_computed:          # 只统计未计算的车辆
                remaining_times.append(self.vehicles[vid].time_remained)

        if len(remaining_times) > 0:
            return len(remaining_times), np.mean(remaining_times)

        return len(remaining_times), 0


    def get_vehicle_state(self, vehicle_id):
        """ 获取指定车辆当前时隙的状态 """
        vehicle = self.vehicles[vehicle_id]

        # 计算其他车辆的平均剩余时间（排除预热车辆）
        num_non_cmp_vehicles, avg_other_time = self._calculate_avg_remaining_time(vehicle_id)

        return np.array([
            vehicle.time_remained,      # 剩余时间
            int( vehicle.received_data / (self.frame_size * 1024 * 8 * vehicle.code_rate) ),      # 已接收帧数
            vehicle.anytime_alpha,                                                              # 误码率下降速率
            num_non_cmp_vehicles,                              # 场景中其它未计算车辆的数量
            avg_other_time                                     # 其它未计算车辆的平均剩余时间
        ], dtype=np.float32)


