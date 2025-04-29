import torch
import os
import numpy as np


class Trainer:
    def __init__(self, policy, env, buffer, agent, args):
        super().__init__()

        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.batch_size = args.batch_size
        self.num_episode = args.num_episode
        self.sim_timesteps = args.sim_timesteps      # 每次采集样本时模拟的时隙长度
        self.epsilon = args.epsilon                  # 选择动作时epsilon-greedy的参数
        self.epsilon_decay = args.epsilon_decay      # 每轮训练后epsilon衰减率
        self.epsilon_decay_interval = args.epsilon_decay_interval
        self.test_interval = args.test_interval
        self.num_test_episodes = 5                   #  每次测试时运行的轮数
        self.args = args
        self.writer = None                           # For logging.

    def train(self, check_dir):

        for episode in range(self.num_episode):
            print(f"Episode {episode + 1}/{self.num_episode}")
            self.env.reset()

            # -------- 每次模拟一段时间的道路环境，获得各车辆的轨迹样本，用buffer来存储（运行固定时间长度，从中提取完整样本，样本数量不固定）
            # -------- 多次模拟，直至样本数量足够一个batch --------
            while not self.buffer.ready(batch_size=self.batch_size):
                self.sim_trajs()

            # -------- 从本次采集的样本中取出batch_size个 --------
            trajs = self.buffer.sample(batch_size=self.batch_size)

            # Track average trajectory reward before training
            avg_reward = sum(traj.traj_reward for traj in trajs) / len(trajs)
            print(f"    Average trajectory reward: {avg_reward:.4f}")

            # -------- 更新网络 --------
            loss = self.policy.learn(trajs, self.buffer, self.args.device)
            print(f"    Policy loss: {loss:.4f}")

            if self.writer:
                self.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
                self.writer.add_scalar('Training/Policy_Loss', loss, episode)
                self.writer.add_scalar('Training/Epsilon', self.epsilon, episode)

            # -------- 更新epsilon-greedy参数 --------
            if (episode + 1) % self.epsilon_decay_interval == 0:
                if self.args.epsilon_decay_type == 'multi':
                    self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
                elif self.args.epsilon_decay_type == 'linear':
                    self.epsilon -= (self.args.epsilon - 0.01) / max(1, self.args.num_episode // self.args.epsilon_decay_interval)
                else:
                    raise ValueError("Unknown epsilon decay type")

            # -------- 定期测试模型 --------
            if (episode + 1) % self.test_interval == 0:
                mean_test_reward = self.test(self.num_test_episodes)
                if self.writer:
                    self.writer.add_scalar('Testing/Average_Reward', mean_test_reward, episode)

            # -------- 定期保存模型 --------
            # if (episode + 1) % self.args.save_model_interval == 0:
            #     checkpoints_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), check_dir)
            #     self.policy.save(os.path.join(checkpoints_dir, f"episode_{episode + 1}_"))

            # -------- 清空buffer，预备下次训练重新采集 --------
            self.buffer.clear()


    def sim_trajs(self):
        '''
        逐时隙运行，每个时隙内，按以下顺序执行：
          1. 每进入一个新的时隙，更新场景中车辆状态（即active_vehicles中的车辆状态，未更新时其中保存的是上一时隙场内车辆）
             更新车辆位置、剩余时间、是否离开、本地持有数据量、传输队列长度（减去上一时隙传输的部分）
             将已超出监测区域的车辆从active_vehicles中移除
             获得已出现车辆中当前时隙仍在场景内的车辆状态，即针对当前时隙的active_vehicles（还没加入当前时隙新到达的）
          2. 添加此时隙新到达的车辆（设置位置、速度），并将id添加到active_vehicles中
             至此，active_vehicles中保存的是当前时隙场景内的所有车辆
          3. 遍历active_vehicles，使每一个非新到达、仍未计算的车辆执行动作决策（新到达的车辆还未接收数据，必定不计算）
             并将其记录进该车的轨迹中
          4. 根据第3步，排除本时隙决定计算的车辆后，获得本时隙参与传输的车辆和车均带宽
             给这些参与传输的车辆更新本时隙的传输速率（速率的计算需要可用带宽）
          5. 基于本时隙及之前的历史车均带宽，给每个新到达车辆选择码率，并设置与码率相关的anytime_alpha系数
          6. 更新所有本时隙参与传输的车辆的传输队列长度（加上此时隙要传的一帧）

        逐步运行完固定时隙长度后，取所有车辆中完整的轨迹样本，放入buffer中
        '''
        states_value_range = [[np.inf, -np.inf] for _ in range(9)]

        for t in range(self.sim_timesteps):
            self.env.current_timeslot = t + 1

            # 1. 更新 active_vehicles
            self.env.update_vehicles()

            # 2. 添加此时隙新到达的车辆及其状态
            vehicles, active_vehicles, new_vehicles = self.env.arrive_new_vehicles()

            # 3. 针对当前场景中每个仍未计算的车辆，执行动作决策
            for vehicle_id in active_vehicles:
                # print('\n【vehicle_id】: ', vehicle_id)
                vehicle = vehicles[vehicle_id]

                if vehicle.is_computed:   # 如果车辆已计算，跳过
                    continue
                if vehicle_id in new_vehicles:  # 如果是新到达的车辆，跳过
                    continue

                # 获取车辆状态
                state = self.env.get_vehicle_state(vehicle_id)   # [计算能力，剩余时间，已接收帧数，下一帧的接收比例，码率，误码率下降速率alpha，与RSU空间距离，场景中其它未计算车辆的数量，这些未计算车辆的平均剩余时间]
                # print('====== state ======')
                # print('计算能力： ', state[0])
                # print('剩余时间： ', state[1])
                # print('已接收帧数： ', state[2])
                # print('下一帧的接收比例： ', state[3])
                # print('码率： ', state[4])
                # print('误码率alpha： ', state[5])
                # print('与RSU空间距离： ', state[6])
                # print('其它未计算车辆的数量： ', state[7])
                # print('其它未计算车辆的平均剩余时间： ', state[8])

                for ii in range(len(state)):                # 记录各状态变量的取值范围
                    states_value_range[ii][0] = min(states_value_range[ii][0], state[ii])
                    states_value_range[ii][1] = max(states_value_range[ii][1], state[ii])

                # 状态取值归一化（此处未完全归一化，有些变量只是根据观测的取值范围，大概使各状态变量取值都趋近于0-1
                state[0] = (state[0] - self.args.f_min) / (self.args.f_max - self.args.f_min)             # 计算能力
                state[1] = state[1] / 7                                                                    # 剩余时间
                state[2] = state[2] / 12                                                                      # 已接收帧数
                state[4] = (state[4] - min(self.args.rates_set)) / (max(self.args.rates_set) - min(self.args.rates_set))    # 码率
                state[5] = (state[5] - min(self.args.any_alpha_set)) / (max(self.args.any_alpha_set) - min(self.args.any_alpha_set))    # 误码率alpha
                state[6] = ((state[6] - vehicle.I2V_channel.get_distance(self.args.monitor_range, min(self.args.orth_pos_list))) /       # 与RSU空间距离
                            (vehicle.I2V_channel.get_distance(0, max(self.args.orth_pos_list)) -
                             vehicle.I2V_channel.get_distance(self.args.monitor_range, min(self.args.orth_pos_list))))
                state[7] = state[7] / 16                                                             # 其它未计算车辆的数量
                state[8] = state[8] / 7                                                             # 其它未计算车辆的平均剩余时间
                # print('state_scaled: ', state)


                # 计算动作概率并选择动作
                action, log_probs = self.agent.action(state, self.epsilon, self.args.device)
                # print('action: ', action)

                # 保存本时隙的状态、动作、奖励
                if action == 0:           # 不计算
                    vehicle.traj.add(state, action, log_probs, 0)   # reward=0
                    # vehicle.compute_step_reward(self.args)
                else:                     # 计算
                    reward = vehicle.compute_step_reward(self.args)
                    vehicle.traj.add(state, action, log_probs, reward)
                    vehicle.is_computed = True

            # 4. 给本时隙参与传输的车辆更新它们在本时隙的传输速率
            bandwidth_per_vehicle, to_trans_vehicles = self.env.update_bandwidth_history()  # 更新本时隙车均带宽、仍需传输的车辆id集合
            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].update_trans_rate(self.args.time_slot, bandwidth_per_vehicle)

            # 5. 给此时隙新到达的车辆设置码率，并设置相应的anytime_alpha系数、传输队列长度(添加一帧的数据)
            for vehicle_id in new_vehicles:
                vehicle = vehicles[vehicle_id]
                code_rate_index = self.env.select_data_rate(vehicle_id)             # 选择码率
                vehicle.code_rate = self.args.rates_set[code_rate_index]
                vehicle.anytime_alpha = self.args.any_alpha_set[code_rate_index]

            # 6. 给所有本时隙参与传输的车辆更新传输队列长度（增加一帧本时隙的数据）
            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].bits_in_queue += self.args.frame_size * 1024 * 8 * vehicles[vehicle_id].code_rate


        # print('====== state 范围 ======')
        # print('计算能力： ', states_value_range[0])
        # print('剩余时间： ', states_value_range[1])
        # print('已接收帧数： ', states_value_range[2])
        # print('下一帧的接收比例： ', states_value_range[3])
        # print('码率： ', states_value_range[4])
        # print('误码率alpha： ', states_value_range[5])
        # print('与RSU空间距离： ', states_value_range[6])
        # print('其它未计算车辆的数量： ', states_value_range[7])
        # print('其它未计算车辆的平均剩余时间： ', states_value_range[8])


        # 取该段时间内所有车辆中完整的轨迹样本，放入buffer中
        for vehicle_id in range(self.env.next_vehicle_id):
            vehicle = self.env.vehicles[vehicle_id]
            # 排除预热车辆
            if not vehicle.is_warm_up:
                checks = [sub[1] for sub in vehicle.traj.states]
                if all(a > b for a, b in zip(checks, checks[1:])):
                    # 已计算的车辆，记录其整条轨迹的reward为计算时隙的reward
                    if vehicle.is_computed:
                        vehicle.traj.compute_trajectory_reward(no_cmp_penalty=None)
                        self.buffer.store_a_traj(vehicle.traj)
                    # 未计算的车辆，包括已离开的和未离开的，未离开的轨迹不完整不记录，已离开的未能及时作出计算决策，在reward中给予惩罚项
                    else:
                        if vehicle.is_left:
                            vehicle.traj.compute_trajectory_reward(no_cmp_penalty=self.args.r_S)
                            self.buffer.store_a_traj(vehicle.traj)


    def test(self, num_test_episodes=5):
        """
        Evaluate the policy without exploration
        Returns average reward across test episodes
        """
        print("Running test evaluation...")
        test_rewards = []
        original_epsilon = self.epsilon  # Save current epsilon
        self.epsilon = 0.0  # Disable exploration during testing

        self.policy.actor.eval()

        for episode in range(num_test_episodes):
            self.env.reset()
            episode_trajs = []

            # Simulate for the same number of timesteps as in training
            self.sim_trajs_test(episode_trajs)

            # Calculate average reward for this test episode
            if episode_trajs:
                avg_reward = sum(traj.traj_reward for traj in episode_trajs) / len(episode_trajs)
                test_rewards.append(avg_reward)
                # print(f"  Test episode {episode + 1}: Average reward = {avg_reward:.4f}")

        self.epsilon = original_epsilon  # Restore original epsilon
        self.policy.actor.train()

        if test_rewards:
            mean_test_reward = sum(test_rewards) / len(test_rewards)
            print(f"Test evaluation complete. Mean test reward: {mean_test_reward:.4f}")
            return mean_test_reward
        else:
            print("No complete trajectories collected during testing.")
            return 0.0


    def sim_trajs_test(self, test_trajs):
        """
        Similar to sim_trajs but specifically for testing (no buffer storage)
        """
        for t in range(self.sim_timesteps):
            self.env.current_timeslot = t + 1

            # 1. Update active_vehicles
            self.env.update_vehicles()

            # 2. Add new vehicles
            vehicles, active_vehicles, new_vehicles = self.env.arrive_new_vehicles()

            # 3. Make decisions for each vehicle
            for vehicle_id in active_vehicles:
                vehicle = vehicles[vehicle_id]

                if vehicle.is_computed or vehicle_id in new_vehicles:
                    continue

                # Get vehicle state
                state = self.env.get_vehicle_state(vehicle_id)

                # 状态取值归一化（此处未完全归一化，有些变量只是根据观测的取值范围，大概使各状态变量取值都趋近于0-1
                state[0] = (state[0] - self.args.f_min) / (self.args.f_max - self.args.f_min)  # 计算能力
                state[1] = state[1] / 7  # 剩余时间
                state[2] = state[2] / 12  # 已接收帧数
                state[4] = (state[4] - min(self.args.rates_set)) / (
                            max(self.args.rates_set) - min(self.args.rates_set))  # 码率
                state[5] = (state[5] - min(self.args.any_alpha_set)) / (
                            max(self.args.any_alpha_set) - min(self.args.any_alpha_set))  # 误码率alpha
                state[6] = ((state[6] - vehicle.I2V_channel.get_distance(self.args.monitor_range,
                                                                         min(self.args.orth_pos_list))) /  # 与RSU空间距离
                            (vehicle.I2V_channel.get_distance(0, max(self.args.orth_pos_list)) -
                             vehicle.I2V_channel.get_distance(self.args.monitor_range, min(self.args.orth_pos_list))))
                state[7] = state[7] / 16  # 其它未计算车辆的数量
                state[8] = state[8] / 7  # 其它未计算车辆的平均剩余时间

                # Get action (with epsilon=0 for pure exploitation)
                action, log_probs = self.agent.action(state, self.epsilon, self.args.device)
                # print('action: ', action)

                # Record state, action, reward
                if action == 0:  # Don't compute
                    vehicle.traj.add(state, action, log_probs, 0)
                else:  # Compute
                    reward = vehicle.compute_step_reward(self.args)
                    vehicle.traj.add(state, action, log_probs, reward)
                    vehicle.is_computed = True

            # 4-6. Update bandwidth, transmission rates, etc.
            bandwidth_per_vehicle, to_trans_vehicles = self.env.update_bandwidth_history()
            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].update_trans_rate(self.args.time_slot, bandwidth_per_vehicle)

            for vehicle_id in new_vehicles:
                vehicle = vehicles[vehicle_id]
                code_rate_index = self.env.select_data_rate(vehicle_id)
                vehicle.code_rate = self.args.rates_set[code_rate_index]
                vehicle.anytime_alpha = self.args.any_alpha_set[code_rate_index]

            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].bits_in_queue += self.args.frame_size * 1024 * 8 * vehicles[vehicle_id].code_rate

        # Collect complete trajectories for testing
        for vehicle_id in range(self.env.next_vehicle_id):
            vehicle = self.env.vehicles[vehicle_id]
            if not vehicle.is_warm_up:
                checks = [sub[1] for sub in vehicle.traj.states]
                if all(a > b for a, b in zip(checks, checks[1:])):
                    if vehicle.is_computed:
                        vehicle.traj.compute_trajectory_reward(no_cmp_penalty=None)
                        test_trajs.append(vehicle.traj)      # 存入专门存放测试样本的列表中
                    elif vehicle.is_left:
                        vehicle.traj.compute_trajectory_reward(no_cmp_penalty=self.args.r_S)
                        test_trajs.append(vehicle.traj)



class Trainer_Rnn:
    def __init__(self, policy, env, buffer, agent, args):
        super().__init__()

        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.batch_size = args.batch_size
        self.num_episode = args.num_episode
        self.sim_timesteps = args.sim_timesteps      # 每次采集样本时模拟的时隙长度
        self.epsilon = args.epsilon                  # 选择动作时epsilon-greedy的参数
        self.epsilon_decay = args.epsilon_decay      # 每轮训练后epsilon衰减率
        self.epsilon_decay_interval = args.epsilon_decay_interval
        self.test_interval = args.test_interval
        self.num_test_episodes = 5                   #  每次测试时运行的轮数
        self.args = args
        self.writer = None                           # For logging.

    def train(self, check_dir):

        for episode in range(self.num_episode):
            print(f"Episode {episode + 1}/{self.num_episode}")
            self.env.reset()

            # -------- 每次模拟一段时间的道路环境，获得各车辆的轨迹样本，用buffer来存储（运行固定时间长度，从中提取完整样本，样本数量不固定）
            # -------- 多次模拟，直至样本数量足够一个batch --------
            while not self.buffer.ready(batch_size=self.batch_size):
                self.sim_trajs()

            # -------- 从本次采集的样本中取出batch_size个 --------
            trajs = self.buffer.sample(batch_size=self.batch_size)

            # Track average trajectory reward before training
            avg_reward = sum(traj.traj_reward for traj in trajs) / len(trajs)
            print(f"    Average trajectory reward: {avg_reward:.4f}")

            # -------- 更新网络 --------
            loss = self.policy.learn(trajs, self.buffer, self.args.device)
            print(f"    Policy loss: {loss:.4f}")

            if self.writer:
                self.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
                self.writer.add_scalar('Training/Policy_Loss', loss, episode)
                self.writer.add_scalar('Training/Epsilon', self.epsilon, episode)

            # -------- 更新epsilon-greedy参数 --------
            if (episode + 1) % self.epsilon_decay_interval == 0:
                if self.args.epsilon_decay_type == 'multi':
                    self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
                elif self.args.epsilon_decay_type == 'linear':
                    self.epsilon -= (self.args.epsilon - 0.01) / max(1, self.args.num_episode // self.args.epsilon_decay_interval)
                else:
                    raise ValueError("Unknown epsilon decay type")


            # -------- 定期测试模型 --------
            if (episode + 1) % self.test_interval == 0:
                mean_test_reward = self.test(self.num_test_episodes)
                if self.writer:
                    self.writer.add_scalar('Testing/Average_Reward', mean_test_reward, episode)

            # -------- 定期保存模型 --------
            if (episode + 1) % self.args.save_model_interval == 0:
                # print(f"======== Episode {episode + 1} Summary ========")
                # print(f"Average training reward (last {self.args.save_model_interval} episodes): {sum(total_rewards[-self.args.save_model_interval:]) / self.args.save_model_interval:.4f}")
                #
                # recent_test_indices = [i for i in range(len(test_rewards) - 1, -1, -1)
                #                        if episode - (i * test_interval) < self.args.save_model_interval]
                # if recent_test_indices:
                #     recent_test_rewards = [test_rewards[i] for i in recent_test_indices]
                #     avg_test_reward = sum(recent_test_rewards) / len(recent_test_rewards)
                #     print(f"Average test reward (recent): {avg_test_reward:.4f}")
                #
                # print(f"Average loss (last {self.args.save_model_interval} episodes): {sum(episode_losses[-self.args.save_model_interval:]) / self.args.save_model_interval:.4f}")

                checkpoints_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), check_dir)
                self.policy.save(os.path.join(checkpoints_dir, f"episode_{episode + 1}_"))

            # -------- 清空buffer，预备下次训练重新采集 --------
            self.buffer.clear()


    def sim_trajs(self):
        '''
        逐时隙运行，每个时隙内，按以下顺序执行：
          1. 每进入一个新的时隙，更新场景中车辆状态（即active_vehicles中的车辆状态，未更新时其中保存的是上一时隙场内车辆）
             更新车辆位置、剩余时间、是否离开、本地持有数据量、传输队列长度（减去上一时隙传输的部分）
             将已超出监测区域的车辆从active_vehicles中移除
             获得已出现车辆中当前时隙仍在场景内的车辆状态，即针对当前时隙的active_vehicles（还没加入当前时隙新到达的）
          2. 添加此时隙新到达的车辆（设置位置、速度），并将id添加到active_vehicles中
             至此，active_vehicles中保存的是当前时隙场景内的所有车辆
          3. 遍历active_vehicles，使每一个非新到达、仍未计算的车辆执行动作决策（新到达的车辆还未接收数据，必定不计算）
             并将其记录进该车的轨迹中
          4. 根据第3步，排除本时隙决定计算的车辆后，获得本时隙参与传输的车辆和车均带宽
             给这些参与传输的车辆更新本时隙的传输速率（速率的计算需要可用带宽）
          5. 基于本时隙及之前的历史车均带宽，给每个新到达车辆选择码率，并设置与码率相关的anytime_alpha系数
          6. 更新所有本时隙参与传输的车辆的传输队列长度（加上此时隙要传的一帧）

        逐步运行完固定时隙长度后，取所有车辆中完整的轨迹样本，放入buffer中
        '''
        states_value_range = [[np.inf, -np.inf] for _ in range(9)]

        for t in range(self.sim_timesteps):
            self.env.current_timeslot = t + 1

            # 1. 更新 active_vehicles
            self.env.update_vehicles()

            # 2. 添加此时隙新到达的车辆及其状态
            vehicles, active_vehicles, new_vehicles = self.env.arrive_new_vehicles()

            # 3. 针对当前场景中每个仍未计算的车辆，执行动作决策
            for vehicle_id in active_vehicles:
                # print('\n【vehicle_id】: ', vehicle_id)
                vehicle = vehicles[vehicle_id]

                if vehicle.is_computed:   # 如果车辆已计算，跳过
                    continue
                if vehicle_id in new_vehicles:  # 如果是新到达的车辆，跳过
                    continue

                # 获取车辆状态
                state = self.env.get_vehicle_state(vehicle_id)   # [计算能力，剩余时间，已接收帧数，下一帧的接收比例，码率，误码率下降速率alpha，与RSU空间距离，场景中其它未计算车辆的数量，这些未计算车辆的平均剩余时间]
                # print('====== state ======')
                # print('计算能力： ', state[0])
                # print('剩余时间： ', state[1])
                # print('已接收帧数： ', state[2])
                # print('下一帧的接收比例： ', state[3])
                # print('码率： ', state[4])
                # print('误码率alpha： ', state[5])
                # print('与RSU空间距离： ', state[6])
                # print('其它未计算车辆的数量： ', state[7])
                # print('其它未计算车辆的平均剩余时间： ', state[8])

                for ii in range(len(state)):                # 记录各状态变量的取值范围
                    states_value_range[ii][0] = min(states_value_range[ii][0], state[ii])
                    states_value_range[ii][1] = max(states_value_range[ii][1], state[ii])

                # 状态取值归一化（此处未完全归一化，有些变量只是根据观测的取值范围，大概使各状态变量取值都趋近于0-1
                state[0] = (state[0] - self.args.f_min) / (self.args.f_max - self.args.f_min)             # 计算能力
                state[1] = state[1] / 7                                                                    # 剩余时间
                state[2] = state[2] / 12                                                                      # 已接收帧数
                state[4] = (state[4] - min(self.args.rates_set)) / (max(self.args.rates_set) - min(self.args.rates_set))    # 码率
                state[5] = (state[5] - min(self.args.any_alpha_set)) / (max(self.args.any_alpha_set) - min(self.args.any_alpha_set))    # 误码率alpha
                state[6] = ((state[6] - vehicle.I2V_channel.get_distance(self.args.monitor_range, min(self.args.orth_pos_list))) /       # 与RSU空间距离
                            (vehicle.I2V_channel.get_distance(0, max(self.args.orth_pos_list)) -
                             vehicle.I2V_channel.get_distance(self.args.monitor_range, min(self.args.orth_pos_list))))
                state[7] = state[7] / 16                                                             # 其它未计算车辆的数量
                state[8] = state[8] / 7                                                             # 其它未计算车辆的平均剩余时间
                # print('state_scaled: ', state)

                # 取所有历史时刻的观测作为输入
                state_input = vehicle.traj.states.copy()
                state_input.append(state)

                # 计算动作概率并选择动作
                action, log_probs = self.agent.action(state_input, self.epsilon, self.args.device)
                # print('action: ', action)

                # 保存本时隙的状态、动作、奖励
                if action == 0:           # 不计算
                    vehicle.traj.add(state, action, log_probs, 0)   # reward=0
                    # vehicle.compute_step_reward(self.args)
                else:                     # 计算
                    reward = vehicle.compute_step_reward(self.args)
                    vehicle.traj.add(state, action, log_probs, reward)
                    vehicle.is_computed = True

            # 4. 给本时隙参与传输的车辆更新它们在本时隙的传输速率
            bandwidth_per_vehicle, to_trans_vehicles = self.env.update_bandwidth_history()  # 更新本时隙车均带宽、仍需传输的车辆id集合
            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].update_trans_rate(self.args.time_slot, bandwidth_per_vehicle)

            # 5. 给此时隙新到达的车辆设置码率，并设置相应的anytime_alpha系数、传输队列长度(添加一帧的数据)
            for vehicle_id in new_vehicles:
                vehicle = vehicles[vehicle_id]
                code_rate_index = self.env.select_data_rate(vehicle_id)             # 选择码率
                vehicle.code_rate = self.args.rates_set[code_rate_index]
                vehicle.anytime_alpha = self.args.any_alpha_set[code_rate_index]

            # 6. 给所有本时隙参与传输的车辆更新传输队列长度（增加一帧本时隙的数据）
            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].bits_in_queue += self.args.frame_size * 1024 * 8 * vehicles[vehicle_id].code_rate


        # print('====== state 范围 ======')
        # print('计算能力： ', states_value_range[0])
        # print('剩余时间： ', states_value_range[1])
        # print('已接收帧数： ', states_value_range[2])
        # print('下一帧的接收比例： ', states_value_range[3])
        # print('码率： ', states_value_range[4])
        # print('误码率alpha： ', states_value_range[5])
        # print('与RSU空间距离： ', states_value_range[6])
        # print('其它未计算车辆的数量： ', states_value_range[7])
        # print('其它未计算车辆的平均剩余时间： ', states_value_range[8])


        # 取该段时间内所有车辆中完整的轨迹样本，放入buffer中
        for vehicle_id in range(self.env.next_vehicle_id):
            vehicle = self.env.vehicles[vehicle_id]
            # 排除预热车辆
            if not vehicle.is_warm_up:
                checks = [sub[1] for sub in vehicle.traj.states]
                if all(a > b for a, b in zip(checks, checks[1:])):
                    # 已计算的车辆，记录其整条轨迹的reward为计算时隙的reward
                    if vehicle.is_computed:
                        vehicle.traj.compute_trajectory_reward(no_cmp_penalty=None)
                        self.buffer.store_a_traj(vehicle.traj)
                    # 未计算的车辆，包括已离开的和未离开的，未离开的轨迹不完整不记录，已离开的未能及时作出计算决策，在reward中给予惩罚项
                    else:
                        if vehicle.is_left:
                            vehicle.traj.compute_trajectory_reward(no_cmp_penalty=self.args.r_S)
                            self.buffer.store_a_traj(vehicle.traj)


    def test(self, num_test_episodes=5):
        """
        Evaluate the policy without exploration
        Returns average reward across test episodes
        """
        print("Running test evaluation...")
        test_rewards = []
        original_epsilon = self.epsilon  # Save current epsilon
        self.epsilon = 0.0  # Disable exploration during testing

        self.policy.actor.eval()

        for episode in range(num_test_episodes):
            self.env.reset()
            episode_trajs = []

            # Simulate for the same number of timesteps as in training
            self.sim_trajs_test(episode_trajs)

            # Calculate average reward for this test episode
            if episode_trajs:
                avg_reward = sum(traj.traj_reward for traj in episode_trajs) / len(episode_trajs)
                test_rewards.append(avg_reward)
                # print(f"  Test episode {episode + 1}: Average reward = {avg_reward:.4f}")

        self.epsilon = original_epsilon  # Restore original epsilon
        self.policy.actor.train()

        if test_rewards:
            mean_test_reward = sum(test_rewards) / len(test_rewards)
            print(f"Test evaluation complete. Mean test reward: {mean_test_reward:.4f}")
            return mean_test_reward
        else:
            print("No complete trajectories collected during testing.")
            return 0.0


    def sim_trajs_test(self, test_trajs):
        """
        Similar to sim_trajs but specifically for testing (no buffer storage)
        """
        for t in range(self.sim_timesteps):
            self.env.current_timeslot = t + 1

            # 1. Update active_vehicles
            self.env.update_vehicles()

            # 2. Add new vehicles
            vehicles, active_vehicles, new_vehicles = self.env.arrive_new_vehicles()

            # 3. Make decisions for each vehicle
            for vehicle_id in active_vehicles:
                vehicle = vehicles[vehicle_id]

                if vehicle.is_computed or vehicle_id in new_vehicles:
                    continue

                # Get vehicle state
                state = self.env.get_vehicle_state(vehicle_id)

                # 状态取值归一化（此处未完全归一化，有些变量只是根据观测的取值范围，大概使各状态变量取值都趋近于0-1
                state[0] = (state[0] - self.args.f_min) / (self.args.f_max - self.args.f_min)  # 计算能力
                state[1] = state[1] / 7  # 剩余时间
                state[2] = state[2] / 12  # 已接收帧数
                state[4] = (state[4] - min(self.args.rates_set)) / (
                            max(self.args.rates_set) - min(self.args.rates_set))  # 码率
                state[5] = (state[5] - min(self.args.any_alpha_set)) / (
                            max(self.args.any_alpha_set) - min(self.args.any_alpha_set))  # 误码率alpha
                state[6] = ((state[6] - vehicle.I2V_channel.get_distance(self.args.monitor_range,
                                                                         min(self.args.orth_pos_list))) /  # 与RSU空间距离
                            (vehicle.I2V_channel.get_distance(0, max(self.args.orth_pos_list)) -
                             vehicle.I2V_channel.get_distance(self.args.monitor_range, min(self.args.orth_pos_list))))
                state[7] = state[7] / 16  # 其它未计算车辆的数量
                state[8] = state[8] / 7  # 其它未计算车辆的平均剩余时间

                # 取所有历史时刻的观测作为输入
                state_input = vehicle.traj.states.copy()
                state_input.append(state)

                # Get action (with epsilon=0 for pure exploitation)
                action, log_probs = self.agent.action(state_input, self.epsilon, self.args.device)
                # print('action: ', action)

                # Record state, action, reward
                if action == 0:  # Don't compute
                    vehicle.traj.add(state, action, log_probs, 0)
                else:  # Compute
                    reward = vehicle.compute_step_reward(self.args)
                    vehicle.traj.add(state, action, log_probs, reward)
                    vehicle.is_computed = True

            # 4-6. Update bandwidth, transmission rates, etc.
            bandwidth_per_vehicle, to_trans_vehicles = self.env.update_bandwidth_history()
            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].update_trans_rate(self.args.time_slot, bandwidth_per_vehicle)

            for vehicle_id in new_vehicles:
                vehicle = vehicles[vehicle_id]
                code_rate_index = self.env.select_data_rate(vehicle_id)
                vehicle.code_rate = self.args.rates_set[code_rate_index]
                vehicle.anytime_alpha = self.args.any_alpha_set[code_rate_index]

            for vehicle_id in to_trans_vehicles:
                vehicles[vehicle_id].bits_in_queue += self.args.frame_size * 1024 * 8 * vehicles[vehicle_id].code_rate

        # Collect complete trajectories for testing
        for vehicle_id in range(self.env.next_vehicle_id):
            vehicle = self.env.vehicles[vehicle_id]
            if not vehicle.is_warm_up:
                checks = [sub[1] for sub in vehicle.traj.states]
                if all(a > b for a, b in zip(checks, checks[1:])):
                    if vehicle.is_computed:
                        vehicle.traj.compute_trajectory_reward(no_cmp_penalty=None)
                        test_trajs.append(vehicle.traj)      # 存入专门存放测试样本的列表中
                    elif vehicle.is_left:
                        vehicle.traj.compute_trajectory_reward(no_cmp_penalty=self.args.r_S)
                        test_trajs.append(vehicle.traj)


