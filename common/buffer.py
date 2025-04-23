"""
这个文件用来定义单条轨迹类，和buffer类
"""

import numpy as np
import torch
import random
from collections import deque



class Trajectory:
    '''一条轨迹，即一段连续时间步的状态、动作、奖励'''
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []          # 每一个时隙输出的动作概率的对数值（后续梯度下降计算用）
        self.rewards = []           # 存每一个时隙的reward，但仅最后一步非零
        self.traj_reward = None

    def add(self, state, action, log_prob, reward):    # 每一个时隙的信息添加
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def compute_perslot_rewards(self, gamma=1.0):
        # 计算每一个时隙的reward，用最后一个时隙的reward为前面时隙赋值，gamma=1表示同等赋值，gamma<1表示衰减赋值
        R = 0.0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def compute_trajectory_reward(self, no_cmp_penalty=None):
        if no_cmp_penalty == None:
            assert self.rewards[-1] != 0, "标记is_computed的车辆没有计算reward（reward=0）"
            self.traj_reward = self.rewards[-1]
        else:
            assert sum(self.rewards) == 0, "未标记is_computed的车辆有非零reward"
            self.traj_reward = - no_cmp_penalty

    def length(self):
        return len(self.states)





class TrajectoryBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def store_a_traj(self, traj):
        self.buffer.append(traj)

    def sample(self, batch_size=20):
        if batch_size >= len(self.buffer):
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def all_traj_rewards(self):
        return [traj.traj_reward for traj in self.buffer]

    def ready(self, batch_size=20):  # 判断当前收集的样本个数是否足够一个batch
        if len(self.buffer) >= batch_size:
            return True
        else:
            return False

    def clear(self):
        self.buffer.clear()

    def get_length(self):
        return len(self.buffer)


