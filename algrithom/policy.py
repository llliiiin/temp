"""
这个文件负责xxx
"""
import torch
from agent.agent import Agent
import torch.nn as nn
import random


class REINFORCE:
    def __init__(self, state_shape, action_shape, optimizer):
        self.state_shape = state_shape      # 状态维度大小
        self.n_actions = action_shape          # 动作维度大小
        self.subsample_ratio = 0.8      # 从一段轨迹中截取时间步的比例 Capital{\tau}
        self.reward_norm_mode = 'batch'
        self.grad_norm_clip = 10  # 防止梯度爆炸
        self.gamma = 1  # 奖励的折扣因子

        self.optimizer = optimizer
        self.MseLoss = nn.MSELoss()


    def learn(self, trajs, buffer):

        # 计算奖励归一化的均值和方差
        if self.reward_norm_mode == "buffer":
            all_rewards = torch.tensor(buffer.all_traj_rewards(), dtype=torch.float32)
            mean = all_rewards.mean()
            std = all_rewards.std() + 1e-8
        else:  # "batch"
            mean = std = None

        # 计算loss
        loss = compute_policy_gradient_loss(trajs, mean=mean, std=std, gamma=self.gamma, subsample_ratio=self.subsample_ratio)

        # 利用梯度下降更新参数
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_norm_clip)  # 防止梯度爆炸
        self.optimizer.step()

    def save(self, path):
        # 保存agent的网络
        filename = f"agent.pth"
        torch.save(self.agent.actor.state_dict(), path + filename)
        print("====================================")
        print("model has been saved...")

    def load(self, path):
        # 将agent的网络load进来
        filename = f"agent.pth"
        self.agent.actor.load_state_dict(torch.load(path + filename))
        print("====================================")
        print("model has been loaded...")



def compute_policy_gradient_loss(trajs, mean=None, std=None, subsample_ratio=None, gamma=1.0):

    # -------- 统计轨迹级奖励并归一化 --------
    traj_rewards = torch.tensor([traj.trajectory_reward() for traj in trajs], dtype=torch.float32)
    if mean is None or std is None:
        mean = traj_rewards.mean()
        std = traj_rewards.std()
    std = std + 1e-8                    # 防止除零
    norm_rewards = (traj_rewards - mean) / std          # shape: (N_traj,)

    # -------- 计算损失 --------
    losses = []
    for traj, R_norm in zip(trajs, norm_rewards):
        T = traj.length()
        if subsample_ratio is not None and 0 < subsample_ratio < 1.0:
            k = max(1, int(T * subsample_ratio))
            idx = random.sample(range(T), k)
            weight = T / k
        else:
            idx = range(T)
            weight = 1.0
        for i in idx:
            losses.append(-traj.log_probs[i] * (R_norm * weight))
    return torch.stack(losses).mean()