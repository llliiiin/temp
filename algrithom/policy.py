"""
这个文件负责xxx
"""
import torch
from agent.agent import Agent
import torch.nn as nn
import random


class REINFORCE:
    def __init__(self, state_shape, action_shape, optimizer, actor):
        self.state_shape = state_shape      # 状态维度大小
        self.n_actions = action_shape          # 动作维度大小
        self.subsample_ratio = 0.8      # 从一段轨迹中截取时间步的比例 Capital{\tau}
        self.reward_norm_mode = 'batch'
        self.grad_norm_clip = 10  # 防止梯度爆炸
        self.actor = actor
        self.optimizer = optimizer
        self.MseLoss = nn.MSELoss()


    def learn(self, trajs, buffer, device):

        # 计算奖励归一化的均值和方差
        if self.reward_norm_mode == "buffer":
            all_rewards = torch.tensor(buffer.all_traj_rewards(), dtype=torch.float32, device=device)
            mean = all_rewards.mean()
            std = all_rewards.std() + 1e-8
        else:  # "batch"
            mean = std = None

        # 计算loss
        loss = compute_policy_gradient_loss(trajs, mean=mean, std=std, subsample_ratio=self.subsample_ratio, device=device)

        # 利用梯度下降更新参数
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)  # 防止梯度爆炸
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        # 保存agent的网络
        filename = f"agent.pth"
        torch.save(self.actor.state_dict(), path + filename)
        print("====================================")
        print("model has been saved...")
        print("====================================")

    def load(self, path):
        # 将agent的网络load进来
        filename = f"agent.pth"
        self.agent.actor.load_state_dict(torch.load(path + filename))
        print("====================================")
        print("model has been loaded...")



def compute_policy_gradient_loss(trajs, mean=None, std=None, subsample_ratio=None, device=None):

    # -------- 统计轨迹级奖励并归一化 --------
    traj_rewards = torch.tensor([traj.traj_reward for traj in trajs], dtype=torch.float32, device=device)
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