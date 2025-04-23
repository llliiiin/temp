"""
这个文件负责统筹规划QMIX里的所有agents
"""

import torch
import numpy as np
import torch.nn.functional as F





class Agent(object):
    def __init__(self, state_dim, action_dim, actor):
        self.state_dim = state_dim          # 状态维度大小
        self.action_dim = action_dim              # 动作维度大小
        self.mlp_hidden_width = 256             # 每diffusion网络中隐藏层的神经元个数
        self.actor = actor


    def action(self, obs, epsilon, device):
        '''根据状态生成一个动作，返回所选择的动作和所有动作的概率'''
        input = torch.tensor(obs, dtype=torch.float).to(device)
        # input = torch.unsqueeze(input, 0)  # inputs是当前智能体的状态,转为tensor

        logits = self.actor.forward(input)          # 获得网络产生的结果（未归一化）
        probs = F.softmax(logits, dim=-1)


        # 通过epsilon-greedy的方式选择动作，增大探索
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.action_dim)  # action是一个整数
        else:
            action = torch.argmax(logits)
            action = action.item()                      # 将tensor的action值转为int,作为动作索引

        log_probs = torch.log(probs[action] + 1e-8)       # 计算log(pi(a|s)), 1e-8为了防止log(0)

        return action, log_probs




