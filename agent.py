"""
这个文件负责统筹规划QMIX里的所有agents
"""

import torch
import numpy as np
import torch.nn.functional as F
from common.networks import MLP, MLP_diff
from .diffusion import Diffusion




class Agent(object):
    def __init__(self, state_shape, n_actions, actor):
        self.state_shape = state_shape          # 状态维度大小
        self.n_actions = n_actions              # 动作维度大小
        self.mlp_hidden_width = 256             # 每diffusion网络中隐藏层的神经元个数
        self.actor = actor


    def action(self, obs, epsilon):
        '''根据状态生成一个动作，返回所选择的动作和所有动作的概率'''
        inputs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)  # inputs是当前智能体的状态（全局状态）,转为tensor

        logits = self.actor.forward(inputs)          # 获得diffusion网络产生的结果（未归一化）
        probs = F.softmax(logits, dim=-1)


        # 通过epsilon-greedy的方式选择动作，增大探索
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.n_actions)  # action是一个整数
        else:
            action = torch.argmax(logits)
            action = action.item()                      # 将tensor的action值转为int,作为动作索引

        log_logits = torch.log(probs[action] + 1e-8)       # 计算log(pi(a|s)), 1e-8为了防止log(0)

        return action, log_logits




