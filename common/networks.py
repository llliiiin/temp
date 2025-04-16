"""
这个文件用来存储不同的网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent.helpers import SinusoidalPosEmb



class MLP(nn.Module):  # q-network：行为控制网络，该网络也是最终被应用的网络
    def __init__(self, state_dim, hidden_dim, n_actions):  # n_actions是当前agent的动作维度大小
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden layer
        self.fc3 = nn.Linear(hidden_dim, n_actions)  # output layer

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out                  # n_actions维动作所对应的值



class MLP_diff(nn.Module):
    def __init__(
        self,
        state_dim,  # 状态维度
        action_dim,  # 动作维度
        hidden_dim,  # diffusion模型中用于学习噪声的MLP中隐藏层的神经元个数
        t_dim = 16,  # denoising step 位置编码的维度大小
    ):
        super(MLP, self).__init__()
        _act = nn.ReLU  # 激活函数
        # self.state_mlp = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     _act(),
        #     nn.Linear(hidden_dim, state_dim)
        # )
        # 因为 denoising step 在reverse process中与位置（第几步）有关，所以利用一个正弦余弦位置编码器，获得 denoising step 的位置编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        # self.mid_layer = nn.Sequential(
        #     nn.Linear(state_dim + action_dim + t_dim, hidden_dim),
        #     _act(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     _act(),
        #     nn.Linear(hidden_dim, state_dim + action_dim + t_dim)
        # )
        self.fc1 = nn.Linear(state_dim + action_dim + t_dim, hidden_dim)  # 输入：环境状态，动作分布x，denoising step的位置编码
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出：噪声（维度和动作维度相同）

    def forward(self, x, time, state):
        # processed_state = self.state_mlp(state)
        # processed_state = state * 1000  # Hongyang乘1000是为了让后续x，t，processed_state处于一个量级
        t = self.time_mlp(time)
        obs = torch.cat([x, t, state], dim=1)  # 输入：环境状态，动作分布x，denoising step的位置编码
        # obs = self.mid_layer(obs)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        q = self.fc3(obs)  # 输出：噪声（维度和动作维度相同），用q的意思是当到解噪的最后一步时，输出的就是不同动作的q值了
        return q


class QMixNet(nn.Module):  # 作为中心式训练网络，接收所有agent的Q值和当前全局状态st，输出st下所有agent联合行为u的行为效用值Qtot
    def __init__(self, state_shape, hyper_hidden_dim, n_agents, qmix_hidden_dim):
        # state_shape是当前全局状态st的维度大小
        # hyper_hidden_dim是参数生成网络隐藏层中的神经元个数
        # n_agents是智能体的个数
        # qmix_hidden_dim是推理网络隐藏层中的神经元个数
        self.n_agents = n_agents
        self.qmix_hidden_dim = qmix_hidden_dim

        super(QMixNet, self).__init__()

        # hyper_w1 网络用于输出推理网络中的第一层神经元所需的weights，共n_agents*qmix_hidden_dim个
        self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hyper_hidden_dim, n_agents * qmix_hidden_dim))
        # hyper_w2 生成推理网络需要的从隐层到输出Qtot的所有weights，共qmix_hidden个
        self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hyper_hidden_dim, qmix_hidden_dim))

        # hyper_b1 生成第一层网络对应维度的偏差bias，共qmix_hidden_dim个
        self.hyper_b1 = nn.Linear(state_shape, qmix_hidden_dim)
        # hyper_b2 生成对应从隐层到输出 Qtot 值层的 bias，共1个
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(qmix_hidden_dim, 1))

    def forward(self, q_values, states):  # 因为我们场景中每个agent网络是纯MLP，并没有利用RNN了，所以与时间无关，因此：
        # states的shape为state_shape，总观察状态的维度大小
        # 传入的q_values的shape是n_agents，智能体的个数

        # 对 state 做标准化处理 使得更好的训练和收敛 其中标准化的数据由之前收集可得
        states_processed = states / 1000

        w1 = torch.abs(self.hyper_w1(states_processed))  # abs是为了确保Qtot对每个agent的Q单调递增，即求导后恒大于零
        b1 = self.hyper_b1(states_processed)  # bias不要求必须正，因为求导后bias就没了

        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵
        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)  # 维度：智能体数目 x qmix隐藏层神经元个数
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)
        # 将q_values也转化为3维
        q_values = q_values.view(-1, 1, self.n_agents)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # 推理网络只含有一个隐藏层hidden

        w2 = torch.abs(self.hyper_w2(states_processed))  # abs是为了确保Qtot对每个agent的Q单调递增，即求导后恒大于零
        b2 = self.hyper_b2(states_processed)  # bias不要求必须正，因为求导后bias就没了

        # 因为生成的hyper_w2需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # 维度： qmix隐藏层神经元个数 x 1
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.squeeze(1)  # 将q_total的维度转为1,即1个scalar
        return q_total  # 输出st下所有agent联合行为u的行为效用值Qtot
