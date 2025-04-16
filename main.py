import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from common.networks import MLP, MLP_diff
from common.buffer import TrajectoryBuffer
from algrithom.training import Trainer
from algrithom.policy import REINFORCE
from agent.diffusion import Diffusion
from agent.agent import Agent
from common.utils import seed_set
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from env import make_aigc_env
import warnings

warnings.filterwarnings('ignore')
# python benchmark/dqn.py

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument('--net_type', type=str, default='MLP', choices=['MLP', 'diffusion'])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--buffer_size', type=int, default=1e6)      #1e6
    parser.add_argument('--num_epochs', type=int, default=5000)           # 1000
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--actor-lr', type=float, default=3e-7)      # 1e-3

    parser.add_argument("--exploration-noise", type=float, default=0.01)
    parser.add_argument('--step-per-epoch', type=int, default=100)# 100
    parser.add_argument('--step-per-collect', type=int, default=1)#1000
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')
    # for diffusion discrete sac
    parser.add_argument('--critic-lr', type=float, default=3e-7)
    parser.add_argument('--alpha', type=float, default=0.05)  # none for action entropy
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=1)  # for diffusion chain
    parser.add_argument('--beta-schedule', type=str, default='vp', choices=['linear', 'cosine', 'vp'])
    parser.add_argument('--pg-coef', type=float, default=1.)
    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.6)#0.6
    parser.add_argument('--prior-beta', type=float, default=0.4)#0.4

    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = get_args()

    selected_seed = args.seed
    # selected_seed = np.random.randint(0, 2 ** 32 - 1)
    # print(">>> seed:", selected_seed)
    seed_set(selected_seed)

    # -------- create environment --------
    '''【to do】'''
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]



    # -------- create actor and optimizer --------
    if args.net_type == 'diffusion':
        # 用于在 reverse process 中产生噪声的 MLP
        actor_net = MLP_diff(state_dim=args.state_shape, hidden_dim=args.mlp_hidden_dim, action_dim=args.action_shape)
        # 配置diffusion model
        actor = Diffusion(state_dim=args.state_shape, action_dim=args.action_shape, model=actor_net, n_timesteps=15)    # 解噪步数候选集{3,5,7,9,11} 备选{3，6，9，12，15}
        actor_optim = torch.optim.Adam(
            actor_net.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'MLP':
        actor = MLP(state_dim=args.state_shape, hidden_dim=args.mlp_hidden_dim, action_dim=args.action_shape)
        actor_optim = torch.optim.Adam(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )

    # -------- create agent --------
    agent = Agent(args.state_shape, args.action_shape, actor)


    # -------- create policy --------
    policy = REINFORCE(args.state_shape, args.action_shape, actor_optim)

    # # load a previous policy
    # if args.resume_path:
    #     ckpt = torch.load(args.resume_path, map_location=args.device)
    #     policy.load_state_dict(ckpt)
    #     print("Loaded agent from: ", args.resume_path)


    # -------- create buffer --------
    buffer = TrajectoryBuffer(args.buffer_size)


    # -------- trainer --------
    trainer = Trainer(policy, env, buffer, args.batch_size, args.num_epochs)
    trainer.train()



