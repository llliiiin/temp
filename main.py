import argparse
import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from common.networks import MLP_diff, MLP, MLP_Norm, En_MLP, Feature_Attn_MLP, En_Feature_Attn_MLP, RNN, RNN_AddAtte, RNN_Multihead_SelfAtte
from common.buffer import TrajectoryBuffer
from common.utils import seed_set, create_logger, log_settings
from common.env import Env
from algrithom.training import Trainer, Trainer_Rnn
from algrithom.policy import REINFORCE
from agent.diffusion import Diffusion
from agent.agent import Agent
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)      # 128
    parser.add_argument("--embed_dim", type=int, default=32)        # 64
    parser.add_argument('--net_type', type=str, default='LSTM_Multihead_SelfAtte',
                        choices=['MLP', 'MLP_Norm', 'diffusion', 'En_MLP', 'Feature_Attn_MLP', 'En_Feature_Attn_MLP', 'GRU', 'LSTM',
                                 'GRU_AddAtte','LSTM_AddAtte','GRU_Multihead_SelfAtte','LSTM_Multihead_SelfAtte'])
    parser.add_argument('--net_layers', type=int, default=1)         # 3
    parser.add_argument('--num_heads', type=int, default=1)         # 4
    parser.add_argument('--dropout', type=int, default=0.1)       # 0.1
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--buffer_size', type=int, default=1e4)      #1e6
    parser.add_argument('--num_episode', type=int, default=3000)           # 1000, 5000
    parser.add_argument('--batch_size', type=int, default=256)              # 512
    parser.add_argument('--wd', type=float, default=1e-6)                    # 1e-4  越小曲线越稳定
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--actor-lr', type=float, default=3e-4)      # 3e-4
    parser.add_argument('--rates_set', type=list, default=[3/2, 2/1, 5/2, 3/1, 4/1])      # 可选码率集合，需为升序排列
    parser.add_argument('--any_alpha_set', type=list, default=[0.1, 0.3, 0.9, 1.5, 1.8])  # 码率对应的anytime因子集合，需与rates_set对应
    parser.add_argument('--diff_step_set', type=list, default=list(range(3, 25, 1)))  # 可选扩散步数集合  **********
    parser.add_argument('--bandwidth', type=float, default=2*1e6)              # (Hz，1MHz=1e6Hz)   【2,5,10-90】
    parser.add_argument('--monitor_range', type=float, default=150)           # 监控范围（m）       【100-250】
    parser.add_argument('--orth_pos_list', type=list, default=[3,6,9])      # 车辆与RSU在垂直方向的距离
    parser.add_argument('--RSU_trans_power', type=float, default=13)       # RSU的发射功率（dBm）    【..-60】23
    parser.add_argument('--noise_power', type=float, default=-114)           # 噪声功率（dBm）
    parser.add_argument('--state_dim', type=int, default=9)
    parser.add_argument('--action_dim', type=int, default=2)
    parser.add_argument('--sim_timesteps', type=int, default=60)            # 每次采样时，模拟的时隙数量
    parser.add_argument('--time_slot', type=int, default=0.5)                 # 时隙长度（s）
    parser.add_argument('--epsilon', type=float, default=1)               # 选择动作时epsilon-greedy的参数
    parser.add_argument('--epsilon_decay', type=float, default=0.995)         # 每轮训练后epsilon衰减率
    parser.add_argument('--epsilon_decay_interval', type=float, default=2)
    parser.add_argument('--epsilon_decay_type', type=str, default='multi', choices=['multi', 'linear'])
    parser.add_argument('--frame_size', type=float, default=400)            # 帧大小（KB）       【几百？】
    parser.add_argument('--f_min', type=float, default=4)                     # 车辆的本地计算能力范围 (GHz)  【200M，1G，4-8G】
    parser.add_argument('--f_max', type=float, default=6)
    parser.add_argument('--r_T', type=float, default=50)                     # 车辆计算结束时刻超过离开时刻的reward惩罚项系数
    parser.add_argument('--r_S', type=float, default=80)                     # 车辆直至到达路口都没有执行计算的reward惩罚项
    parser.add_argument('--lamda', type=float, default=1)               # 目标函数比例系数 （误差 + lamda * 传输量）
    parser.add_argument('--warm_up_vehicles', type=float, default=60)   # 预热车辆数量，这些车为仿真初始时段的车辆，环境不稳定，轨迹不计入训练
    parser.add_argument('--G0', type=float, default=1.6)                 # 预测模型一步扩散的计算量  (xx G次运算)
    parser.add_argument('--save_model_interval', type=int, default=200)
    parser.add_argument('--test_interval', type=int, default=20)
    parser.add_argument('--v_min', type=float, default=20)               # 车辆速度  22m/s浮动
    parser.add_argument('--v_max', type=float, default=30)
    parser.add_argument('--vehicle_arrival_rate', type=int, default=2)      # 车辆泊松到达过程中，平均每时隙到达率(veh/slot)
    parser.add_argument('--beta', type=float, default=0.5)               # anytime误码率参数
    parser.add_argument('--a', type=float, default=2)                   # 计算车辆紧迫性权重的参数
    parser.add_argument('--gamma', type=float, default=0.5)              # 计算码率倾向因子的放缩参数，＜1
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = get_args()

    # selected_seed = args.seed
    selected_seed = np.random.randint(0, 2 ** 30 - 1)
    args.seed = selected_seed
    seed_set(selected_seed)


    # -------- create environment --------
    env = Env(args)


    # -------- create actor and optimizer --------
    if args.net_type == 'MLP':
        actor = MLP(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'MLP_Norm':
        actor = MLP_Norm(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, dropout=0.1, num_layers=args.net_layers).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'En_MLP':
        actor = En_MLP(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, n_heads=args.num_heads, dropout=0.1, num_layers=args.net_layers).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'Feature_Attn_MLP':
        actor = Feature_Attn_MLP(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, n_heads=args.num_heads, dropout=args.dropout).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'En_Feature_Attn_MLP':
        actor = En_Feature_Attn_MLP(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, n_heads=args.num_heads, dropout=args.dropout, num_layers=args.net_layers).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'diffusion':
        # 用于在 reverse process 中产生噪声的 MLP
        actor_net = MLP_diff(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim).to(args.device)
        # 配置diffusion model
        actor = Diffusion(state_dim=args.state_dim, action_dim=args.action_dim, model=actor_net, n_timesteps=15)    # 解噪步数候选集{3,5,7,9,11} 备选{3，6，9，12，15}
        optimizer = torch.optim.Adam(
            actor_net.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'GRU':
        actor = RNN(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, rnn_type='GRU', dropout=args.dropout, num_layers=args.net_layers).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'LSTM':
        actor = RNN(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, rnn_type='LSTM', dropout=args.dropout, num_layers=args.net_layers).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'GRU_AddAtte':
        actor = RNN_AddAtte(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, rnn_type='GRU', dropout=args.dropout, num_layers=args.net_layers).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'LSTM_AddAtte':
        actor = RNN_AddAtte(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, rnn_type='LSTM', dropout=args.dropout, num_layers=args.net_layers).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'GRU_Multihead_SelfAtte':
        actor = RNN_Multihead_SelfAtte(state_dim=args.state_dim, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, rnn_type='GRU', dropout=args.dropout, num_layers=args.net_layers, num_heads=args.num_heads).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    elif args.net_type == 'LSTM_Multihead_SelfAtte':
        actor = RNN_Multihead_SelfAtte(state_dim=args.state_dim, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim, rnn_type='LSTM', dropout=args.dropout, num_layers=args.net_layers, num_heads=args.num_heads).to(args.device)
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.wd
        )
    else:
        raise ValueError("Do not have this net_type:{}".format(args.net_type))

    # -------- create agent --------
    agent = Agent(args.state_dim, args.action_dim, actor)


    # -------- create policy --------
    policy = REINFORCE(args.state_dim, args.action_dim, optimizer, actor)

    # # load a previous policy
    # if args.resume_path:
    #     ckpt = torch.load(args.resume_path, map_location=args.device)
    #     policy.load_state_dict(ckpt)
    #     print("Loaded agent from: ", args.resume_path)


    # -------- create buffer --------
    buffer = TrajectoryBuffer(args.buffer_size)


    # -------- trainer --------
    if args.net_type == 'GRU' or args.net_type == 'LSTM' or args.net_type == 'GRU_AddAtte' or args.net_type == 'LSTM_AddAtte' or args.net_type == 'GRU_Multihead_SelfAtte' or args.net_type == 'LSTM_Multihead_SelfAtte':
        trainer = Trainer_Rnn(policy, env, buffer, agent, args)
    else:
        trainer = Trainer(policy, env, buffer, agent, args)

    # -------- logging information --------
    check_dir = f"checkpoints/{args.net_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # os.makedirs(check_dir, exist_ok=True)
    log_dir = f"runs/{args.net_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.log_dir = log_dir
    writer = SummaryWriter(log_dir)     # 初始化 TensorBoard 日志记录器，用于可视化 训练过程
    logger = create_logger(os.path.join(log_dir, 'log.txt'))    # 创建日志记录器（logger），把参数保存到log.txt文件中
    log_settings(logger, args)
    trainer.writer = writer

    # -------- training --------
    trainer.train(check_dir)

    writer.close()


