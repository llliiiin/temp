"""
这个文件用来存储通用的一些函数
"""
import os
import random
import torch
import numpy as np
import logging
import math

def seed_set(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # PyTorch种子设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时用

    # 关键配置：禁用非确定性算法
    # torch.backends.cudnn.deterministic = True         # 可没有
    # torch.backends.cudnn.benchmark = False  # 固定卷积算法，可没有

    # 启用确定性算法模式，需要有
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或 ":16:8"
    torch.use_deterministic_algorithms(True)


# def calculate_prediction_error(frame_num, alpha, time_remained, F, timeslot):
#     x = frame_num
#     y = alpha
#     # 二次多项式
#     f1 = (1.166403 + -0.064556 * x + -0.063117 * y + 0.003125 * x*y + 0.005111 * x**2 + 0.032678 * y**2
#           + -0.000140 * x**3 + -0.011636 * y**3 + -0.000085 * x*x*y + 0.000092 * x*y*y)
#     f1 = f1 * 3                                    # 放大
#     f1 = f1 + int(x==0) * 2                   # 当frame_num = 0时，多加一点惩罚
#     # sigmoid
#     f2 = 0.8422 + 0.1076 / (1 + np.exp(-0.0409 * (time_remained / timeslot - 62.8610)))
#     f2 = f2 * 3
#     # 指数函数（双指数函数）
#     f3 = 10 * np.exp(-2.44 * F) + 0.835513 * np.exp(-0.203203 * F) + 1.361693
#     f3 = f3 * 3

#     print('f(frame_num,alpha)={:.3f}, f(time_remained)={:.3f}, f(F)={:.3f}.'.format(f1, f2, f3))
#     return f1 * f2 * f3



def calculate_prediction_error(frame_num, alpha, time_remained, F, timeslot=0.5):

    # 1) 归一化 frame_num：0→1，20→0
    f1 = ((15.0 - frame_num) / (15.0-1)) ** 2

    # 2) 归一化 alpha：0.1→1，1.8→0
    #    alpha_range = 1.8 - 0.1 = 1.7
    f2 = (1.8 - alpha) / (1.7-0.1)

    # 3) 归一化 F：3→1，24→0
    #    F_range = 24 - 3 = 21
    f3 = ((24 - F) / (21.0-1)) ** 2

    # 4) 归一化 time_remained：0→0，15→1
    #    (timeslot=0.5 对归一化不起影响，因为 time_remained/timeslot∈(0,30)，再除以30即 time_remained/15)
    f4 = (time_remained / (5.0-0.5)) ** 2

    # 5) 合并：简单双向平均
    score = (f1 + f2 + f3 + f4) / 4.0

    # print('f(frame_num)={:.3f}, f(alpha)={:.3f}, f(F)={:.3f}, f(time_remained)={:.3f}.'.format(f1, f2, f3, f4))
    return 20 * score * 4           # * 4



# def calculate_prediction_error(frame_num, alpha, time_remained, F, timeslot):
#
#     # f1: decreases linearly from 1 at frame_num=0 to ~0.048 at frame_num=20
#     f1 = (10 - frame_num)
#
#     # f2: decreases linearly from ~0.95 at alpha=0.1 to ~0.10 at alpha=1.8
#     f2 = (2 - alpha) / 1
#
#     # f3: increases proportional to time_remained/timeslot, normalized by 30
#     f3 = (time_remained * 0.5 / timeslot)
#
#     # f4: decreases linearly with F, scaled so the max product ≈30
#     f4 =  (25 - F)
#
#     print('f(frame_num)={:.3f}, f(alpha)={:.3f}, f(time_remained)={:.3f}, f(F)={:.3f}.'.format(f1, f2, f3, f4))
#
#     return f1 * f2 * f3 * f4






def create_logger(filename, file_handle=True):
    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger

def log_settings(logger, args):
    logger.info("======= Experiment Settings =======")
    for key in sorted(vars(args).keys()):
        logger.info(f"{key}: {getattr(args, key)}")
    logger.info("===================================")

