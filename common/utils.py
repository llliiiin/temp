"""
这个文件用来存储通用的一些函数
"""
import os
import random
import torch
import numpy as np
from pathlib import Path

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



def save_results(rewards, cveh_num, sveh_num, path='./results'):
    filename = f"rewards_cv{cveh_num}_sv{sveh_num}.npy"
    np.save(path + filename, rewards)
    print("====================================")
    print('result has been saved...')
    print("====================================")



def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
