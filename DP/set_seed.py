import os
import torch
import numpy as np
import random
import time




def set_seed(seed=42):
    """
    设置全局随机种子以确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 以下设置确保 GPU 算法的确定性（但可能会略微降低模拟速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 环境变量设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"系统种子已设置为: {seed} | 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")