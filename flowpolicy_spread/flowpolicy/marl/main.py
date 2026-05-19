from common.arguments import get_args
from envs.env_mpe import make_env
from runner import Runner
import numpy as np
import random
import torch
import os
load_path = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 针对 CuDNN 的确定性设置（保证在 GPU 上跑的结果也完全一致）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    env, args = make_env(args)

    runner = Runner(args, env)
    if load_path is not None:
        runner.maddpg.load(load_path)
    runner.run()
