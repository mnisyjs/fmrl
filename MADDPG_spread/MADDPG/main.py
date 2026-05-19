from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
import os

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
    # get the params
    print("START RUNNING")
    args = get_args()
    set_seed(args.seed)
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
