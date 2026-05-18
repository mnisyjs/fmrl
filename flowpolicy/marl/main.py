import numpy as np
import random
import torch
import os
from common.arguments import get_args
from envs.env_mpe import make_env


def make_runner(args, env):
    if args.scenario_name == "simple_spread":
        from runner import Runner
    elif args.scenario_name == "simple_adversary":
        from runner_adversary import Runner
    else:
        raise ValueError("Unknown env")

    return Runner(args, env)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"检测到随机种子: {seed}，已锁定全局环境。")


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed)

    env, args = make_env(args)

    runner = make_runner(args, env)

    runner.run()