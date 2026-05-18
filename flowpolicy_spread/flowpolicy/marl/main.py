from common.arguments import get_args
from envs.env_mpe import make_env
from runner import Runner
load_path = None

args = get_args()
env, args = make_env(args)

runner = Runner(args, env)
if load_path is not None:
    runner.maddpg.load(load_path)
runner.run()
