from pettingzoo.mpe import simple_spread_v3

def make_env(args):
    env = simple_spread_v3.parallel_env(
        N=3,  # 先写死或从 scenario 推断
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions = True
    )
    env.reset(seed=args.seed)
    args.n_agents = len(env.agents)
    print(env.action_space(env.agents[0]))
    # obs / action 维度
    obs_shape = []
    action_shape = []
    for agent in env.agents:
        obs_shape.append(env.observation_space(agent).shape[0])
        action_shape.append(env.action_space(agent).shape[0])
    args.obs_shape = obs_shape
    args.action_shape = action_shape
    # 动作上下界
    action_space = env.action_space(env.agents[0])
    args.high_action = action_space.high[0]
    args.low_action = action_space.low[0]
    return env, args
