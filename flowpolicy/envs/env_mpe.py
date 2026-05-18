from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3

def make_env(args):
    if args.scenario_name == "simple_spread":
        env = simple_spread_v3.parallel_env(
            N=3,  # 先写死或从 scenario 推断
            local_ratio=0.5,
            max_cycles=25,
            continuous_actions = True
        )
    elif args.scenario_name == "simple_adversary":
        # 1个对手, 2个好人, 2个地标
        env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True)
    else:
        raise NotImplementedError

    env.reset()
    args.n_agents = len(env.agents)
    args.adversary_indices = [i for i, agent in enumerate(env.agents) if "adversary" in agent]
    args.good_indices = [i for i, agent in enumerate(env.agents) if "agent" in agent]

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
    args.high_action = 1.0
    return env, args
