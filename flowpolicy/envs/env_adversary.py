from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3


def make_env(args):
    if args.scenario_name == "simple_spread":
        env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=args.max_episode_len,
                                            continuous_actions=True)
    elif args.scenario_name == "simple_adversary":
        # 1个对手, 2个好人, 2个地标
        env = simple_adversary_v3.parallel_env(num_adversaries=1, num_good=2, num_landmarks=2,
                                               max_cycles=args.max_episode_len, continuous_actions=True)
    else:
        raise NotImplementedError

    env.reset()
    args.n_agents = len(env.agents)
    # 重要：对抗环境下每个 agent 的观测维度可能不同，必须记录为列表
    args.obs_shape = [env.observation_space(agent).shape[0] for agent in env.agents]
    args.action_shape = [env.action_space(agent).shape[0] for agent in env.agents]
    args.high_action = 1.0
    return env, args