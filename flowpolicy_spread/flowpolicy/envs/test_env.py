from mpe2 import simple_spread_v3

# 测试离散版本
env_discrete = simple_spread_v3.parallel_env(N=3)
print("离散版本:")
print(f"动作空间类型: {type(env_discrete.action_space('agent_0'))}")
print(f"动作空间: {env_discrete.action_space('agent_0')}")

# 测试连续版本
env_continuous = simple_spread_v3.parallel_env(N=3, continuous_actions=True)
print("\n连续版本:")
print(f"动作空间类型: {type(env_continuous.action_space('agent_0'))}")
print(f"动作空间: {env_continuous.action_space('agent_0')}")