"""
获取 Simple Adversary 连续动作空间的完整信息
"""
from pettingzoo.mpe import simple_adversary_v3
import numpy as np

def get_adversary_info():
    print("="*60)
    print("Simple Adversary 环境信息（连续动作）")
    print("="*60)

    # 测试不同的配置
    configurations = [{"N": 2, "desc": "2个合作方（默认配置）"}]

    for config in configurations:
        print(f"\n--- 配置: {config['desc']} (N={config['N']}) ---")

        env = simple_adversary_v3.parallel_env(
            N=config['N'],
            continuous_actions=True,
            max_cycles=25
        )
        env.reset()

        print(f"总智能体数量: {len(env.agents)}")
        print(f"智能体列表: {env.agents}")

        # 分析智能体类型
        adversary_count = sum(1 for a in env.agents if 'adversary' in a)
        agent_count = len(env.agents) - adversary_count
        print(f"  - 对手数量: {adversary_count}")
        print(f"  - 合作方数量: {agent_count}")

        # 获取动作空间信息
        for agent in env.agents:
            action_space = env.action_space(agent)
            print(f"\n  智能体 [{agent}] 动作空间:")
            print(f"    类型: {type(action_space).__name__}")
            print(f"    形状: {action_space.shape}")
            print(f"    下界 (low): {action_space.low}")
            print(f"    上界 (high): {action_space.high}")
            print(f"    数据类型: {action_space.dtype}")

        # 获取观察空间信息
        print(f"\n  观察空间信息（以 {env.agents[0]} 为例）:")
        obs_space = env.observation_space(env.agents[0])
        print(f"    形状: {obs_space.shape}")
        print(f"    下界: {obs_space.low[:5]}... (前5维)")
        print(f"    上界: {obs_space.high[:5]}... (前5维)")

        env.close()

    # 测试实际的动作采样
    print("\n" + "="*60)
    print("实际动作采样示例")
    print("="*60)

    env = simple_adversary_v3.parallel_env(N=2, continuous_actions=True)
    env.reset()

    print("\n随机采样的连续动作:")
    for agent in env.agents:
        action = env.action_space(agent).sample()
        print(f"  {agent}: {action}")

    # 测试边界动作
    print("\n边界动作测试:")
    for agent in env.agents:
        action_space = env.action_space(agent)
        min_action = action_space.low
        max_action = action_space.high
        print(f"  {agent}: 最小值 {min_action}, 最大值 {max_action}")

    env.close()

if __name__ == "__main__":
    get_adversary_info()