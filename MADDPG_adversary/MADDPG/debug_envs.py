from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
import numpy as np

def test_env(env_module):
    print(f"\nTesting {env_module.__name__}")
    env = env_module.parallel_env(continuous_actions=True)
    env.reset()
    print(f"Agents: {env.agents}")
    for agent in env.agents:
        print(f"Agent {agent} obs shape: {env.observation_space(agent).shape}")
        print(f"Agent {agent} action shape: {env.action_space(agent).shape}")
    env.close()

if __name__ == "__main__":
    test_env(simple_spread_v3)
    test_env(simple_adversary_v3)
    test_env(simple_tag_v3)
