import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_adversary", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--max-episodes", type=int, default=50000)
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-flow", type=float, default=1e-4, help="learning rate of flow actor")
    parser.add_argument("--lr-critic", type=float, default=2e-4, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy, how to make it suit our model")
    parser.add_argument("--noise_rate", type=float, default=0.1,
                        help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.006, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
                        help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--alpha_policy_loss", type=float, default=0.3)
    parser.add_argument("--alpha_fm_loss", type=float, default=0.02)
    parser.add_argument("--alpha_diversity_loss", type=float, default=0.01)
    parser.add_argument("--k_samples", type=int, default=4)

    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducibility")
    parser.add_argument("--start_episode", type=int, default=0)
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    args = parser.parse_args()

    return args
