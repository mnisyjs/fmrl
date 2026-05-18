import torch
import numpy as np
import time
import os
import pygame
# 导入你的模块
from marl.common.arguments import get_args
from envs.env_mpe import make_env
from marl.maddpg.maddpg import MADDPG


def visualize_adversary():
    # 1. 获取参数
    args = get_args()
    env, args = make_env(args)
    # 2. 初始化环境 (确保是 simple_adversary)
    # 如果你的 make_env 支持传参，确保它初始化的是 adversary 场景
    # 这里我们直接用 PettingZoo 的原生调用方式以防万一
    from pettingzoo.mpe import simple_adversary_v3
    env = simple_adversary_v3.parallel_env(
        N=2,  # 2个好人 + 1个对手 = 3个agent
        max_cycles=args.max_episode_len,
        continuous_actions=True,
        render_mode="human"  # 弹出窗口的关键
    )

    # 3. 补充 args 里的维度信息（MADDPG 初始化需要）
    env.reset()
    args.obs_shape = [env.observation_space(agent).shape[0] for agent in env.agents]
    args.action_shape = [env.action_space(agent).shape[0] for agent in env.agents]

    # 4. 初始化模型
    state_dim = sum(args.obs_shape)
    action_dim = sum(args.action_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.agent_ids = [agent for agent in env.agents]
    maddpg = MADDPG(args, state_dim, action_dim, device)

    # 5. 【关键】加载你训练好的权重
    # 请修改下面这个路径，指向你保存的 .pt 文件
    model_path = "../marl/models/seed0_final_adversary_model.pt"
    if os.path.exists(model_path):
        print(f"正在加载模型: {model_path}")
        maddpg.load(model_path)
    else:
        print(f"警告：未找到模型文件 {model_path}，将使用随机策略演示！")

    # 6. 开始循环播放
    for episode in range(20):  # 跑10个回合看看
        obs_dict, infos = env.reset()
        episode_reward = 0

        # print(f"--- 第 {episode + 1} 个回合开始 ---")

        for step in range(args.max_episode_len):
            # 1. 准备包含所有人观测的列表 (直接放在循环外，只算一次)
            state_list = [obs_dict[agent] for agent in env.agents]

            with torch.no_grad():
                # 2. 一次性获取拼接动作 (输出的是 15 维的 tensor)
                actions_tensor = maddpg.select_action(state_list, explore=False, num_samples=4, temperature=0.5)

                # 3. 压平成 1D numpy 数组，准备切片分发
                combined_actions = actions_tensor.cpu().numpy().flatten()

                action_dict = {}
                cursor = 0

                # 4. 按每个 Agent 的真实动作维度 (a_dim) 划蛋糕
                for i, agent in enumerate(env.agents):
                    a_dim = args.action_shape[i]  # 在 adversary 里通常是 5
                    action_dict[agent] = combined_actions[cursor: cursor + a_dim]
                    cursor += a_dim

            # 5. 与环境交互
            obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)
            episode_reward += sum(rewards.values())

            # 画面渲染
            env.render()
            pygame.event.pump()
            # time.sleep(0.01)  # 稍微减慢速度，方便观察细节

            if any(terminations.values()) or any(truncations.values()):
                break

        print(f"Episode {episode} | Total Reward: {episode_reward:.2f}")

    env.close()
    print("可视化结束。")


if __name__ == "__main__":
    visualize_adversary()