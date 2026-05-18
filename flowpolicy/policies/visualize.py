import torch
import numpy as np
import time
import os
import pygame
# 导入你自己的模块
from marl.common.arguments import get_args
from envs.env_mpe import make_env  # 确保路径正确
from marl.maddpg.maddpg import MADDPG

def visualize():
    # 1. 获取基础参数
    args = get_args()

    # 2. 调用你写的 make_env 来初始化环境并补充 args (重要！)
    # 注意：为了看到画面，这里强制覆盖 render_mode
    env, args = make_env(args)

    from pettingzoo.mpe import simple_spread_v3
    env = simple_spread_v3.parallel_env(
        N=3,
        local_ratio=0.5,
        max_cycles=args.max_episode_len,
        continuous_actions=True,
        dynamic_rescaling = True,
        render_mode="human"
    )

    # 3. 初始化模型
    state_dim = sum(args.obs_shape)
    action_dim = sum(args.action_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maddpg = MADDPG(args, state_dim, action_dim, device)

    # 4. 加载权重 (请确保路径指向你保存的 .pt 文件)
    model_path = "C:/Users/mnisyjs/PycharmProjects/flowpolicy/marl/models/final_model.pt"
    if os.path.exists(model_path):
        maddpg.load(model_path)
        print(f"成功加载模型: {model_path}")
    else:
        print(f"警告: 未找到模型文件 {model_path}，将使用随机策略进行演示。")

    # 5. 开始运行
    for episode in range(10):
        obs, infos = env.reset()
        episode_reward = 0
        maddpg.args.noise_rate = 0.0

        for step in range(args.max_episode_len):
            # 将 obs 转换为 tensor list 给 maddpg
            # 注意：env.agents 的顺序要和 obs_shape 对应
            state_list = [obs[agent] for agent in env.agents]

            with torch.no_grad():
                # 使用 select_action 采样动作
                # 因为训练时你用了 epsilon-greedy，推理时建议把 epsilon 设为 0
                actions_tensor = maddpg.select_action(state_list, explore=False)

                # 将合在一起的 actions 拆分回 dict 格式给环境
                action_dict = {}
                combined_actions = actions_tensor.cpu().numpy().flatten()

                cursor = 0
                for i, agent in enumerate(env.agents):
                    a_dim = args.action_shape[i]
                    action_dict[agent] = combined_actions[cursor: cursor + a_dim]
                    cursor += a_dim

            # 与环境交互
            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            # 渲染画面
            env.render()
            pygame.event.pump()
            # time.sleep(0.01)  # 稍微减慢速度，方便观察细节

            episode_reward += sum(rewards.values())

            if any(terminations.values()) or any(truncations.values()):
                break

        print(f"Episode {episode} | Total Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    visualize()