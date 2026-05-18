import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from common.arguments import get_args
from common.utils import make_env
from agent import Agent


def plot_training_history(save_path, smooth=True):
    """
    加载并绘制训练历史 (returns.pkl.npy)
    """
    history_file = os.path.join(save_path, 'returns.pkl.npy')
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return

    try:
        returns = np.load(history_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading history: {e}")
        return

    if len(returns) == 0:
        print("History is empty.")
        return

    plt.style.use('bmh') # 使用美观的样式
    plt.figure(figsize=(12, 7))
    
    # 原始曲线
    plt.plot(returns, alpha=0.3, color='#1f77b4', label='Raw Reward')
    print(returns)
    print(returns.shape)
    
    # 平滑曲线
    if smooth and len(returns) > 10:
        window_size = max(1, len(returns) // 20)
        smoothed_returns = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(returns)), smoothed_returns, color='#ff7f0e', linewidth=2, label='Smoothed Reward')

    plt.xlabel('Evaluation Point', fontsize=12)
    plt.ylabel('Average Returns', fontsize=12)
    plt.title('MADDPG Training Progress - simple_spread', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存绘图
    plot_save_path = os.path.join(save_path, 'training_reward_curve.png')
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    print(f"Training reward curve saved to {plot_save_path}")
    
    # 显示奖励曲线 (会阻塞，直到关闭窗口)
    print("Showing reward curve window. Close it to proceed to agent visualization...")
    plt.show()


def plot_eval_rewards(rewards_list, save_path):
    """
    绘制当前可视化运行的奖励
    """
    if not rewards_list:
        return
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards_list) + 1), rewards_list, marker='o', color='#2ca02c', linestyle='-', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Visualization Run - Episode Rewards', fontsize=14, fontweight='bold')
    plt.xticks(range(1, len(rewards_list) + 1))
    plt.grid(True, linestyle=':', alpha=0.8)
    
    plot_save_path = os.path.join(save_path, 'eval_reward_curve.png')
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    print(f"Evaluation reward curve saved to {plot_save_path}")


def load_model(agents, model_path):
    """
    为每个agent加载最新的权重文件
    """
    for i, agent in enumerate(agents):
        agent_path = os.path.join(model_path, f'agent_{i}')
        if not os.path.exists(agent_path):
            print(f"Agent {i} model path not found: {agent_path}")
            continue
        
        # 查找最新的权重文件 (可能是 actor_params.pkl 或者 数字_actor_params.pkl)
        files = [f for f in os.listdir(agent_path) if f.endswith('actor_params.pkl')]
        if not files:
            print(f"No actor model found for agent {i} in {agent_path}")
            continue
        
        # 如果有多个文件，取最大的那个 (假设文件名格式为 'num_actor_params.pkl' 或 'actor_params.pkl')
        def get_num(filename):
            if '_' in filename:
                return int(filename.split('_')[0])
            return -1 # 'actor_params.pkl' 优先级最低或作为默认
            
        latest_file = max(files, key=get_num)
        actor_path = os.path.join(agent_path, latest_file)
        
        # 加载权重
        try:
            agent.policy.actor_network.load_state_dict(torch.load(actor_path, map_location='cpu'))
            print(f"Agent {i} loaded model from {latest_file}")
        except Exception as e:
            print(f"Failed to load model for agent {i}: {e}")


def visualize_spread(args, env, agents):
    """
    实现 agent spread 环境的移动过程可视化
    """
    print("\nStarting visualization...")
    for episode in range(5):  # 运行5个回合进行展示
        print(f"Episode {episode + 1} starting...")
        obs_dict, _ = env.reset()
        # 对齐智能体
        active_agents = list(env.agents)[:args.n_agents]
        s = [obs_dict[agent] for agent in active_agents]
        
        total_reward = 0
        for step in range(args.evaluate_episode_len):
            try:
                env.render()
            except Exception as e:
                if step == 0:
                    print(f"Render failed: {e}")
            
            actions = []
            with torch.no_grad():
                for i in range(len(s)):
                    # 使用 epsilon=0, noise=0 进行纯策略选择
                    action = agents[i].select_action(s[i], 0, 0)
                    actions.append(action)
            
            # 执行动作
            action_dict = {agent: actions[i] for i, agent in enumerate(active_agents)}
            obs_next_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            
            s_next = [obs_next_dict[agent] for agent in active_agents]
            rewards = [reward_dict.get(agent, 0.0) for agent in active_agents]
            done = [term_dict.get(agent, False) or trunc_dict.get(agent, False) for agent in active_agents]
            
            total_reward += np.sum(rewards)
            s = s_next
            
            if all(done):
                break
        
        print(f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f}")


if __name__ == '__main__':
    # 获取参数
    args = get_args()
    
    # 路径设置
    model_root = os.path.join(args.save_dir, args.scenario_name)
    
    # 1. 绘图训练过程
    print("--- Plotting Training History ---")
    plot_training_history(model_root)
    
    # 2. 初始化环境和智能体进行可视化
    print("\n--- Visualizing Agent Movement ---")
    
    # 先调用 make_env 获取基础 args (包含 obs_shape, action_shape, high_action 等)
    env, args = make_env(args)
    
    # 尝试重新创建带 render_mode='human' 的环境进行展示
    try:
        from pettingzoo.mpe import simple_spread_v3
        # 保持与 make_env 一致的参数，仅修改 render_mode
        env = simple_spread_v3.parallel_env(
            N=3, 
            local_ratio=0.5, 
            max_cycles=args.evaluate_episode_len, 
            continuous_actions=True, 
            render_mode='human'
        )
        env.reset()
    except Exception as e:
        print(f"Could not create human-render env: {e}. Using default env.")

    # 创建智能体
    agents = [Agent(i, args) for i in range(args.n_agents)]
    
    # 加载权重
    load_model(agents, model_root)
    
    # 运行可视化
    visualize_spread(args, env, agents)
    
    env.close()
