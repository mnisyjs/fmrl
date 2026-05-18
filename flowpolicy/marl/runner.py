import matplotlib
matplotlib.use("Agg")
from common.replay_buffer import ReplayBuffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from maddpg.maddpg import MADDPG
import csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episodes
        self.env = env
        self.seed = args.seed  # 记录当前种子
        state_dim = sum(args.obs_shape)
        action_dim = sum(args.action_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.maddpg = MADDPG(args, state_dim, action_dim, device=self.device)
        self.buffer = ReplayBuffer(args, state_dim, action_dim)
        self.reward_history = []
        self.total_step = 0
        self.save_dir = "models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.true_reward_history = []

        log_name = f"training_log_seed_{self.seed}.csv"
        self.log_file = open(log_name, "w", newline="")
        self.logger = csv.writer(self.log_file)

        header = [
            "episode","seed", "reward", "true_reward",
            "q_actor", "q_mean", "q_max", "q_min",
            "fm_loss", "policy_loss", "critic_loss", "action_mean",
            "action_std", "target_q_mean", "adv_mean", "adv_std"
        ]
        self.logger.writerow(header)
        self.last_train_info = {k: 0.0 for k in header if k != "episode" and k != "reward"}

    def get_dist_to_nearest_landmark(self, obs):
        lm1 = obs[4:6]
        lm2 = obs[6:8]
        lm3 = obs[8:10]

        d1 = np.linalg.norm(lm1)
        d2 = np.linalg.norm(lm2)
        d3 = np.linalg.norm(lm3)

        return min(d1, d2, d3)

    def run(self):
        for episode in range(self.args.max_episodes): # episode
            self._adjust_learning_rate(episode)

            if episode == 0:
                obs, infos = self.env.reset(seed=self.seed)
            else:
                obs, infos = self.env.reset()

            episode_reward = 0
            true_episode_reward = 0 # 记录真实奖励

            for step in range(self.args.max_episode_len): # step
                # parallel_env 返回 dict
                state_list = [obs[agent] for agent in self.env.agents] # 获取agent的观测存入列表
                joint_state = np.concatenate(state_list) # 拼接

                action_tensor = self.maddpg.select_action(state_list) # 动作分布
                action = action_tensor.squeeze(0).cpu().detach().numpy() # 一堆动作的集合体
                action = np.clip(action, 0, 1)

                # split 成 dict
                actions = {}
                idx = 0
                for agent, dim in zip(self.env.agents, self.args.action_shape):
                    actions[agent] = action[idx:idx + dim] # 把一堆动作拆成一个个动作
                    idx += dim

                agent_ids = self.env.agents.copy()
                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                done = all(terms.values()) or all(truncs.values())

                if not done:
                    next_state_list = [next_obs[agent] for agent in agent_ids]
                    joint_next_state = np.concatenate(next_state_list)
                else:
                    joint_next_state = np.zeros_like(joint_state)

                # reward = np.array([rewards[a] for a in agent_ids]) # 每个agent一个奖励
                reward = [] # 每个agent有自己的奖励
                for agent in agent_ids:
                    obs_i = obs[agent]
                    dist = self.get_dist_to_nearest_landmark(obs_i)

                    r_global = rewards[agent]  # 原始global
                    r_local = -dist
                    # 混合（关键） 并且给上距离近的奖励
                    cover_threshold = 0.1
                    r_bonus = 1.0 if dist < cover_threshold else 0.0
                    r_i = 0.7 * r_global + 0.3 * r_local + r_bonus

                    reward.append(r_i)

                reward = np.array(reward)
                # 原始 reward 可能太温和了
                # shaped_reward = np.where(reward > -25, reward * 1.0, reward * 3.0)  # 缩小惩罚

                self.buffer.store(joint_state,action,reward,joint_next_state,done)

                if self.total_step % 3 == 0: # 每3步更新一次
                    info = self.maddpg.train(self.buffer, self.total_step)
                    # print("reward:", reward) # 检查全局奖励是否被分解
                    if info is not None:
                        self.last_train_info = info

                obs = next_obs
                episode_reward += reward.sum()
                true_episode_reward += sum([rewards[a] for a in agent_ids])

                self.total_step += 1
                if done:
                    break

            if episode % 1000 == 0 and episode > 0:
                path = os.path.join(self.save_dir, f"model_ep{episode}.pt")
                self.maddpg.save(path)
            if episode == self.args.max_episodes - 1:
                self.maddpg.save(os.path.join(self.save_dir, "final_model.pt")) # 保存模型

            raw_row = [
                episode,
                self.seed,
                float(np.mean(episode_reward)),
                float(np.mean(true_episode_reward)),
                self.last_train_info.get("q_actor", 0),
                self.last_train_info.get("q_mean", 0),
                self.last_train_info.get("q_max", 0),
                self.last_train_info.get("q_min", 0),
                self.last_train_info.get("fm_loss", 0),
                self.last_train_info.get("policy_loss", 0),
                self.last_train_info.get("critic_loss", 0),
                self.last_train_info.get("action_mean", 0),
                self.last_train_info.get("action_std", 0),
                self.last_train_info.get("target_q_mean", 0),
                self.last_train_info.get("adv_mean", 0),
                self.last_train_info.get("adv_std", 0)
            ]
            row = [raw_row[0],  # episode 保持原样
                *[round(float(x), 4) for x in raw_row[1:]]  # 其余全部 round
            ]
            self.logger.writerow(row)
            self.log_file.flush()  # 实验日志

            self.reward_history.append(episode_reward)
            self.true_reward_history.append(true_episode_reward)
            if episode % 10 == 0 and episode > 0:
                print(f"Episode {episode} | Reward {episode_reward:.3f}")
                print(f"Episode {episode} | True Reward {true_episode_reward:.3f}")

        self._plot_reward()

    def _plot_reward(self):
        rewards = np.array(self.reward_history)
        true_rewards = np.array(self.true_reward_history)
        # 设置学术风格
        plt.rcParams['font.family'] = 'serif'
        plt.style.use('seaborn-v0_8-paper')  # 或者 'ggplot'

        plt.figure(figsize=(10, 6), dpi=300)  # 高清输出

        window = 50
        if len(rewards) >= window:
            # 计算平滑曲线
            shaped_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            true_avg = np.convolve(true_rewards, np.ones(window) / window, mode='valid')

            # 绘制原始数据（浅色细线）
            plt.plot(rewards[:len(shaped_avg)], color='blue', alpha=0.1, linewidth=0.5)
            plt.plot(true_rewards[:len(true_avg)], color='orange', alpha=0.1, linewidth=0.5)

            # 绘制平滑后的主曲线（深色粗线）
            plt.plot(shaped_avg, label='Shaped Reward (Smooth)', color='blue', linewidth=1.8)
            plt.plot(true_avg, label='True Reward (Smooth)', color='orange', linewidth=1.8)

        # 细节优化
        plt.title(f'Training Performance (Seed: {self.seed})', fontsize=14, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Cumulative Reward', fontsize=12)

        # 添加精细网格线
        plt.grid(True, linestyle='--', alpha=0.6)

        # 移除上方和右方的边框（显得简洁）
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.legend(frameon=True, loc='lower right', fontsize=10)

        # 🚀 同样，图片命名也要带上种子
        plt.savefig(f'training_curve_seed_{self.seed}.png', bbox_inches='tight')
        plt.close()

    def _adjust_learning_rate(self, episode):
        """里程碑式学习率衰减"""
        # 设定衰减节点和因子
        if episode == 20000 or episode == 35000:
            decay_factor = 0.5
        else:
            return

        # 更新 Actor 学习率 (针对每个 agent 的 optimizer)
        for opt in self.maddpg.actor_optims:
            for param_group in opt.param_groups:
                param_group['lr'] *= decay_factor

        # 更新 Critic 学习率
        for param_group in self.maddpg.critic_optim.param_groups:
            param_group['lr'] *= decay_factor

        print(f"\n>>> Episode {episode}: Learning rate decayed by {decay_factor}. "
              f"New Actor LR: {self.maddpg.actor_optims[0].param_groups[0]['lr']}")


