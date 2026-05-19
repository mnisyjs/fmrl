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
        self.env = env
        state_dim = sum(args.obs_shape)
        action_dim = sum(args.action_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # =====================================================
        # Seed 固定
        # =====================================================
        self.seed = args.seed
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # 保证 cudnn 可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        args.agent_ids = self.env.agents.copy()
        self.maddpg = MADDPG(args, state_dim, action_dim, device=self.device)
        # model_path = "models/adversary_model_ep30000.pt"
        model_path = getattr(args, "model_dir", "")
        if model_path != "":
            self.maddpg.load(model_path)
            print(f"Loaded model from {model_path}")
        # python main.py - -seed 0 - -model - dir models / xxx.pt 直接运行
        self.buffer = ReplayBuffer(args, state_dim, action_dim)
        self.warmup_episodes = 500
        self.total_step = self.args.start_episode * self.args.max_episode_len
        self.fixed_lure = None
        self.fixed_occupy = None
        self.prev_adv_pos = None
        self.episode_role_steps = 0
        self.episode_deception_steps = 0
        self._true_landmark = None
        self._fake_landmark = None
        self.save_dir = "models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.good_reward_history = []
        self.adv_reward_history = []
        self.total_reward_history = []
        self.reward_window = []
        self.critic_gain_history = []
        self.action_diversity_history = []
        self.reward_debug = {
            "team_env": 0.0,
            "r_lure": 0.0,
            "r_occupy": 0.0,
            "r_escape": 0.0,
            "r_idle": 0.0,
            "r_avoid": 0.0,
            "r_separate": 0.0,
            "r_participate": 0.0,
            "r_lure_occupy": 0.0,
            "final_reward": 0.0,
            "count": 0
        }
        self.log_file = open(f"training_log_adversary_seed{self.seed}.csv", "w", newline="")
        # self.log_file = open("training_log_adversary.csv", "w", newline="")
        self.logger = csv.writer(self.log_file)
        header = [
            "episode",
            "good_reward",
            "adv_reward",
            "total_reward",
            "good_policy_loss",
            "adv_policy_loss",
            "good_action_std",
            "adv_action_std",
            "good_fm_loss",
            "adv_fm_loss",
            "role_consistency",  # 新增
            "deception_rate",  # 新增
            "r_lure",
            "r_occupy",
            "final_reward",
            "critic_gain",
            "action_diversity"
        ]
        self.logger.writerow(header)
        self.last_train_info = {k: 0.0 for k in header if k not in ["episode", "reward"]}

    def get_reward(self, agent_ids, rewards_env, next_obs, pos_dict):
        rewards = []
        good_ids = [a for a in agent_ids if "agent" in a]
        adv_ids = [a for a in agent_ids if "adversary" in a]
        final_rewards = {}

        # 🔴 Adversary 不变
        for agent_id in adv_ids:
            final_rewards[agent_id] = rewards_env[agent_id]

        if len(good_ids) == 2 and len(adv_ids) > 0:
            g0, g1 = good_ids[0], good_ids[1]

            # ---- 从 observation 反推真地标绝对位置 ----
            goal_rel_0 = np.array(next_obs[g0][:2], dtype=np.float32)
            goal_rel_1 = np.array(next_obs[g1][:2], dtype=np.float32)
            p0_arr = np.array(pos_dict[g0], dtype=np.float32)
            p1_arr = np.array(pos_dict[g1], dtype=np.float32)
            true_landmark = (p0_arr + goal_rel_0 + p1_arr + goal_rel_1) / 2.0

            # ---- 假地标 ----
            lm1_rel_0 = np.array(next_obs[g0][2:4], dtype=np.float32)
            lm2_rel_0 = np.array(next_obs[g0][4:6], dtype=np.float32)
            lm1_abs = p0_arr + lm1_rel_0
            lm2_abs = p0_arr + lm2_rel_0
            if np.linalg.norm(lm1_abs - true_landmark) < np.linalg.norm(lm2_abs - true_landmark):
                fake_landmark = lm2_abs
            else:
                fake_landmark = lm1_abs

            self._true_landmark = true_landmark.copy()
            self._fake_landmark = fake_landmark.copy()
            adv_pos = pos_dict[adv_ids[0]]
            team_env = (rewards_env[g0] + rewards_env[g1]) / 2.0

            # 固定角色
            lure_id = self.fixed_lure
            occupy_id = self.fixed_occupy

            for agent_id in good_ids:
                pos_i = pos_dict[agent_id]
                d_opp = np.linalg.norm(pos_i - adv_pos)
                d_true = np.linalg.norm(pos_i - true_landmark)
                d_fake = np.linalg.norm(pos_i - fake_landmark)

                # ---- 初始化角色奖励 ----
                r_lure, r_lure_occupy, r_occupy, r_escape, r_separate = 0.0, 0.0, 0.0, 0.0, 0.0

                if agent_id == lure_id:
                    # 诱敌者
                    d_adv_true = np.linalg.norm(adv_pos - true_landmark)
                    proximity_fake = np.exp(-2.0 * d_fake)
                    r_lure = 0.35 * np.tanh(d_adv_true) + 0.15 * proximity_fake  # prox 系数 0.15
                    r_lure_occupy = 1.8 * np.exp(-5.0 * d_fake)  # 假地标驻守奖励
                    r_separate = -0.2 * np.exp(-3.0 * d_true)  # 远离真地标
                elif agent_id == occupy_id:
                    # 占点者
                    # occupancy_base = 1.0 * np.exp(-5.0 * d_true)  # 衰减 5.0
                    # safe_factor = np.clip((d_opp - 0.2) / 0.6, 0.25, 1.0)  # 下限 0.25
                    # r_occupy = occupancy_base * safe_factor
                    r_occupy = 3.0 * np.exp(-6.0 * d_true)
                    if d_opp < 0.5:
                        r_escape = 0.35 * (0.5 - d_opp)

                # ---- 自奖励参与（个体 decoupled） ----
                if agent_id == lure_id:
                    d_task = d_fake
                else:
                    d_task = d_true
                r_participate_self = 0.12 * np.exp(-3.0 * d_task)

                # ---- 角色条件 idle ----
                if agent_id == lure_id:
                    idle_score = min(d_fake, d_opp)
                else:
                    idle_score = d_true
                r_idle = -0.25 if idle_score > 1.0 else 0.0

                # ---- 躲避 ----
                r_avoid = -0.08 * np.exp(-2.0 * d_opp)

                # ---- 组合 ----
                base_reward = (
                        0.25 * team_env
                        + r_lure
                        + r_lure_occupy
                        + r_occupy
                        + r_escape
                        + r_separate
                        + r_participate_self
                        + r_idle
                        + r_avoid
                )
                final_rewards[agent_id] = base_reward

                # Debug 累加（保持字段一致）
                self.reward_debug["team_env"] += team_env
                self.reward_debug["r_lure"] += r_lure
                self.reward_debug["r_occupy"] += r_occupy
                self.reward_debug["r_escape"] += r_escape
                self.reward_debug["r_idle"] += r_idle
                self.reward_debug["r_avoid"] += r_avoid
                self.reward_debug["r_separate"] += r_separate
                self.reward_debug["r_participate"] += r_participate_self  # 个体值累加
                self.reward_debug["r_lure_occupy"] += r_lure_occupy
                self.reward_debug["final_reward"] += base_reward
                self.reward_debug["count"] += 1
        # 构建返回数组
        for agent_id in agent_ids:
            rewards.append(final_rewards[agent_id])
        return np.array(rewards, dtype=np.float32)

    def run(self):
        for episode in range(self.args.start_episode, self.args.start_episode + self.args.max_episodes): # episode
            num_steps = 0
            self.episode_role_steps = 0
            self.episode_deception_steps = 0
            obs, infos = self.env.reset(seed=self.seed + episode)
            initial_pos = {name: agent.state.p_pos.copy() for name, agent in
                           zip(self.env.agents, self.env.unwrapped.world.agents)}
            adv_name = [a for a in self.env.agents if "adversary" in a][0]
            good_names = [a for a in self.env.agents if "agent" in a]
            g0, g1 = good_names
            d0 = np.linalg.norm(initial_pos[g0] - initial_pos[adv_name])
            d1 = np.linalg.norm(initial_pos[g1] - initial_pos[adv_name])
            self.fixed_lure = g0 if d0 < d1 else g1
            self.fixed_occupy = g1 if self.fixed_lure == g0 else g0
            episode_good_reward = 0
            episode_adv_reward = 0
            episode_total_reward = 0

            for step in range(self.args.max_episode_len): # step
                # parallel_env 返回 dict
                obs_list = [obs[agent] for agent in self.env.agents] # 获取agent的观测存入列表
                joint_state = np.concatenate(obs_list) # 拼接

                action_tensor, q_selected, q_all, action_entropy = self.maddpg.select_action(obs_list)
                critic_gain = ((q_selected - q_all.mean()) / (q_all.std() + 1e-6)).item()
                self.critic_gain_history.append(critic_gain)
                self.action_diversity_history.append(action_entropy)

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
                world = self.env.unwrapped.world
                pos_dict = {agent.name: agent.state.p_pos.copy() for agent in world.agents}
                adv_ids = [a for a in agent_ids if "adversary" in a]

                if not done:
                    next_state_list = [next_obs[agent] for agent in agent_ids]
                    joint_next_state = np.concatenate(next_state_list)
                else:
                    joint_next_state = np.zeros_like(joint_state)

                reward_array = self.get_reward(agent_ids, rewards, next_obs, pos_dict)

                good_reward = sum([reward_array[i] for i, a in enumerate(agent_ids) if "agent" in a])
                adv_reward = sum([reward_array[i] for i, a in enumerate(agent_ids) if "adversary" in a])
                total_reward = np.mean(reward_array)

                if self._true_landmark is not None and self._fake_landmark is not None:
                    # Role consistency: lure agent 靠近假地标且 occupy agent 靠近真地标
                    pos_lure = pos_dict[self.fixed_lure]
                    pos_occ = pos_dict[self.fixed_occupy]
                    d_fake_lure = np.linalg.norm(pos_lure - self._fake_landmark)
                    d_true_occ = np.linalg.norm(pos_occ - self._true_landmark)
                    if d_fake_lure < 0.4 and d_true_occ < 0.4:
                        self.episode_role_steps += 1

                    # Deception rate: adversary 靠近假地标
                    d_adv_fake = np.linalg.norm(pos_dict[adv_ids[0]] - self._fake_landmark)
                    if d_adv_fake < 0.4:
                        self.episode_deception_steps += 1

                self.buffer.store(joint_state, action, reward_array, joint_next_state, done)

                if episode > self.warmup_episodes:
                    if self.total_step % 5 == 0:
                        # 1) 更新 Critic 两次（冻结 Actor）
                        for _ in range(2):
                            self.maddpg.train(self.buffer, self.total_step,
                                              train_adv=False, train_good=False, update_critic=True)
                        # 2) 更新所有 Actor（不更新 Critic）
                        info = self.maddpg.train(self.buffer, self.total_step,
                                                 train_adv=True, train_good=True, update_critic=False)
                        if info is not None:
                            self.last_train_info = info

                obs = next_obs
                episode_good_reward += good_reward
                episode_adv_reward += adv_reward
                episode_total_reward += total_reward

                num_steps += 1
                self.total_step += 1
                if done:
                    break

            if episode % 2000 == 0 and episode > 0:
                # path = os.path.join(self.save_dir, f"adversary_model_ep{episode}.pt")
                path = os.path.join(self.save_dir, f"seed{self.seed}_adversary_model_ep{episode}.pt")
                self.maddpg.save(path)
            if episode == self.args.max_episodes - 1:
                # self.maddpg.save(os.path.join(self.save_dir, "final_adversary_model.pt")) # 保存模型
                self.maddpg.save(os.path.join(self.save_dir, f"seed{self.seed}_final_adversary_model.pt"))

            role_consistency = self.episode_role_steps / max(1, num_steps)
            deception_rate = self.episode_deception_steps / max(1, num_steps)
            mean_gain = np.mean(self.critic_gain_history)
            mean_entropy = np.mean(self.action_diversity_history)

            avg_r_lure = 0
            avg_r_occupy = 0
            avg_final_reward = 0

            if episode % 50 == 0 and episode > 0:
                c = max(1, self.reward_debug["count"])
                avg_r_lure = (self.reward_debug["r_lure"] / c)
                avg_r_occupy = (self.reward_debug["r_occupy"] / c)
                avg_final_reward = (self.reward_debug["final_reward"] / c)
                print(
                    f"\n[Reward Debug @ Episode {episode}] "
                    f"r_lure={avg_r_lure:.3f} | "
                    f"r_occupy={avg_r_occupy:.3f} | "
                    f"final_reward={avg_final_reward:.3f}"
                )

            if episode % 20 == 0 and episode > 0:
                raw_row = [
                    episode,
                    float(episode_good_reward),
                    float(episode_adv_reward),
                    float(episode_total_reward),

                    self.last_train_info.get("good_policy_loss", 0),
                    self.last_train_info.get("adv_policy_loss", 0),

                    self.last_train_info.get("good_action_std", 0),
                    self.last_train_info.get("adv_action_std", 0),

                    self.last_train_info.get("good_fm_loss", 0),
                    self.last_train_info.get("adv_fm_loss", 0),
                    role_consistency,  # 新增
                    deception_rate,  # 新增

                    avg_r_lure,
                    avg_r_occupy,
                    avg_final_reward,

                    mean_gain,
                    mean_entropy
                ]
                row = [raw_row[0], *[round(float(x), 3) for x in raw_row[1:]]]
                self.logger.writerow(row)
                self.log_file.flush()  # 实验日志

            self.good_reward_history.append(episode_good_reward)
            self.adv_reward_history.append(episode_adv_reward)
            self.total_reward_history.append(episode_total_reward)

            if episode % 10 == 0 and episode > 0:
                print(
                    f"Episode {episode} | "
                    f"Good Reward {episode_good_reward:.3f} | "
                    f"Adv Reward {episode_adv_reward:.3f} | "
                    f"Total Reward {episode_total_reward:.3f}"
                )

                print(
                    f"Role={role_consistency:.3f} | "
                    f"Deception={deception_rate:.3f}"
                )

        self.plot_rewards()

    def plot_rewards(self):
        import matplotlib.pyplot as plt
        import numpy as np

        window = 200
        good_rewards = np.array(self.good_reward_history)
        adv_rewards = np.array(self.adv_reward_history)
        total_rewards = np.array(self.total_reward_history)
        # ===== 移动平均 =====
        if len(good_rewards) >= window:
            good_ma = np.convolve(good_rewards, np.ones(window) / window, mode='valid')
            adv_ma = np.convolve(adv_rewards, np.ones(window) / window, mode='valid')
            total_ma = np.convolve(total_rewards, np.ones(window) / window, mode='valid')
        else:
            good_ma = good_rewards
            adv_ma = adv_rewards
            total_ma = total_rewards

        # =====================================================
        # Figure 1: Good vs Adv
        # =====================================================
        plt.figure(figsize=(10, 6))
        plt.plot(good_ma, label="Good Agents Reward", linewidth=2)
        plt.plot(-adv_ma, label="Adversary Suppression", linewidth=2)
        # 中线
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Good vs Adversary Reward")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'good_vs_adv_reward_seed{self.seed}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # =====================================================
        # Figure 2: Total Reward
        # =====================================================
        plt.figure(figsize=(10, 6))
        plt.plot(total_ma, label="Total Reward", linewidth=2)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Total Reward Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'total_reward_curve_seed{self.seed}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Reward curves saved.")
