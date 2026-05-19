import torch
import torch.nn.functional as F
from marl.maddpg.actor_critic import Critic
from policies.flow_model import FlowActor
import numpy as np
import random

class MADDPG:
    def __init__(self, args, state_dim, action_dim, device):
        """
        初始化critic，对每一个agent初始化cfm actor，设置好优化器，学习率通过传参传入
        :param args: 参数列表
        :param state_dim: 观测空间维度
        :param action_dim: 动作空间维度
        :param device: 运行设备，一般是cuda
        """
        self.args = args
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_step = 0

        # ===== 流匹配网络初始化，每一个agent一个cfm actor =====
        self.n_agents = len(args.obs_shape)
        self.actors = []
        self.target_actors = []
        self.actor_optims = []
        for i in range(self.n_agents): # 对每一个agent都初始化
            agent_name = self.args.agent_ids[i]
            actor = FlowActor(args.obs_shape[i], args.action_shape[i], device=device).to(device) # 训练网络
            target_actor = FlowActor(args.obs_shape[i], args.action_shape[i], device=device).to(device) # 目标网络
            target_actor.load_state_dict(actor.state_dict())
            if "adversary" in agent_name:
                optim = torch.optim.Adam(actor.parameters(), lr=args.lr_flow * 1.4)
            else:
                optim = torch.optim.Adam(actor.parameters(), lr=args.lr_flow)

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.actor_optims.append(optim)

        # ==== critic网络初始化 ====
        self.critic_adv = Critic(args).to(device)
        self.critic_good = Critic(args).to(device)

        self.target_critic_adv = Critic(args).to(device)
        self.target_critic_good = Critic(args).to(device)

        self.target_critic_adv.load_state_dict(self.critic_adv.state_dict())
        self.target_critic_good.load_state_dict(self.critic_good.state_dict())

        self.critic_optim_adv = torch.optim.Adam(self.critic_adv.parameters(), lr=args.lr_critic)
        self.critic_optim_good = torch.optim.Adam(self.critic_good.parameters(), lr=args.lr_critic)

    # =========================================================
    # 选择动作
    # =========================================================
    def select_action(self, obs_list, explore=True, temperature=0.5):
        with torch.no_grad():
            state_list = [torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1) for obs in obs_list]

            # Step 1: 对每个 agent 采样 num_samples 个候选动作
            all_candidates = []
            for i in range(self.n_agents):
                candidates_i = self.actors[i].sample_action(state_list[i], num_samples=self.args.k_samples)
                # candidates_i shape: (1, K, action_dim_i)
                all_candidates.append(candidates_i)  # 列表，每个元素 (1, K, dim_i)

            if self.args.k_samples == 1:
                # 单采样时直接合并
                final_action = torch.cat([cand[:, 0, :] for cand in all_candidates], dim=1)
                q_selected = torch.tensor(0.0, device=self.device)
                q_all = torch.tensor([0.0], device=self.device)
                action_diversity = 0.0
            else:
                # Step 2: 构造 K 个候选联合动作
                K = self.args.k_samples
                joint_candidates = []  # 列表，每个元素为 (1, total_action_dim)
                for k in range(K):
                    joint_k = torch.cat([cand[:, k, :] for cand in all_candidates], dim=1)
                    joint_candidates.append(joint_k)
                joint_candidates = torch.stack(joint_candidates, dim=1)  # (1, K, total_dim)

                # Step 3: 使用 Good Critic 对每个联合动作评分（这里使用 good critic 的 mean Q）
                q_values = []
                for k in range(K):
                    joint_k = joint_candidates[:, k, :]  # (1, total_dim)
                    state_list_for_q = state_list  # 直接使用全局观测
                    action_list_for_q = self._split_action(joint_k)
                    q = self.critic_good(state_list_for_q, action_list_for_q)  # (1, n_agents)
                    # 取所有 good agent 的 Q 均值作为团队得分
                    good_indices = [i for i, name in enumerate(self.args.agent_ids) if "agent" in name]
                    q_team = q[:, good_indices].mean(dim=1)  # (1,)
                    q_values.append(q_team)
                q_values = torch.cat(q_values, dim=0)  # (K,)
                action_diversity = joint_candidates.std(dim=1).mean().item()

                # Step 4: Softmax 采样（或 argmax）
                if explore and self.args.k_samples > 1:
                    # 使用温度参数控制 softmax 或直接 argmax
                    probs = torch.softmax(q_values / temperature, dim=0).cpu().numpy()
                    selected_idx = np.random.choice(K, p=probs)  # Softmax 采样
                    q_selected = q_values[selected_idx]
                    q_all = q_values
                else:
                    selected_idx = q_values.argmax(dim=0).item()  # 直接选最优
                    q_selected = q_values[selected_idx]
                    q_all = q_values
                final_action = joint_candidates[:, selected_idx, :]  # (1, total_dim)

            # 添加噪声（只在训练阶段使用）
            if explore and self.args.k_samples == 1:  # 对单采样加噪声
                noise = torch.randn_like(final_action) * self.args.noise_rate
                final_action = torch.clamp(final_action + noise, 0, 1)
            else:
                final_action = torch.clamp(final_action, 0, 1)
        return final_action, q_selected, q_all, action_diversity

# =========================================================
    # 训练
    # =========================================================
    def train(self, replay_buffer, total_step, train_adv=True, train_good=True, update_critic=True):
        self.total_step = total_step
        if len(replay_buffer) < self.args.batch_size: # warmup
            return None
        train_info = {}
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.args.batch_size) # 采样1个batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device) # batch个
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device) # in a corporate scene
        dones = torch.FloatTensor(dones).to(self.device) # 转换为tensor格式准备计算

        state_list = self._split_state(states)
        action_list = self._split_action(actions)

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1) # 调整维度以适配计算图

        # =====================================================
        # 1 Critic 更新
        # =====================================================
        with torch.no_grad():
            next_state_list = self._split_state(next_states)
            next_actions = []

            for i in range(self.n_agents):
                next_a = self.target_actors[i].sample_action(next_state_list[i])[:,0].detach() # 采样下一个动作
                noise = torch.randn_like(next_a) * 0.05
                next_a = torch.clamp(next_a + noise, 0, 1) # 探索噪声
                next_actions.append(next_a)
            next_actions = torch.cat(next_actions, dim=1) # 拼接成联合动作

            # =====================================================
            # ⭐ 分阵营 critic 更新
            # =====================================================

        current_q_adv = self.critic_adv(state_list, action_list)
        current_q_good = self.critic_good(state_list, action_list)
        with torch.no_grad():
            q_next_adv = self.target_critic_adv(next_state_list, self._split_action(next_actions))
            q_next_good = self.target_critic_good(next_state_list, self._split_action(next_actions))

            q_next_adv = torch.clamp(q_next_adv, -50, 50)
            q_next_good = torch.clamp(q_next_good, -50, 50)

            # ⭐ mask
            adv_mask = torch.tensor([1 if "adversary" in name else 0 for name in self.args.agent_ids], device=self.device).view(1, -1)
            good_mask = 1 - adv_mask

            adv_indices = adv_mask.squeeze(0).bool()
            good_indices = good_mask.squeeze(0).bool()

            target_q_adv = rewards[:, adv_indices] + self.args.gamma * (1 - dones) * q_next_adv[:, adv_indices]
            target_q_good = rewards[:, good_indices] + self.args.gamma * (1 - dones) * q_next_good[:, good_indices]

        loss_adv = F.mse_loss(current_q_adv[:, adv_indices], target_q_adv)
        loss_good = F.mse_loss(current_q_good[:, good_indices], target_q_good)

        if update_critic:
            self.critic_optim_adv.zero_grad()
            loss_adv.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_adv.parameters(), 0.5)  # 梯度裁剪 防止训练过快
            self.critic_optim_adv.step()

            self.critic_optim_good.zero_grad()
            loss_good.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_good.parameters(), 0.5)  # 梯度裁剪 防止训练过快
            self.critic_optim_good.step()

            self._soft_update(self.critic_adv, self.target_critic_adv)
            self._soft_update(self.critic_good, self.target_critic_good)
        # =====================================================
        # 2 Flow Actor 更新（无 AWR 版本）
        # =====================================================
        state_list = self._split_state(states)
        action_list = self._split_action(actions)

        adv_policy_losses = []
        good_policy_losses = []
        adv_fm_losses = []
        good_fm_losses = []
        adv_actions = []
        good_actions = []
        for i in range(self.n_agents):
            agent_name = self.args.agent_ids[i]
            # ====== ⭐ 关键：冻结 good agents ======
            # adversary!!!!!!!!!!!!!!!!!!!!!!!!!
            if "adversary" in agent_name and not train_adv:
                continue
            if "agent" in agent_name and not train_good:
                continue

            # ===== Flow Matching Loss（仍然用 buffer 行为）=====
            flow_loss_i = self.actors[i].compute_flow_loss(state_list[i], action_list[i])
            fm_loss = flow_loss_i.mean()

            if "adversary" in agent_name:
                adv_fm_losses.append(fm_loss.item())
            else:
                good_fm_losses.append(fm_loss.item())
            # ===== Policy Gradient Loss =====
            a_actor = self.actors[i].sample_action(state_list[i])[:, 0]
            new_actions = []

            for j in range(self.n_agents):
                if j == i:
                    a_j = a_actor
                else:
                    a_j = action_list[j]  # 全部用 buffer
                new_actions.append(a_j)

            if "adversary" in agent_name:
                agent_idx = i
                q_actor = self.critic_adv(state_list, new_actions)
                q_i = q_actor[:, agent_idx].unsqueeze(1)
                # baseline_adv = q_i.mean(dim=0, keepdim=True)
                # adv_adv = q_i - baseline_adv
                # policy_loss = -adv_adv.mean()
                policy_loss = -q_i.mean()
                adv_policy_losses.append(policy_loss.item())
                adv_actions.append(a_actor.detach())
            else:
                q_actor = self.critic_good(state_list, new_actions)
                # 只取 good agents 的 Q
                good_indices = [idx for idx, name in enumerate(self.args.agent_ids) if "agent" in name]
                team_q = q_actor[:, good_indices].mean(dim=1, keepdim=True)
                policy_loss = -team_q.mean()
                good_policy_losses.append(policy_loss.item())
                good_actions.append(a_actor.detach())

            # ===== 总 Loss =====
            alpha_fm_loss = max(0.003, 0.01 * (1 - (total_step - 30000) / (60000 * 25 * 0.8)))
            alpha_fm_loss = 0.01

            loss = (alpha_fm_loss * fm_loss + self.args.alpha_policy_loss * policy_loss)

            self.actor_optims[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optims[i].step()

            self._soft_update(self.actors[i], self.target_actors[i])

        # ===== Policy Loss =====
        train_info["adv_policy_loss"] = np.mean(adv_policy_losses) if len(adv_policy_losses) > 0 else 0
        train_info["good_policy_loss"] = np.mean(good_policy_losses) if len(good_policy_losses) > 0 else 0

        # ===== Action Stats =====
        if len(adv_actions) > 0:
            adv_actions_tensor = torch.cat(adv_actions, dim=0)
            train_info["adv_action_std"] = adv_actions_tensor.std().item()
        else:
            train_info["adv_action_std"] = 0

        if len(good_actions) > 0:
            good_actions_tensor = torch.cat(good_actions, dim=0)
            train_info["good_action_std"] = good_actions_tensor.std().item()
        else:
            train_info["good_action_std"] = 0

        # ===== FM Loss 日志 =====
        train_info["good_fm_loss"] = np.mean(good_fm_losses) if len(good_fm_losses) > 0 else 0
        train_info["adv_fm_loss"] = np.mean(adv_fm_losses) if len(adv_fm_losses) > 0 else 0

        return train_info

    # =========================================================
    # Soft Update
    # =========================================================
    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # =========================================================
    # 状态拆分
    # =========================================================
    def _split_state(self, joint_state):
        split = torch.split(joint_state, self.args.obs_shape, dim=1)
        return list(split)

    # =========================================================
    # 动作拆分
    # =========================================================
    # def _split_action(self, joint_action):
    #     split = torch.chunk(joint_action, self.n_agents, dim=1)
    #     return list(split)
    def _split_action(self, joint_action):
        split = torch.split(joint_action, self.args.action_shape, dim=1)
        return list(split)

    def save(self, path):
        save_dict = {
            "good_critic": self.critic_good.state_dict(),
            "good_target_critic": self.target_critic_good.state_dict(),
            "adv_critic": self.critic_adv.state_dict(),
            "adv_target_critic": self.target_critic_adv.state_dict(),
            "actors": [actor.state_dict() for actor in self.actors],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "good_critic_optim": self.critic_optim_good.state_dict(),
            "adv_critic_optim": self.critic_optim_adv.state_dict(),
            "actor_optims": [opt.state_dict() for opt in self.actor_optims]
        }
        torch.save(save_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.critic_good.load_state_dict(checkpoint["good_critic"])
        self.target_critic_good.load_state_dict(checkpoint["good_target_critic"])
        self.critic_adv.load_state_dict(checkpoint["adv_critic"])
        self.target_critic_adv.load_state_dict(checkpoint["adv_target_critic"])

        for actor, sd in zip(self.actors, checkpoint["actors"]):
            actor.load_state_dict(sd)

        for actor, sd in zip(self.target_actors, checkpoint["target_actors"]):
            actor.load_state_dict(sd)

        self.critic_optim_good.load_state_dict(checkpoint["good_critic_optim"])
        self.critic_optim_adv.load_state_dict(checkpoint["adv_critic_optim"])

        for opt, sd in zip(self.actor_optims, checkpoint["actor_optims"]):
            opt.load_state_dict(sd)
