import torch
import torch.nn.functional as F
from maddpg.actor_critic import Critic
from maddpg.actor_critic import Actor   # 从 spread baseline 导入的确定性 Actor
import numpy as np

class MADDPG_Baseline:
    def __init__(self, args, state_dim, action_dim, device):
        self.args = args
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_step = 0

        self.n_agents = len(args.obs_shape)
        self.actors = []
        self.target_actors = []
        self.actor_optims = []
        for i in range(self.n_agents):
            agent_name = self.args.agent_ids[i]
            actor = Actor(args, i).to(device)           # 确定性 Actor
            target_actor = Actor(args, i).to(device)
            target_actor.load_state_dict(actor.state_dict())
            if "adversary" in agent_name:
                optim = torch.optim.Adam(actor.parameters(), lr=args.lr_actor * 1.4)
            else:
                optim = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.actor_optims.append(optim)

        # 分阵营 Critic，与主模型完全相同
        self.critic_adv = Critic(args).to(device)
        self.critic_good = Critic(args).to(device)
        self.target_critic_adv = Critic(args).to(device)
        self.target_critic_good = Critic(args).to(device)
        self.target_critic_adv.load_state_dict(self.critic_adv.state_dict())
        self.target_critic_good.load_state_dict(self.critic_good.state_dict())
        self.critic_optim_adv = torch.optim.Adam(self.critic_adv.parameters(), lr=args.lr_critic)
        self.critic_optim_good = torch.optim.Adam(self.critic_good.parameters(), lr=args.lr_critic)

    def select_action(self, obs_list, explore=True, noise_rate=None):
        """确定性动作输出 + 可选噪声（无 multi‑sample）"""
        if noise_rate is None:
            noise_rate = self.args.noise_rate
        with torch.no_grad():
            state_list = [torch.tensor(o, dtype=torch.float32, device=self.device).view(1, -1) for o in obs_list]
            actions = []
            for i in range(self.n_agents):
                a = self.actors[i](state_list[i])
                if explore:
                    noise = torch.randn_like(a) * noise_rate
                    a = torch.clamp(a + noise, 0, 1)
                else:
                    a = torch.clamp(a, 0, 1)
                actions.append(a)
            final_action = torch.cat(actions, dim=1)
        return final_action

    def train(self, replay_buffer, total_step, train_adv=True, train_good=True, update_critic=True):
        # ---------- Critic update 部分与主模型完全相同（抄 original maddpg.py）----------
        self.total_step = total_step
        if len(replay_buffer) < self.args.batch_size:
            return None
        train_info = {}
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.args.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)

        state_list = self._split_state(states)
        action_list = self._split_action(actions)

        with torch.no_grad():
            next_state_list = self._split_state(next_states)
            next_actions = []
            for i in range(self.n_agents):
                next_a = self.target_actors[i](next_state_list[i]).detach()
                noise = torch.randn_like(next_a) * 0.05
                next_a = torch.clamp(next_a + noise, 0, 1)
                next_actions.append(next_a)
            next_actions = torch.cat(next_actions, dim=1)

        current_q_adv = self.critic_adv(state_list, action_list)
        current_q_good = self.critic_good(state_list, action_list)
        with torch.no_grad():
            q_next_adv = self.target_critic_adv(next_state_list, self._split_action(next_actions))
            q_next_good = self.target_critic_good(next_state_list, self._split_action(next_actions))
            q_next_adv = torch.clamp(q_next_adv, -50, 50)
            q_next_good = torch.clamp(q_next_good, -50, 50)
            adv_mask = torch.tensor(
                [1 if "adversary" in name else 0 for name in self.args.agent_ids],
                device=self.device
            ).view(1, -1)
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
            torch.nn.utils.clip_grad_norm_(self.critic_adv.parameters(), 0.5)
            self.critic_optim_adv.step()
            self.critic_optim_good.zero_grad()
            loss_good.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_good.parameters(), 0.5)
            self.critic_optim_good.step()
            self._soft_update(self.critic_adv, self.target_critic_adv)
            self._soft_update(self.critic_good, self.target_critic_good)

        # ---------- Actor update（去掉 FM，只保留 Policy Loss）----------
        state_list = self._split_state(states)
        action_list = self._split_action(actions)
        adv_policy_losses, good_policy_losses = [], []
        adv_actions, good_actions = [], []
        for i in range(self.n_agents):
            agent_name = self.args.agent_ids[i]
            if "adversary" in agent_name and not train_adv:
                continue
            if "agent" in agent_name and not train_good:
                continue

            a_actor = self.actors[i](state_list[i])          # 确定性输出
            new_actions = []
            for j in range(self.n_agents):
                if j == i:
                    a_j = a_actor
                else:
                    a_j = action_list[j]
                new_actions.append(a_j)

            if "adversary" in agent_name:
                q_actor = self.critic_adv(state_list, new_actions)
                q_i = q_actor[:, i].unsqueeze(1)
                policy_loss = -q_i.mean()
                adv_policy_losses.append(policy_loss.item())
                adv_actions.append(a_actor.detach())
            else:
                q_actor = self.critic_good(state_list, new_actions)
                good_indices = [idx for idx, name in enumerate(self.args.agent_ids) if "agent" in name]
                team_q = q_actor[:, good_indices].mean(dim=1, keepdim=True)
                policy_loss = -team_q.mean()
                good_policy_losses.append(policy_loss.item())
                good_actions.append(a_actor.detach())

            loss = self.args.alpha_policy_loss * policy_loss   # 无 fm_loss
            self.actor_optims[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optims[i].step()
            self._soft_update(self.actors[i], self.target_actors[i])

        train_info["adv_policy_loss"] = np.mean(adv_policy_losses) if adv_policy_losses else 0
        train_info["good_policy_loss"] = np.mean(good_policy_losses) if good_policy_losses else 0
        if adv_actions:
            adv_actions_tensor = torch.cat(adv_actions, dim=0)
            train_info["adv_action_std"] = adv_actions_tensor.std().item()
        else:
            train_info["adv_action_std"] = 0
        if good_actions:
            good_actions_tensor = torch.cat(good_actions, dim=0)
            train_info["good_action_std"] = good_actions_tensor.std().item()
        else:
            train_info["good_action_std"] = 0
        return train_info

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def _split_state(self, joint_state):
        split = torch.split(joint_state, self.args.obs_shape, dim=1)
        return list(split)

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