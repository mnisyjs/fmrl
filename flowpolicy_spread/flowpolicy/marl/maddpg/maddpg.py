import torch
import torch.nn.functional as F
from marl.maddpg.actor_critic import Critic
from policies.flow_model import FlowActor
import numpy as np

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
            actor = FlowActor(args.obs_shape[i], args.action_shape[i], device=device).to(device) # 训练网络
            target_actor = FlowActor(args.obs_shape[i], args.action_shape[i], device=device).to(device) # 目标网络
            target_actor.load_state_dict(actor.state_dict())
            optim = torch.optim.Adam(actor.parameters(), lr=args.lr_flow)

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.actor_optims.append(optim)

        # ==== critic网络初始化 ====
        self.critic = Critic(args).to(device)
        self.target_critic = Critic(args).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

    def get_dist_to_nearest_landmark(self, obs_batch):
        """
        obs_batch: 形状为 (batch_size, 18) 的张量
        返回: 形状为 (batch_size, 1) 的张量，表示到最近地标的距离
        """
        # 提取 3 个地标的相对位置
        # landmark 1: obs[:, 4:6]
        # landmark 2: obs[:, 6:8]
        # landmark 3: obs[:, 8:10]

        lm1_dist = torch.norm(obs_batch[:, 4:6], p=2, dim=1, keepdim=True)
        lm2_dist = torch.norm(obs_batch[:, 6:8], p=2, dim=1, keepdim=True)
        lm3_dist = torch.norm(obs_batch[:, 8:10], p=2, dim=1, keepdim=True)

        # 在 3 个距离里取最小值
        min_dist, _ = torch.min(torch.cat([lm1_dist, lm2_dist, lm3_dist], dim=1), dim=1, keepdim=True)
        return min_dist

    # =========================================================
    # 选择动作
    # =========================================================
    def select_action(self, obs_list, explore=True):
        """
        输入obs，输出一个联合动作
        :param obs_list:观测列表
        :return:
        """
        state_list = []
        for i in range(self.n_agents):
            s = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            state_list.append(s) # 将状态也依次拼接，添加batch维度

            # ==== 重点 ====
            # baseline actions
        base_actions = [] # 构建基准动作，这样才能评估某个智能体的新动作相对原本动作的提升
        for j in range(self.n_agents):
            a = self.actors[j].sample_action(state_list[j], num_samples=self.args.k_samples) # 每个agent采样num个动作
            base_actions.append(a[:, 0, :]) # 得到一个联合动作样本

        best_actions = []
        for i in range(self.n_agents):
            # 每个agent都有k个候选动作
            candidates = self.actors[i].sample_action(state_list[i], num_samples=self.args.k_samples)  # (1, K, action_dim)
            # 构造 joint actions batch
            joint_actions = []
            for k in range(self.args.k_samples):
                action_list = base_actions.copy()
                action_list[i] = candidates[:, k]
                joint_actions.append(torch.cat(action_list, dim=1)) # 固定其他智能体的动作，只改变智能体i的动作

            joint_actions = torch.cat(joint_actions, dim=0)
            # state expand
            joint_state = torch.cat(state_list, dim=1)
            joint_state = joint_state.repeat(self.args.k_samples, 1) # 状态也拼接，便于打分

            # critic batch evaluation
            q_values = self.critic(self._split_state(joint_state), self._split_action(joint_actions))
            q_values = q_values[:, i] # 维度
            q_values = q_values.view(-1, self.args.k_samples)
            # best_idx = torch.argmax(q_values, dim=1)
            # best_action = candidates[torch.arange(candidates.shape[0]), best_idx, :] # critic打分，哪一个分高就说明哪一个候选动作最好

            temperature_q = 1.0  # 可以调 0.5~2.0
            q_mean = q_values.mean(dim=1, keepdim=True)
            q_std = q_values.std(dim=1, keepdim=True) + 1e-6
            q_norm = (q_values - q_mean) / q_std # 标准化作用：防止“一个极端Q值”统治选择（你现在最大的问题之一） 2026.4.11晚上11：52 进行完训练12之后给出的综合性意见1
            probs = torch.softmax(q_norm / temperature_q, dim=1)

            epsilon = max(0.05, 0.1 * (1 - self.total_step / (62500 * 25)))
            if np.random.rand() < epsilon: # 探索机制加入：防止 critic 一直带偏（现在就是这个问题） 2026.4.11晚上11：52 进行完训练12之后给出的综合性意见2
                best_idx = torch.randint(0, self.args.k_samples, (1,))
            else:
                best_idx = torch.multinomial(probs, num_samples=1).squeeze()
            # 从概率分布采样
            # best_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # best_action = candidates[torch.arange(candidates.shape[0]), best_idx, :] 调整维度问题
            assert candidates.shape[0] == 1, f"Batch size not 1: {candidates.shape}" # 维度

            best_action = candidates[0, best_idx.item(), :].unsqueeze(0)
            best_actions.append(best_action)

        final_action = torch.cat(best_actions, dim=1) # 这里是联合动作，便于采样
        if explore:
            noise = torch.randn_like(final_action) * self.args.noise_rate # 添加噪声，实施探索
            final_action = torch.clamp(final_action + noise, 0, 1) # 为什么需要剪切而不用映射？剪切会丢失一部分数据吧
        else:
            final_action = torch.clamp(final_action, 0, 1)

        assert final_action.shape == (1, sum(self.args.action_shape)), f"Action shape wrong: {final_action.shape}"
        # print("final_action shape:", final_action.shape)

        return final_action

    # =========================================================
    # 训练
    # =========================================================
    def train(self, replay_buffer, total_step):
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
        dist_list = []
        for i in range(self.n_agents):
            obs_i = state_list[i]  # (batch, obs_dim)
            dist_i = self.get_dist_to_nearest_landmark(obs_i)
            dist_list.append(dist_i)

        # if rewards.dim() == 2:
            # rewards = rewards.sum(dim=1, keepdim=True) * 0.5 # 如果有多个奖励，取总和 调整奖励：Q scale 被放大 → critic更不稳定 2026.4.11晚上11：52 进行完训练12之后给出的综合性意见3（自己发现）
            # 但是mean可提供的奖励信息又太少了，不能给critic提供训练信息，因此换成sum，但是乘以一个倍数
            # rewards = rewards - lambda_dist * dist_tensor  # ⭐关键
            # rewards = rewards.sum(dim=1, keepdim=True) * 0.5 # 加入距离限制
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

            # q_next = self.target_critic(self._split_state(next_states), self._split_action(next_actions)).view(-1, 1)
            q_next = self.target_critic(self._split_state(next_states), self._split_action(next_actions))
            q_next = torch.clamp(q_next, -300, 300) # 防止数值爆炸
            # rewards = rewards.mean(dim=1, keepdim=True) + 0.1 * rewards.std(dim=1, keepdim=True) # 防止维度出错，但是很重要的点：mean是否会导致q值过于平均！
            # 删掉了mean，开始把critic的输出分给每个agent
            target_q = rewards + self.args.gamma * (1 - dones) * q_next # 目标Q值 让critic更容易学习
            # target_q = rewards_i + self.args.gamma * (1 - dones) * q_next
            # TODO: check q_next
            # todo: seems to be ok

        current_q = self.critic(self._split_state(states), self._split_action(actions)) # 实际Q值
        # print(f"{len(current_q)} | {len(target_q)} | {rewards.size()} | {dones.size()} | {len(q_next)} | {type(q_next)}")
        critic_loss = F.mse_loss(current_q, target_q) # 平滑L1，比MSE更鲁棒（？）
        # 平滑S1太保守了，换成mse

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # 梯度裁剪 防止训练过快
        self.critic_optim.step() # 经典优化器三件套

        self._soft_update(self.critic, self.target_critic) # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        # 你居然漏了target_critic更新？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        # =====================================================
        # 2 Flow Actor 更新（无 AWR 版本）
        # =====================================================
        state_list = self._split_state(states)
        action_list = self._split_action(actions)

        train_info.update({
            "critic_loss": critic_loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
            "q_min": current_q.min().item(),
            "target_q_mean": target_q.mean().item(),
            "target_q_std": target_q.std().item(),
            "batch_reward_mean": rewards.mean().item()
        })

        for i in range(self.n_agents):
            # ===== Flow Matching Loss（仍然用 buffer 行为）=====
            flow_loss_i = self.actors[i].compute_flow_loss(state_list[i], action_list[i])
            fm_loss = flow_loss_i.mean()

            # ===== Policy Gradient Loss =====
            a_actor = self.actors[i].sample_action(state_list[i])[:, 0]

            new_actions = action_list.copy()
            new_actions[i] = a_actor

            q_actor = self.critic(state_list, new_actions)

            # maximize Q → minimize -Q
            temperature_adv = 1.0 # 降低 advantage 温度，减少随机性，增强利用 2026.4.12下午18：24 进行完训练14之后给出的综合性意见2
            temperature_adv = max(0.3, 1.0 - total_step / (10000 * 25))
            # 前期让模型不贪婪选择高分动作，后期直接利用
            # 2026.4.12下午18：29开始进行第15次训练
            with torch.no_grad():
                baseline = q_actor.mean()
            adv = q_actor - baseline
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            weight = torch.tanh(adv / temperature_adv) # 又加了一个系数）# 让 advantage “变温和” 2026.4.12下午16:57 进行完训练13.75之后给出的综合性意见1
            # 18.3 暂时撤掉adv，对于argmax来说有点危险了，万一得到一个虚假的超高q，很容易学偏

            # policy_loss = -(weight.detach() * q_actor).mean() # 引入类awr机制：让 critic “能排序” 2026.4.12早9：48 进行完训练13之后给出的综合性意见1
            # policy_loss = -q_actor.mean() # 调整pg loss：critic 一点误差 → actor 被放大10倍带偏2026.4.11晚上11：52 进行完训练12之后给出的综合性意见5
            # 换成每个agent有自己的q
            # policy_loss = - (torch.softmax(q_actor / temp, dim=0) * q_actor).mean() # temp = 0.1
            q_i = q_actor[:, i].unsqueeze(1)
            policy_loss = -q_i.mean()
            # 2026.4.12凌晨0:30开始进行第13次训练
            # 2026.4.12早上9:57开始进行第13.5次训练
            # policy_loss += 0.01 * ((a_actor - action_list[i]) ** 2).mean() # 稳定训练（加强fm）
            # 要加回来了？训练14 要把action std拉回去 不加！ 这个作用是让action往buffer靠，稳定训练，但是不适合训练14时的情况
            # 加回来了，适合训练17 轻量行为约束 用它替代一部分 FM 的作用，但是0.01倍
            # 我去这个有点猛，和fm一起会把动作锁的很死，导致直接冲不出buffer了 还是得删掉 感觉这个基本上不好用
            action_std = a_actor.std()
            alpha_diversity_loss = max(0, 0.005 * (1 - total_step / (40000 * 25)))
            # # 动态调整diversity权重，前期探索，后期利用 2026.4.13早10：48 进行完训练15之后给出的综合性意见2
            diversity_loss = -alpha_diversity_loss * torch.log(action_std + 1e-6) # 降低 diversity，现在太强了，要稍微听critic说一下话 2026.4.12下午18：24 进行完训练14之后给出的综合性意见1
            # 奖励初步上升，开始动态控制方差 18.2
            # target_std = 0.25
            # diversity_loss = (action_std - target_std).pow(2)
            # 控制方差 训练16.7意见

            # ===== 总 Loss =====
            # alpha = self.args.alpha / (1 - total_step / 2500000)
            # loss = self.args.alpha * policy_loss
            alpha_fm_loss = max(0.0005, 0.002 * (1 - total_step / (60000 * 25)))
            # 动态调整FM权重，前期探索，后期利用 2026.4.13早10：48 进行完训练15之后给出的综合性意见1
            # 调参ing 2026.4.13下午16：44 进行完训练16之后给出的综合性意见
            # 假设总训练轮数为 20000
            action_penalty = min(0.003, 0.005 * (self.total_step / 37500))
            loss = alpha_fm_loss * fm_loss + self.args.alpha_policy_loss * policy_loss + diversity_loss + action_penalty * torch.norm(a_actor, p=2, dim=-1).mean() # 再次加入fm：这是你现在最缺的稳定器 2026.4.11晚上11：52 进行完训练12之后给出的综合性意见4
            # 对fm loss权重调整：现在 critic 不可靠，必须靠 FM 保住 policy 不乱飞 2026.4.12早9：48 进行完训练13之后给出的综合性意见2
            # 对总loss权重调整：加入action std，不要只让 Actor 最大化 Q 值，要惩罚它“动作太单一” 2026.4.12下午16:57 进行完训练13.75之后给出的综合性意见2
            # 2026.4.12 下午5：08开始进行第14次训练
            # 加入动作惩罚，让模型后期逐渐趋于稳定

            if i == 0:
                train_info.update({
                    "q_actor": q_actor.mean().item(),
                    "fm_loss": fm_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "action_mean": a_actor.mean().item(),
                    "action_std": a_actor.std().item(),
                    "adv_mean":adv.mean().item(),
                    "adv_std":adv.std().item()
                })

            # ===== debug（建议保留）=====
            if total_step % 2000 == 0 and i == 0:
                print(f"Q_actor: {q_actor.mean().item():.3f} | Current_q: {current_q.mean().item():.3f}")
                print(f"max: {current_q.max().item():.3f} | min: {current_q.min().item():.3f}")
                print(f"FM loss: {fm_loss.item():.4f} | Policy loss: {policy_loss.item():.4f} | Critic loss: {critic_loss.item():.4f}")
                print(f"Action Mean: {a_actor.mean().item():.4f}, Action Std: {a_actor.std().item():.4f}")
                print(f"Rewards Mean: {rewards.mean():.4f}, rewards Std: {rewards.std():.4f}")
                print(f"target_q Mean: {target_q.mean():.4f}, target_q Std: {target_q.std():.4f}")
                print(f"adv mean: {adv.mean():.4f}, std: {adv.std():.4f}")

            self.actor_optims[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optims[i].step()

            self._soft_update(self.actors[i], self.target_actors[i])
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
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actors": [actor.state_dict() for actor in self.actors],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "critic_optim": self.critic_optim.state_dict(),
            "actor_optims": [opt.state_dict() for opt in self.actor_optims]
        }
        torch.save(save_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])

        for actor, sd in zip(self.actors, checkpoint["actors"]):
            actor.load_state_dict(sd)

        for actor, sd in zip(self.target_actors, checkpoint["target_actors"]):
            actor.load_state_dict(sd)

        self.critic_optim.load_state_dict(checkpoint["critic_optim"])

        for opt, sd in zip(self.actor_optims, checkpoint["actor_optims"]):
            opt.load_state_dict(sd)
