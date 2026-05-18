import torch
import torch.nn as nn
import torch.optim as optim
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import numpy as np

class FlowActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, euler_steps=4, device="cuda"):
        super(FlowActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.euler_steps = euler_steps
        self.device = device

        # ===== Velocity Network =====
        # 获取速度向量场的网络模型
        # input: xt + t + state
        # output: velocity (same dim as action)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # ===== Flow Matcher (tool class) =====
        self.flow_matcher = ConditionalFlowMatcher()
        self.to(device)

    # =========================================================
    # Velocity field forward
    # =========================================================
    def forward(self, xt, t, state):
        """
        前向过程
        xt: (batch, action_dim)
        t:  (batch)
        state: (batch, state_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([xt, t, state], dim=-1)
        v = self.net(inp)
        return v

    # =========================================================
    # Compute Flow Matching loss (supervised stage)
    # =========================================================
    def compute_flow_loss(self, state, action):
        """
        state:  (batch, state_dim)
        action: (batch, action_dim)
        """
        batch_size = action.shape[0]
        # x0: gaussian noise
        x0 = torch.randn_like(action) # action from buffer, the "action" holds the size of action, making dimx0 = action dim?
        # x1: target action
        x1 = action
        # sample intermediate location and target velocity
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1) # 生成t,xt,ut
        t = torch.clamp(t, 0.01, 0.99)

        t = t.to(self.device)
        xt = xt.to(self.device)
        ut = ut.to(self.device)
        state = state.to(self.device)

        v_prediction = self.forward(xt, t, state) # 生成条件向量场
        # MSE loss
        loss = ((v_prediction - ut) ** 2).mean(dim=1) # 计算流匹配损失
        return loss

    # =========================================================
    # Euler integration sampling
    # =========================================================
    def sample_action(self, state, num_samples=1, euler_steps=4): # 用euler采样生成动作
        batch_size = state.shape[0]
        state = state.unsqueeze(1).repeat(1, num_samples, 1)
        state = state.reshape(batch_size * num_samples, -1)
        x = torch.randn(batch_size * num_samples, self.action_dim).to(self.device) * 0.6
        x.requires_grad_(True)
        dt = 1.0 / euler_steps

        for i in range(euler_steps):
            t = torch.ones(batch_size * num_samples, 1).to(self.device) * (i / euler_steps)
            v = self.forward(x, t, state)
            x = x + v * dt
        action = torch.sigmoid(x)
        action = action.view(batch_size, num_samples, self.action_dim)
        if torch.isnan(action).any():
            action = torch.rand_like(action)
        return action

    # =========================================================
    # Environment interface
    # =========================================================
    def select_action(self, state_np, epsilon=0.1, noise_rate=0.1):
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)

            # ===== epsilon-greedy =====
            if np.random.rand() < epsilon:
                action = torch.randn(1, self.action_dim).to(self.device)  # [0,1]
            else:
                action = self.sample_action(state)

                # ===== add gaussian noise =====
                noise = noise_rate * torch.randn_like(action)
                action = action + noise
                action = torch.clamp(action, 0.0, 1.0)

        self.train()
        return action.squeeze(0).cpu().numpy()

