import torch
import torch.nn.functional as F


class JointActionSampler:
    def __init__(self, flow_actor, critic, K=5, beta=1.0):
        self.flow_actor = flow_actor
        self.critic = critic
        self.K = K
        self.beta = beta

    def select_action(self, state_tensor):
        """
        state_tensor: (1, state_dim)
        return: joint_action_tensor (1, action_dim)
        """

        candidates = []
        scores = []

        with torch.no_grad():
            for _ in range(self.K):
                action = self.flow_actor.sample_action(state_tensor)
                state_list = split_state(state_tensor)
                action_list = split_action(action)
                q_value = self.critic(state_list, action_list)

                candidates.append(action)
                scores.append(q_value.squeeze())

            scores = torch.stack(scores)
            probs = F.softmax(self.beta * scores, dim=0)

            idx = torch.multinomial(probs, 1).item()
            chosen_action = candidates[idx]

        return chosen_action