import numpy as np

class ReplayBuffer:
    def __init__(self, args, state_dim, action_dim):
        self.size = args.buffer_size
        self.batch_size = args.batch_size
        self.ptr = 0
        self.current_size = 0

        self.states = np.zeros((self.size, state_dim))
        self.actions = np.zeros((self.size, action_dim))
        self.rewards = np.zeros((self.size, args.n_agents))
        self.next_states = np.zeros((self.size, state_dim))
        self.dones = np.zeros((self.size, 1))

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.choice(self.current_size, batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.current_size