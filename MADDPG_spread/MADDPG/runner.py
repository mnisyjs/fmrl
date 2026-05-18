from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ 必须在 import pyplot 之前，切换为纯文件渲染后端
import matplotlib.pyplot as plt
import csv



class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episodes
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = 'MADDPG/' + self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.episode_count = 0
        self.log_data = []


    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents


    def _log_metrics(self, episode, step, agent_rewards, metrics_buffer):
        total_rew = np.sum(agent_rewards)
        avg_rew = np.mean(agent_rewards)
        max_rew = np.max(agent_rewards)
        min_rew = np.min(agent_rewards)

        m_log = {}
        for k, v in metrics_buffer.items():
            if len(v) > 0:
                if k in ['action_mean', 'action_std']:
                    m_log[k] = np.mean([np.mean(x) for x in v])
                else:
                    m_log[k] = np.mean(v)
            else:
                m_log[k] = 0
        
        # Order: episode, total_reward, q_mean, q_max, q_min, critic_loss, policy_loss, action_mean, action_std, timestep, avg_reward, best_reward, worst_reward, advantage
        row = [episode, total_rew, m_log['q_mean'], m_log['q_max'], m_log['q_min'],
               m_log['critic_loss'], m_log['actor_loss'], m_log['action_mean'], m_log['action_std'],
               step, avg_rew, max_rew, min_rew, m_log['advantage']]
        
        self.log_data.append(row)
        print(f"\nEpisode {episode}, Timestep {step}, Total Reward: {total_rew:.2f}, Critic Loss: {m_log['critic_loss']:.4f}")

    def _write_log(self):
        if not self.log_data:
            return
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.log_data)
            f.flush()
        self.log_data = []

    def save_model(self, label=None):
        for agent in self.agents:
            agent.policy.save_model(label)


    def run(self):
        # Logging setup
        self.log_file = os.path.join(self.save_path, 'train_log.csv')
        self.csv_header = ['episode', 'total_reward', 'q_mean', 'q_max', 'q_min', 'critic_loss', 'policy_loss', 
                           'action_mean', 'action_std', 'timestep', 'avg_reward', 'best_reward', 'worst_reward', 'advantage']

        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)

        returns = []
        current_agent_rewards = np.zeros(self.args.n_agents)
        steps_in_episode = 0
        
        # Training metrics aggregation
        metrics_buffer = {k: [] for k in ['critic_loss', 'actor_loss', 'q_mean', 'q_max', 'q_min', 'advantage', 'action_mean', 'action_std']}
        
        for time_step in tqdm(range(self.args.time_steps)):

            # === LR decay ===
            if time_step == 500000:
                for agent in self.agents:
                    for p in agent.policy.actor_optim.param_groups:
                        p['lr'] *= 0.5
                    for p in agent.policy.critic_optim.param_groups:
                        p['lr'] *= 0.5

            if time_step == 875000:
                for agent in self.agents:
                    for p in agent.policy.actor_optim.param_groups:
                        p['lr'] *= 0.5
                    for p in agent.policy.critic_optim.param_groups:
                        p['lr'] *= 0.5

            # reset the environment
            if time_step % self.episode_limit == 0:
                if time_step > 0 and steps_in_episode > 0:
                    self._log_metrics(self.episode_count, time_step, current_agent_rewards, metrics_buffer)
                    self.save_model() # Save latest every episode
                    self.episode_count += 1
                    
                    if self.episode_count % 100 == 0:
                        self._write_log()
                        self.save_model(self.episode_count) # Save numbered checkpoint every 100 episodes
                    
                    for k in metrics_buffer: metrics_buffer[k] = []
                
                steps_in_episode = 0
                current_agent_rewards = np.zeros(self.args.n_agents)
                obs_dict, _ = self.env.reset()
                s = [obs_dict[agent] for agent in list(self.env.agents)[:self.args.n_agents]]
            
            u = []
            actions = []
            with torch.no_grad():
                for agent_id in range(len(s)):
                    self.epsilon = max(0.05, 0.1 * (1 - time_step / (62500 * 25)))
                    action = self.agents[agent_id].select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)

            active_agents = list(self.env.agents)[:len(actions)]
            action_dict = {agent: actions[i] for i, agent in enumerate(active_agents)}
            obs_next, reward, term, trunc, info = self.env.step(action_dict)

            s_next = [obs_next[agent] for agent in active_agents]
            done = [(term.get(agent, False) or trunc.get(agent, False)) for agent in active_agents]
            
            if all(done):
                # Episode ended early
                if steps_in_episode > 0:
                    self._log_metrics(self.episode_count, time_step + 1, current_agent_rewards, metrics_buffer)
                    self.save_model() # Save latest every episode
                    self.episode_count += 1
                    
                    if self.episode_count % 100 == 0:
                        self._write_log()
                        self.save_model(self.episode_count) # Save numbered checkpoint every 100 episodes
                        
                    for k in metrics_buffer: metrics_buffer[k] = []

                steps_in_episode = 0
                current_agent_rewards = np.zeros(self.args.n_agents)
                obs_dict, _ = self.env.reset()
                s = [obs_dict[agent] for agent in list(self.env.agents)[:self.args.n_agents]]
                continue



            r = [reward.get(agent, 0.0) for agent in active_agents]
            current_agent_rewards += np.array(r)
            steps_in_episode += 1

            assert len(u) == self.args.n_agents, f"u len {len(u)}"
            assert len(r) == self.args.n_agents, f"r len {len(r)}"
            assert len(s_next) == self.args.n_agents
            assert len(done) == self.args.n_agents
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], done[:self.args.n_agents])
            s = s_next
            
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    info = agent.learn(transitions, other_agents)
                    # Collect metrics
                    for k in metrics_buffer:
                        metrics_buffer[k].append(info[k])

            # Save model every 100 episodes
            # Model saving and logging are handled at episode boundaries


            if time_step > 0 and time_step % self.args.evaluate_rate == 0:

                eval_reward = self.evaluate()
                returns.append(eval_reward)
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                plt.close()
                np.save(self.save_path + '/returns.npy', returns)


    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            obs_dict, _ = self.env.reset()
            # s = [obs_dict[agent] for agent in self.env.agents]
            # ✅ 同上：对齐观测长度
            s = [obs_dict[agent] for agent in list(self.env.agents)[:self.args.n_agents]]
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if episode == 0:
                    self.env.render()
                # actions = []
                # with torch.no_grad():
                #     for agent_id, agent in enumerate(self.agents):
                #         action = agent.select_action(s[agent_id], 0, 0)
                #         actions.append(action)
                        
                # action_dict = {agent: actions[i] for i, agent in enumerate(self.env.agents)}

                # obs_next, reward, term, trunc, info = self.env.step(action_dict)

                # s_next = [obs_next[agent] for agent in self.env.agents]
                # r = [reward[agent] for agent in self.env.agents]
                # done = [term[agent] or trunc[agent] for agent in self.env.agents]
                # rewards += np.sum(r)
                # s = s_next
                # ==========md2
                actions = []
                with torch.no_grad():
                    for agent_id in range(len(s)):
                        action = self.agents[agent_id].select_action(s[agent_id], 0, 0)
                        actions.append(action)
                
                # ✅ 新增：对齐长度
                active_agents = list(self.env.agents)[:len(actions)]
                action_dict = {agent: actions[i] for i, agent in enumerate(active_agents)}

                obs_next, reward, term, trunc, info = self.env.step(action_dict)

                s_next = [obs_next[agent] for agent in active_agents]
                r = [reward.get(agent, 0.0) for agent in active_agents]
                done = [(term.get(agent, False) or trunc.get(agent, False)) for agent in active_agents]
                rewards += np.sum(r)
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
