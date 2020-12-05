from lib import utils
from lib.agents import BATCH_SIZE, LEARN_EVERY, UPDATES_PER_LEARN, GAMMA
import matplotlib.pyplot as plt
import numpy as np
import os
import time

"""
Basic trainer for a single agent acting jointly over possibly multiple agents in the environment.
"""
class Trainer():
    def __init__(self, agent, env, average_window = 100, solve_score = 30, max_episodes = 200, max_steps_per_episode = 1000, save_dir = "./"):
        self.agent = agent
        self.env = env
        self.average_window = average_window
        self.solve_score = solve_score
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_dir = save_dir
        self.last_time = time.time()
        self.all_agent_scores = []
    
    def train(self):
        self.all_agent_scores = []
        for i in range(self.max_episodes):

            _, states, _ = self.env.reset()
            scores = np.zeros(self.env.num_agents)
            for timestep in range(self.max_steps_per_episode):
                # Select an action
                actions = self.agent.act(states, add_noise=True) # Network expects inputs in batches so feed all at once

                # Act
                rewards, next_states, dones = self.env.step(actions)

                # Learn
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    self.agent.step(state, action, reward, next_state, done, timestep)

                # Update
                scores += rewards
                states = next_states
                
                # Exit if any of the agents finish 
                if np.any(dones):
                    break
                
            print("Episode {} score: {}, timesteps: {}, epsilon: {}".format(i, scores, timestep, self.agent.epsilon))
            self.all_agent_scores.append(scores)
            t = time.time()
            mvg_avg = np.mean(self.all_agent_scores[-self.average_window:])

            print('Episode {} ({:.2f}s) -- Min: {:.2f} -- Max: {:.2f} -- Mean: {:.2f} -- Moving Average: {:.2f}'
                .format(i, t - self.last_time, np.min(scores), np.max(scores), np.mean(scores), mvg_avg))
            self.last_time = t
            # if mvg_avg > self.solve_score and len(all_agent_scores) >= self.average_window:
            #     break
            
            if i % 100 == 0:
                self.save_progress(i)
        
        self.save_progress(i)
        return self.all_agent_scores

    def save_progress(self, episode_idx):
        file_prefix = "{:04d}_".format(episode_idx)

        # Save model
        self.agent.save(os.path.join(self.save_dir, file_prefix + 'checkpoint.pth'))

        # Save scores for analysis
        scores = np.array(self.all_agent_scores)
        np.save(os.path.join(self.save_dir, file_prefix + 'scores.npy'), scores)

        # Save plot of results
        plt.figure(figsize=(20, 10))
        for agent_idx in range(scores.shape[1]):
            plt.plot(scores[:, agent_idx])
        plt.fill_between(x=range(len(scores)), y1=scores.min(axis=1), y2=scores.max(axis=1), alpha=0.2)
        mvg_avgs = utils.moving_averages(np.mean(scores, axis=1), window=self.average_window)
        plt.axhline(y = self.solve_score, color="red")
        plt.plot(mvg_avgs, color='black', linewidth=2)
        plt.ylabel("score")
        plt.xlabel("episode")
        plt.savefig(os.path.join(self.save_dir, file_prefix + "scores.png"))
        plt.close()

"""
Multiple agents acting independently with no sharing of information.
"""
class MultiAgentIndependentTrainer():
    def __init__(self, agents, env, average_window = 100, solve_score = 30, max_episodes = 200, max_steps_per_episode = 1000, save_dir = "./"):
        self.agents = agents
        self.env = env
        self.average_window = average_window
        self.solve_score = solve_score
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_dir = save_dir
        self.last_time = time.time()
        self.all_agent_scores = []
    
    def train(self):
        self.all_agent_scores = []
        for i in range(self.max_episodes):

            _, states, _ = self.env.reset()
            scores = np.zeros(self.env.num_agents)
            for timestep in range(self.max_steps_per_episode):
                # Select an action
                actions = [agent.act(states[a][np.newaxis, ...], add_noise=True) for a, agent in enumerate(self.agents)]

                # Act
                rewards, next_states, dones = self.env.step(actions)

                # Learn
                for agent, state, action, reward, next_state, done in zip(self.agents, states, actions, rewards, next_states, dones):
                    agent.step(state, action, reward, next_state, done, timestep)

                # Update
                scores += rewards
                states = next_states
                
                # Exit if any of the agents finish 
                if np.any(dones):
                    break
                
            print("Episode {} score: {}, timesteps: {}, epsilon: {}".format(i, scores, timestep, self.agents[0].epsilon))
            self.all_agent_scores.append(scores)
            t = time.time()
            mvg_avg = np.mean(self.all_agent_scores[-self.average_window:])

            print('Episode {} ({:.2f}s) -- Min: {:.2f} -- Max: {:.2f} -- Mean: {:.2f} -- Moving Average: {:.2f}'
                .format(i, t - self.last_time, np.min(scores), np.max(scores), np.mean(scores), mvg_avg))
            self.last_time = t
            # if mvg_avg > self.solve_score and len(all_agent_scores) >= self.average_window:
            #     break
            
            if i % 100 == 0:
                self.save_progress(i)
        
        self.save_progress(i)
        return self.all_agent_scores

    def save_progress(self, episode_idx):
        file_prefix = "{:04d}_".format(episode_idx)

        # Save model
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(self.save_dir, file_prefix + 'agent{}_checkpoint.pth'.format(i)))

        # Save scores for analysis
        scores = np.array(self.all_agent_scores)
        np.save(os.path.join(self.save_dir, file_prefix + 'scores.npy'), scores)

        # Save plot of results
        plt.figure(figsize=(20, 10))
        for agent_idx in range(scores.shape[1]):
            plt.plot(scores[:, agent_idx])
        plt.fill_between(x=range(len(scores)), y1=scores.min(axis=1), y2=scores.max(axis=1), alpha=0.2)
        mvg_avgs = utils.moving_averages(np.mean(scores, axis=1), window=self.average_window)
        plt.axhline(y = self.solve_score, color="red")
        plt.plot(mvg_avgs, color='black', linewidth=2)
        plt.ylabel("score")
        plt.xlabel("episode")
        plt.savefig(os.path.join(self.save_dir, file_prefix + "scores.png"))
        plt.close()

"""
Multi-agent trainer with centralized training (update step) and decentralized execution (action step).
"""
class MADDPGTrainer(MultiAgentIndependentTrainer):
    def train(self):
        self.all_agent_scores = []
        for i in range(self.max_episodes):

            _, states, _ = self.env.reset()
            scores = np.zeros(self.env.num_agents)
            for timestep in range(self.max_steps_per_episode):
                # Select an action
                actions = [agent.act(states[a][np.newaxis, ...], add_noise=True) for a, agent in enumerate(self.agents)]

                # Act
                rewards, next_states, dones = self.env.step(actions)

                # Save to SHARED replay buffer
                # Using agent[0]'s buffer as the shared replay buffer
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    self.agents[0].memory.add(state, action, reward, next_state, done)
                
                # all_states = states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
                # all_next_states = next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
                # self.agents[0].memory.add(all_states, actions, rewards, all_next_states, dones)

                for agent in self.agents:
                    # Learn if enough samples are available in memory
                    if len(self.agents[0].memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
                        for _ in range(UPDATES_PER_LEARN):
                            experiences = self.agents[0].memory.sample() # Everyone samples from the shared buffer
                            agent.learn(experiences, GAMMA)
                
                # Update
                scores += rewards
                states = next_states
                
                # Exit if any of the agents finish 
                if np.any(dones):
                    break
                
            print("Episode {} score: {}, timesteps: {}, epsilon: {}".format(i, scores, timestep, self.agents[0].epsilon))
            self.all_agent_scores.append(scores)
            t = time.time()
            mvg_avg = np.mean(self.all_agent_scores[-self.average_window:])

            print('Episode {} ({:.2f}s) -- Min: {:.2f} -- Max: {:.2f} -- Mean: {:.2f} -- Moving Average: {:.2f}'
                .format(i, t - self.last_time, np.min(scores), np.max(scores), np.mean(scores), mvg_avg))
            self.last_time = t
            # if mvg_avg > self.solve_score and len(all_agent_scores) >= self.average_window:
            #     break
            
            if i % 100 == 0:
                self.save_progress(i)
        
        self.save_progress(i)
        return self.all_agent_scores
