from lib import env, agents
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time

class Trainer:
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

            _, states, _ = env.reset()
            scores = np.zeros(self.env.num_agents)
            for timestep in range(self.max_steps_per_episode):
                # select an action (for each agent)
                actions = self.agent.act(states, add_noise=True) # Network expects inputs in batches so feed all at once

                # Act
                rewards, next_states, dones = self.env.step(actions)
                self.agent.step(states, actions, rewards, next_states, dones, timestep)

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
        mvg_avgs = moving_averages(np.mean(scores, axis=1), window=self.average_window)
        plt.axhline(y = self.solve_score, color="red")
        plt.plot(mvg_avgs, color='black', linewidth=2)
        plt.ylabel("score")
        plt.xlabel("episode")
        plt.savefig(os.path.join(self.save_dir, file_prefix + "scores.png"))

def moving_averages(values, window=100):
    return [np.mean(values[:i+1][-window:]) for i, _ in enumerate(values)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--env_path', type=str, default="./envs/Tennis_Linux_NoVis/Tennis.x86_64", help='path to Unity ML Agents environnment file')
    parser.add_argument('--average_window', type=int, default=100, help='window size for moving average score')
    parser.add_argument('--solve_score', type=int, default=0.5, help='target score to consider training solved')
    parser.add_argument('--max_episodes', type=int, default=1000, help='maximum number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='maximum time steps per training episode')
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints", help='path to save checkpoint files including models, scores, and plots')
    args = parser.parse_args()

    # Setup
    env = env.EnvUnityMLAgents(args.env_path, train_mode=True)
    agent = agents.DDPGAgent(env.state_size, env.action_size, random_seed=0)

    # Train
    trainer = Trainer(agent, env,
        average_window=args.average_window, 
        solve_score=args.solve_score,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.checkpoints_path)
    scores = trainer.train()
    env.close()