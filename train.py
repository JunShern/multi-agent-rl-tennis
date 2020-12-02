from lib import env, agents
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time

class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.last_time = time.time()

    def train(self, average_window = 100, solve_score = 30, max_episodes = 200, save_dir = "./"):
        all_agent_scores = []
        for i in range(max_episodes):

            _, states, _ = env.reset()
            scores = np.zeros(self.env.num_agents)
            timestep = 0
            while True:
                # select an action (for each agent)
                actions = self.agent.act(states, add_noise=True) # Network expects inputs in batches so feed all at once

                # Act
                rewards, next_states, dones = self.env.step(actions)
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    self.agent.step(state, action, reward, next_state, done, timestep)

                # Update
                scores += rewards
                states = next_states
                
                # Exit if any of the agents finish 
                if np.any(dones):
                    # This termination criteria may cause us to never learn the late-game states,
                    # but it should be fine in this case since the problem does not evolve over time
                    break
                timestep += 1
            
            all_agent_scores.append(scores)
            t = time.time()
            mvg_avg = np.mean(all_agent_scores[-average_window:])

            print('Episode {} ({:.2f}s) -- Min: {:.2f} -- Max: {:.2f} -- Mean: {:.2f} -- Moving Average: {:.2f}'
                .format(i, t - self.last_time, np.min(scores), np.max(scores), np.mean(scores), mvg_avg))
            self.last_time = t
            if mvg_avg > 30 and len(all_agent_scores) >= 100:
                break
        
        # Save model
        self.agent.save(path=save_dir)
        # Save scores for analysis
        all_agent_scores = np.array(all_agent_scores)
        np.save(os.path.join(save_dir, 'all_scores.npy'), all_scores)
        return 

def moving_averages(values, window=100):
    return [np.mean(values[:i+1][-window:]) for i, _ in enumerate(values)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--env_path', type=str, default="./envs/Tennis_Linux_NoVis/Tennis.x86_64", help='path to Unity ML Agents environnment file')
    parser.add_argument('--average_window', type=int, default=100, help='window size for moving average score')
    parser.add_argument('--solve_score', type=int, default=30, help='target score to consider training solved')
    parser.add_argument('--max_episodes', type=int, default=200, help='maximum number of training episodes')
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints", help='path to save checkpoint_actor.pth and checkpoint_critic.pth models')
    args = parser.parse_args()

    # Setup
    env = env.EnvUnityMLAgents(args.env_path, train_mode=True)
    agent = agents.DDPGAgent(env.state_size, env.action_size, random_seed=0)
    trainer = Trainer(agent, env)

    # Train
    all_scores = trainer.train(
        average_window=args.average_window, 
        solve_score=args.solve_score,
        max_episodes=args.max_episodes,
        save_dir=args.checkpoints_path)
    env.close()

    # Plot results
    plt.figure(figsize=(20, 10))
    for agent_idx in range(all_scores.shape[1]):
        plt.plot(all_scores[:, agent_idx])
    plt.fill_between(x=range(len(all_scores)), y1=all_scores.min(axis=1), y2=all_scores.max(axis=1), alpha=0.2)
    mvg_avgs = moving_averages(np.mean(all_scores, axis=1), window=args.average_window)
    print(mvg_avgs)
    plt.axhline(y = args.solve_score, color="red")
    plt.plot(mvg_avgs, color='black', linewidth=2)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.savefig("scores.png")