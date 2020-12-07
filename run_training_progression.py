from lib.agents import DDPGAgent
from lib.env import EnvUnityMLAgents
import argparse
import glob
import numpy as np
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained agent.')
    parser.add_argument('--env_path', type=str, default="./envs/Tennis_Linux/Tennis.x86_64", help='path to Unity ML Agents environnment file')
    parser.add_argument('--models_path', type=str, default="./checkpoints", 
        help="path containing .pth files for multi-epoch sets of Agent models following [EPOCH]_agent[AGENT_IDX]_checkpoint.pth naming convention.")
    parser.add_argument('--steps_per_epoch', type=int, default=200, help='Number of timesteps to show at each epoch')
    args = parser.parse_args()

    # Setup
    env = EnvUnityMLAgents(args.env_path, train_mode=False)
    
    # Load trained agents
    model_paths = [
        sorted(glob.glob(os.path.join(args.models_path, "*_agent{}_checkpoint.pth".format(i)))) 
        for i in range(env.num_agents)
    ]
    model_paths = list(zip(*model_paths)) # Convert AGENT-length list of epochs to EPOCH-length list of agents

    sets_of_agents = []
    for epoch_paths in model_paths:
        agents = [DDPGAgent(env.state_size, env.action_size, random_seed=0) for _ in range(env.num_agents)]
        for agent, model_path in zip(agents, epoch_paths):
            agent.load(path=model_path)
        sets_of_agents.append(agents)
    
    for i, agents in enumerate(sets_of_agents):
        print("Currently showing", model_paths[i])
        steps = 0
        while steps < args.steps_per_epoch:
            scores = np.zeros(env.num_agents)
            _, states, _ = env.reset()
            while True:
                actions = [agent.act(states[a][np.newaxis, ...], add_noise=False) for a, agent in enumerate(agents)]
                rewards, states, dones = env.step(actions)
                scores += rewards
                steps += 1
                if np.any(dones) or steps >= args.steps_per_epoch:
                    break
    env.close()