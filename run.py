from lib.agents import DDPGAgent
from lib.env import EnvUnityMLAgents
import argparse
import numpy as np
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained agent.')
    parser.add_argument('--env_path', type=str, default="./envs/Tennis_Linux/Tennis.x86_64", help='path to Unity ML Agents environnment file')
    parser.add_argument('--model_paths', nargs='+', 
        default=["./checkpoints/agent0_checkpoint.pth", "./checkpoints/agent1_checkpoint.pth"], 
        help='paths to .pth files containing models for each trained Agent')
    args = parser.parse_args()

    # Setup
    env = EnvUnityMLAgents(args.env_path, train_mode=False)
    if (len(args.model_paths) != env.num_agents):
        print("Incorrect number of input models: {} (should be {})".format(args.model_paths, env.num_agents))
        sys.exit()
    # Load trained agents
    agents = [DDPGAgent(env.state_size, env.action_size, random_seed=0) for _ in range(env.num_agents)]
    for agent, model_path in zip(agents, args.model_paths):
        agent.load(path=model_path)

    for i in range(10):
        scores = np.zeros(env.num_agents)
        _, states, _ = env.reset()
        while True:
            actions = [agent.act(states[a][np.newaxis, ...], add_noise=False) for a, agent in enumerate(agents)]
            rewards, states, dones = env.step(actions)
            scores += rewards
            if np.any(dones):
                break
        print("Episode {} score: {}".format(i, scores))

    env.close()
    print('Total score (averaged over agents): {}'.format(np.mean(scores)))