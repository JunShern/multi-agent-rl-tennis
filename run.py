from lib import env, agents
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained agent.')
    parser.add_argument('--env_path', type=str, default="./envs/Tennis_Linux/Tennis.x86_64", help='path to Unity ML Agents environnment file')
    parser.add_argument('--model_path', type=str, default="./checkpoints/checkpoint.pth", help='path to .pth file containing actor and critic models')
    args = parser.parse_args()

    # Setup
    env = env.EnvUnityMLAgents(args.env_path, train_mode=False)
    agent = agents.DDPGAgent(env.state_size, env.action_size, random_seed=0)
    agent.load(path=args.model_path)

    for i in range(10):
        scores = np.zeros(env.num_agents)
        _, states, _ = env.reset()
        while True:
            actions = agent.act(states, add_noise=False)
            rewards, states, dones = env.step(actions)
            scores += rewards
            if np.any(dones):
                break
        print("Episode {} score: {}".format(i, scores))

    env.close()
    print('Total score (averaged over agents): {}'.format(np.mean(scores)))