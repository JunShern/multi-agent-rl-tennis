from lib.agents import DDPGAgent
from lib.env import EnvUnityMLAgents
from lib.trainers import Trainer, MultiAgentIndependentTrainer, MADDPGTrainer
import argparse

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
    env = EnvUnityMLAgents(args.env_path, train_mode=True)
    # agent = DDPGAgent(env.state_size, env.action_size, random_seed=0)
    agents = [DDPGAgent(env.state_size, env.action_size, random_seed=0) for _ in range(env.num_agents)]

    # Train
    # trainer = Trainer(agent, env,
    #     average_window=args.average_window, 
    #     solve_score=args.solve_score,
    #     max_episodes=args.max_episodes,
    #     max_steps_per_episode=args.max_steps,
    #     save_dir=args.checkpoints_path)
    # trainer = MultiAgentIndependentTrainer(agents, env,
    #     average_window=args.average_window, 
    #     solve_score=args.solve_score,
    #     max_episodes=args.max_episodes,
    #     max_steps_per_episode=args.max_steps,
    #     save_dir=args.checkpoints_path)
    trainer = MADDPGTrainer(agents, env,
        average_window=args.average_window, 
        solve_score=args.solve_score,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.checkpoints_path)
    scores = trainer.train()
    env.close()