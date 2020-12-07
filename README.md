# multi-agent-rl-tennis
This repository contains my submission for Project 3: Collaboration and Competition of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details

The assignment is to train an agent that solves the Unity ML-Agents [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

The solution implements a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm based on [[1]](#maddpg_paper) to solve the environment. For implementation and algorithm details, please see [Report.md](Report.md).

![trained_agent](assets/trained_agents.gif)

_The goal of the environment is to effectively control two agents to play a simplified game of tennis. To see the full progression of both agents training from scratch, check out [this video](https://youtu.be/KMBmxojIP58)!_

#### Environment

_(The below description is replicated from the [udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) repository.)_

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting started

#### Prerequisites
- Python >= 3.6
- A GPU is not required; training on CPU can take up to 5 hours to solve the environment

#### Installation
1. Clone the repository.
```bash
git clone https://github.com/JunShern/multi-agent-rl-tennis.git
```

2. Create a virtual environment to manage your dependencies.
```bash
cd multi-agent-rl-tennis/
python3 -m venv .venv
source .venv/bin/activate # Activate the virtualenv
```

3. Install python dependencies
```bash
cd multi-agent-rl-tennis/python
pip install .
```

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    _(For training on a remote server without virtual display e.g. AWS, use "Linux Headless")_

    Place the file in the `./env` folder of this repository, and unzip (or decompress) the file.

## Instructions

There are two entrypoints for the project: `train.py` to train the model from scratch, and `run.py` to run a trained agent in the environment.

1. Activate the virtualenv
```bash
cd multi-agent-rl-tennis/
source .venv/bin/activate # Activate the virtualenv
```
2. Run training. (Skip if you just want to run using the solved models in `./checkpoints`) 
```bash
python train.py
``` 
3. To run a trained agent.
```bash
python run.py
```
4. Exit the virtualenv when done
```bash
deactivate
```

## References

- <a name="maddpg_paper">[1]</a> Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in neural information processing systems. 2017.