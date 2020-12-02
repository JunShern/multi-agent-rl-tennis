from unityagents import UnityEnvironment

class EnvUnityMLAgents:
    def __init__(self, file_name, train_mode=True):
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        self.train_mode = train_mode
        
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        self.num_agents = len(env_info.agents)
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        states = env_info.vector_observations
        self.state_size = states.shape[1]

        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        print('The state for the first agent looks like:', states[0])

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        rewards = env_info.rewards
        next_states = env_info.vector_observations
        dones = env_info.local_done
        return rewards, next_states, dones

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return rewards, next_states, dones

    def close(self):
        self.env.close()