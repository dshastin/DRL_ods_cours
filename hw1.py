import gym
import numpy as np
import time


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, smoothing=None, l_value=None) -> None:
        self.state_n = state_n
        self.action_n = action_n
        self.smoothing = smoothing
        self.l_value = l_value
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        probs = self.model[state]
        try:
            action = np.random.choice(np.arange(self.action_n), p=probs)
            return action
        except:
            print(np.sum(probs))


    
    def fit(self, elite_trajectories):
        """Fit model

        Attrs:
            smoothing_type (str): type of smoothing ['None', 'laplace', 'policy']
            smoothing_lambda (int/float): lamda-value 
        """
        new_model = np.zeros((self.state_n, self.action_n))

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) != 0:
                if not self.smoothing:
                    new_model[state] = new_model[state] / np.sum(new_model[state])
                elif self.smoothing == 'laplace':
                    new_model[state] = ((new_model[state] + self.l_value) / 
                                        (np.sum(new_model[state]) + self.l_value * self.action_n))
                elif self.smoothing == 'policy':
                    new_model[state] = (self.model[state] * (1 - self.l_value) + 
                                        (new_model[state] / np.sum(new_model[state])) * self.l_value)
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model


def get_trajectory(env, agent, max_steps, visualize=False):
    state = env.reset()
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    for s in range(max_steps):
        trajectory['states'].append(state)
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)

        if done:
            break
        if visualize:
            env.render()
            time.sleep(0.1)
    return trajectory

def get_stats(trajectories, display=True):
    iteration_stats = dict()
    iteration_stats['rewards'] = []
    iteration_stats['steps'] = []
    iteration_stats['deliveries'] = []
    iteration_stats['penalties'] = []

    for t in trajectories:
        iteration_stats['rewards'].append(np.sum(t['rewards']))
        iteration_stats['steps'].append(len(t['rewards']))
        iteration_stats['deliveries'].append(len(list(filter(lambda x: x == 20, t['rewards']))))
        iteration_stats['penalties'].append(len(list(filter(lambda x: x == -10, t['rewards']))))

    if display:
        print(f'\ttotal delivered: {np.sum(iteration_stats["deliveries"])}/{len(trajectories)}')
        print(f'\tmean penalties: {np.mean(iteration_stats["penalties"])}')
        print(f'\tmean steps: {np.mean(iteration_stats["steps"])}')
        print(f'\tmean reward: {np.mean(iteration_stats["rewards"])}')
    return iteration_stats

def get_elite_trajectories(trajectories, q):
        mean_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        quantile = np.quantile(mean_rewards, q)
        elite_trajectories = [trajectory for trajectory in trajectories if np.sum(trajectory['rewards']) > quantile]
        return elite_trajectories


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    action_space = env.action_space.n
    obesrv_space = env.observation_space.n
    n_iterations = 500
    n_trajectories = 1000
    n_steps = 200
    smoothing_type = 'laplace'
    q = 0.7
    l = 1

    agent = CrossEntropyAgent(action_n=action_space, state_n=obesrv_space, 
                              smoothing=smoothing_type, l_value=l)
    
    start = time.time()

    for i in range(n_iterations):
        iteration_trajectories = []
        for n in range(n_trajectories):
            trajectory = get_trajectory(env, agent, n_steps)
            iteration_trajectories.append(trajectory)
        if i % 10 == 0:
            print(f'iteration: {i}/{n_iterations}')
            get_stats(iteration_trajectories, display=True)

        elite_trajectories = get_elite_trajectories(iteration_trajectories, q)
        agent.fit(elite_trajectories)

    print(f'total time: {time.time() - start}')
