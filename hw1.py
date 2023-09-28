import gym
import numpy as np
import time


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, smoothing=None, l_value=None, q_value=0.6) -> None:
        self.state_n = state_n
        self.action_n = action_n
        self.smoothing = smoothing
        self.q_value = q_value
        self.l_value = l_value
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        probs = self.model[state] / np.sum(self.model[state])
        action = np.random.choice(np.arange(self.action_n), p=probs)
        return action
    
    def _policy_smoothing(self, new_model):
        for state in range(self.state_n):
            if not np.sum(new_model[state]) == 0:
                self.model[state] = new_model[state] * self.l_value + self.model[state] * (1 - self.l_value) 
    
    def _laplace_smoothing(self, new_model):
        for state in range(self.state_n):
            if not np.sum(new_model[state]) == 0:
                self.model[state] = new_model[state] + self.l_value

    def _no_smoothing(self, new_model):
        self.model[np.sum(new_model, axis=1) != 0] = new_model[np.sum(new_model, axis=1) != 0]
    
    def _get_elite_trajectories(self, trajectories):
        mean_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        quantile = np.quantile(mean_rewards, self.q_value)
        elite_trajectories = [trajectory for trajectory in trajectories if np.sum(trajectory['rewards']) > quantile]
        return elite_trajectories

    def fit(self, trajectories):
        """Fit model

        Attrs:
            smoothing_type (str): type of smoothing ['None', 'laplace', 'policy']
            smoothing_lambda (int/float): lamda-value 
        """
        new_model = np.zeros((self.state_n, self.action_n))

        elite_trajectories = self._get_elite_trajectories(trajectories)
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
        
        if self.smoothing == 'policy':
            self._policy_smoothing(new_model)
        elif self.smoothing == 'laplace':
            self._laplace_smoothing(new_model)
        elif not self.smoothing:
            self._no_smoothing(new_model)


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
            time.sleep(0.5)
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

if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    action_space = env.action_space.n
    obesrv_space = env.observation_space.n
    n_iterations = 500
    n_trajectories = 1000
    n_steps = 200
    q = 0.7
    l = 2

    agent = CrossEntropyAgent(action_n=action_space, state_n=obesrv_space, 
                              smoothing='laplace', l_value=l, q_value=q)
    
    start = time.time()
    for i in range(n_iterations):
        iteration_trajectories = []
        for n in range(n_trajectories):
            trajectory = get_trajectory(env, agent, n_steps)
            iteration_trajectories.append(trajectory)
        if i % 10 == 0:
            print(f'iteration: {i}/{n_iterations}')
            get_stats(iteration_trajectories, display=True)

        agent.fit(iteration_trajectories)
    print(f'total time: {time.time() - start}')
