import gym
import numpy as np
import time


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, smoothing_type=None, l_value=None) -> None:
        self.state_n = state_n
        self.action_n = action_n
        self.smoothing = smoothing_type
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

def get_stats(trajectories, display=False):
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

def fit(env, agent, q, stochastic_env, n_iterations, n_trajectories, n_steps, n_packs, display=False):
    mean_rewards_list = []
    if not stochastic_env:
        for i in range(n_iterations):
            iteration_trajectories = []
            for n in range(n_trajectories):
                trajectory = get_trajectory(env, agent, n_steps)
                iteration_trajectories.append(trajectory)
            rewards = get_stats(iteration_trajectories)['rewards']
            mean_rewards_list.append(np.mean(rewards))
            if i % 10 == 0:
                print(f'iteration: {i}/{n_iterations}')
                get_stats(iteration_trajectories, display=display)

            elite_trajectories = get_elite_trajectories(iteration_trajectories, q)
            agent.fit(elite_trajectories)

    else:
        print('stochastic env')
        for iteration in range(n_iterations):
            iteration_seed = np.random.randint(0,31337)
            trajectory_packs = []
            for i in range(n_packs):
                env.reset(seed=iteration_seed)
                trajectory_pack = []
                for n in range(n_trajectories):
                    trajectory = get_trajectory(env, agent, n_steps)
                    trajectory_pack.append(trajectory)
                trajectory_packs.append(trajectory_pack)
            
            mean_pack_rewards = []
            for pack_idx in range(n_packs):
                total_rewards = [np.sum(t['rewards']) for t in trajectory_packs[pack_idx]]
                mean_pack_rewards.append(np.mean(total_rewards))

            quantile = np.quantile(mean_pack_rewards, q)
            elite_packs_idx = np.where(np.array(mean_pack_rewards) > quantile)
            elite_packs = np.array(trajectory_packs)[elite_packs_idx]

            print(f'iteration: {iteration}/{n_iterations}')
            get_stats(np.array(trajectory_packs).reshape(-1), display=True)

            agent.fit(elite_packs.reshape(-1))
    return mean_rewards_list

if __name__ == '__main__':
    agent_params = dict(
        action_n = 6,
        state_n = 500,
        smoothing_type = None,
        l_value = 0.1,
    )

    fit_params = dict(
        n_iterations = 200,
        n_trajectories = 500,
        n_packs = 20,
        n_steps = 200,
        stochastic_env = False,
        q = 0.6,
    )

    env = gym.make('Taxi-v3')
    agent = CrossEntropyAgent(**agent_params)
    rewards = fit(env, agent, **fit_params)
