import numpy as np
from tqdm.notebook import tqdm


class TestBed:
    def __init__(self, k=10, n_problems=2000, method='sample_avg', eps=0.1, alpha=0.1, walk=False):
        self.k = k
        self.n_problems = n_problems
        self.action_values = np.vstack([np.random.normal(size=k) for _ in range(n_problems)])
        self.method = method
        self.n_moves = np.zeros(shape=(n_problems, k), dtype='int')
        self.eps = eps
        self.alpha = alpha
        self.estimated_values = np.zeros(shape=(n_problems, k))
        self.walk = walk
        self.average_reward = []

    def pick_action(self):
        greedy = np.random.uniform(size=self.n_problems) >= self.eps
        greedy_actions = self.estimated_values.argmax(axis=1)
        random_actions = np.random.choice(self.k, self.n_problems)
        return np.where(greedy, greedy_actions, random_actions)

    def reply_to_actions(self, actions):
        mean_rewards = self.action_values[np.arange(self.n_problems), actions]
        if self.walk:
            self.action_values += np.random.normal(loc=0, scale=0.01, size=(self.n_problems, self.k))
        return np.random.normal(loc=mean_rewards)

    def update_estimates(self, actions, replies):
        if self.method == 'constant_step':
            self.estimated_values[np.arange(self.n_problems), actions] += \
                self.alpha * (replies - self.estimated_values[np.arange(self.n_problems), actions])
        elif self.method == 'sample_avg':
            self.n_moves[np.arange(self.n_problems), actions] += 1
            self.estimated_values[np.arange(self.n_problems), actions] += \
                (replies - self.estimated_values[np.arange(self.n_problems), actions]) / \
                self.n_moves[np.arange(self.n_problems), actions]

    def play(self, n_turns):
        for _ in tqdm(range(n_turns)):
            actions = self.pick_action()
            replies = self.reply_to_actions(actions)
            self.average_reward.append(replies.mean())
            self.update_estimates(actions, replies)
