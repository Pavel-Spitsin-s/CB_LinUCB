import numpy as np
import math
from .base import Bandit

class EpsilonGreedy(Bandit):
    def __init__(self, n_arms, eps=0.1):
        super().__init__(n_arms)
        self.eps = eps
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.n_arms)
        return np.argmax(self.values)

    def update_arm(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


class Softmax(Bandit):
    def __init__(self, n_arms, temperature=1.0):
        super().__init__(n_arms)
        self.temperature = temperature
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, *args, **kwargs) -> int:
        exp_values = np.exp(self.values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(self.n_arms, p=probs)

    def update_arm(self, chosen_arm: int, reward: float, *args, **kwargs):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

    def get_exploration_rate(self):
        best_arm = np.argmax(self.values)
        exp_values = np.exp(self.values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        return 1 - probs[best_arm]


class VDBE(Bandit):
    def __init__(self, n_arms, sigma=0.33, delta=None):
        super().__init__(n_arms)
        self.sigma = sigma
        self.delta = delta if delta is not None else 1.0 / n_arms
        self.epsilon = 1.0
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.prev_values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        return np.argmax(self.values)

    def update_arm(self, chosen_arm, reward):
        self.prev_values[chosen_arm] = self.values[chosen_arm]
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

        td_error = abs(self.values[chosen_arm] - self.prev_values[chosen_arm])
        f = (1 - np.exp(-abs(td_error) / self.sigma)) / (1 + np.exp(-abs(td_error) / self.sigma))
        self.epsilon = self.delta * f + (1 - self.delta) * self.epsilon


class UCB(Bandit):
    def __init__(self, n_arms: int, alpha: float = 1.0):
        super().__init__(n_arms)
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self):
        if self.total_counts < self.n_arms:
            return self.total_counts

        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                exploration_bonus = math.sqrt(self.alpha * math.log(self.total_counts)) / self.counts[arm]
                ucb_values[arm] = self.values[arm] + exploration_bonus

        return np.argmax(ucb_values)

    def update_arm(self, chosen_arm: int, reward: float):
        self.total_counts += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        current_value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * current_value + (1 / n) * reward
