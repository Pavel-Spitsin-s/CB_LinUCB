import numpy as np
from scipy import linalg
from .base import Bandit

class LinUCBDisjoint(Bandit):
    def __init__(self, n_arms, d, alpha=1.0):
        super().__init__(n_arms)
        self.d = d
        self.alpha = alpha
        self.arms = [self._create_arm() for i in range(n_arms)]

    def _create_arm(self):
        return {
            'A': np.identity(self.d),
            'b': np.zeros([self.d, 1])
        }

    def select_arm(self, x_array):
        highest_ucb = -float('inf')
        candidate_arms = []

        for i, arm in enumerate(self.arms):
            A_inv = np.linalg.inv(arm['A'])
            theta = np.dot(A_inv, arm['b'])
            x = x_array[i].reshape(-1, 1)

            p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

            if p > highest_ucb:
                highest_ucb = p
                candidate_arms = [i]
            elif p == highest_ucb:
                candidate_arms.append(i)

        return np.random.choice(candidate_arms)

    def update_arm(self, chosen_arm, reward, x_array):
        arm = self.arms[chosen_arm]
        x = x_array[chosen_arm].reshape(-1, 1)
        arm['A'] += np.dot(x, x.T)
        arm['b'] += reward * x


class LinUCBHybrid(Bandit):
    def __init__(self, n_arms, d, alpha=1.0, k=2):
        super().__init__(n_arms)
        self.d = d
        self.alpha = alpha
        self.k = k

        self.A0 = np.identity(self.k)
        self.b0 = np.zeros((self.k, 1))
        self.z = np.zeros((self.k, 1))

        self.arms = [self._create_arm() for i in range(n_arms)]

    def _create_arm(self):
        return {
            'A': np.identity(self.d),
            'B': np.zeros((self.d, self.k)),
            'b': np.zeros((self.d, 1))
        }

    def select_arm(self, z, x_array):
        self.z = z.reshape(-1, 1)

        beta_hat = np.dot(linalg.inv(self.A0), self.b0)

        highest_ucb = -float('inf')
        candidate_arms = []

        for i, arm in enumerate(self.arms):
            x = x_array[i].reshape(-1, 1)

            A_inv = linalg.inv(arm['A'])
            A0_inv = linalg.inv(self.A0)
            theta_hat = np.dot(A_inv, arm['b'] - np.dot(arm['B'], beta_hat))

            s1 = np.dot(self.z.T, np.dot(A0_inv, self.z))
            s2 = np.dot(self.z.T, np.dot(A0_inv, np.dot(arm['B'].T, np.dot(A_inv, x))))
            s3 = np.dot(x.T, np.dot(A_inv, x))
            s4 = np.dot(x.T, np.dot(A_inv, np.dot(arm['B'], np.dot(A0_inv,
                                np.dot(arm['B'].T, np.dot(A_inv, x))))))

            s = s1 - 2*s2 + s3 + s4
            p = np.dot(self.z.T, beta_hat) + np.dot(x.T, theta_hat) + self.alpha * np.sqrt(s)

            if p > highest_ucb:
                highest_ucb = p
                candidate_arms = [i]
            elif p == highest_ucb:
                candidate_arms.append(i)

        return np.random.choice(candidate_arms)

    def update_arm(self, chosen_arm, reward, z, x_array):
        self.z = z.reshape(-1, 1)
        arm = self.arms[chosen_arm]
        x = x_array[chosen_arm].reshape(-1, 1)

        self.A0 += np.dot(arm['B'].T, np.dot(linalg.inv(arm['A']), arm['B']))
        self.b0 += np.dot(arm['B'].T, np.dot(linalg.inv(arm['A']), arm['b']))

        arm['A'] += np.dot(x, x.T)
        arm['B'] += np.dot(x, self.z.T)
        arm['b'] += reward * x

        self.A0 += np.dot(self.z, self.z.T)
        self.A0 -= np.dot(arm['B'].T, np.dot(linalg.inv(arm['A']), arm['B']))

        self.b0 += reward * self.z
        self.b0 -= np.dot(arm['B'].T, np.dot(linalg.inv(arm['A']), arm['b']))
