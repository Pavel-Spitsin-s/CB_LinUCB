import numpy as np
from .base import Bandit

class DLinUCB(Bandit):
    def __init__(self, n_arms, d, alpha, gamma=0.95, lambda_=1.0):
        super().__init__(n_arms)
        self.d = d
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        self.arms = [self._create_arm() for _ in range(n_arms)]
        self.t = 0
        self.gamma2_t = 1

    def _create_arm(self):
        return {
            'hat_theta': np.zeros(self.d),
            'cov': self.lambda_ * np.identity(self.d),
            'cov_squared': self.lambda_ * np.identity(self.d),
            'b': np.zeros(self.d)
        }

    def select_arm(self, x_array):
        delta = 0.1
        c_delta = 2 * np.log(1 / delta)
        const1 = np.sqrt(self.lambda_) * 1.0
        beta_t = const1 + np.sqrt(c_delta + self.d * np.log(1 + (1-self.gamma2_t)/
                      (self.d * self.lambda_ * (1 - self.gamma**2))))

        best_ucb = -float('inf')
        candidate_arms = []

        for i, arm in enumerate(self.arms):
            x = x_array[i]
            invcov = np.linalg.pinv(arm['cov'])
            invcov_a = invcov @ arm['cov_squared'] @ invcov @ x
            current_ucb = np.dot(arm['hat_theta'], x) + self.alpha * beta_t * np.sqrt(np.dot(x, invcov_a))

            if current_ucb > best_ucb:
                best_ucb = current_ucb
                candidate_arms = [i]
            elif current_ucb == best_ucb:
                candidate_arms.append(i)

        return np.random.choice(candidate_arms)

    def update_arm(self, chosen_arm, reward, x_array):
        arm = self.arms[chosen_arm]
        x = x_array[chosen_arm]

        # update model parameters with discounting
        aat = np.outer(x, x)
        self.gamma2_t *= self.gamma ** 2

        arm['cov'] = self.gamma * arm['cov'] + aat + (1-self.gamma) * self.lambda_ * np.identity(self.d)
        arm['cov_squared'] = (self.gamma ** 2 * arm['cov_squared'] + aat +
                           (1-self.gamma**2) * self.lambda_ * np.identity(self.d))
        arm['b'] = self.gamma * arm['b'] + reward * x

        arm['hat_theta'] = np.linalg.pinv(arm['cov']) @ arm['b']
        self.t += 1
