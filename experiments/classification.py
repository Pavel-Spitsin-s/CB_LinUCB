import numpy as np
import copy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from .base import BaseExperiment
from bandits.simple import EpsilonGreedy, UCB 
from bandits.linear import LinUCBDisjoint, LinUCBHybrid
from bandits.dynamic import DLinUCB

class ClassificationExperiment(BaseExperiment):
    def __init__(self, k=2, random_k_features=None, normalize=True, random_state=42):
        super().__init__()
        self.k = k
        self.random_k_features = random_k_features
        self.normalize = normalize
        self.random_state = random_state
        self.X, self.y = self.load_data()
        self.X, self.y = self._preprocess_data(self.X, self.y, self.normalize)
        self.n_arms = len(np.unique(self.y))
        self.d = self.X.shape[1]

    def load_data(self):
        """Load the classification dataset"""
        raise NotImplementedError("Subclasses should implement this method")

    def _preprocess_data(self, X, y, normalize):
        """Preprocess the dataset for bandit experiments"""
        np.random.seed(self.random_state)

        # Convert string labels to numeric if needed
        if isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Convert categorical features if needed
        categorical_indices = [i for i in range(X.shape[1]) if isinstance(X[0, i], str)]
        if categorical_indices:
            encoder = OneHotEncoder(sparse_output=False)
            X_categorical = encoder.fit_transform(X[:, categorical_indices])
            X_numeric = np.delete(X, categorical_indices, axis=1).astype(float)
            X = np.hstack((X_numeric, X_categorical))

        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Randomly select k features if specified
        if self.random_k_features is not None:
            k_indices = np.random.choice(X.shape[1], self.random_k_features, replace=False)
            X = X[:, k_indices]

        return X, y

    def _setup_bandits(self, bandit_configs):
        """Setup bandit instances from configurations"""
        bandits = []
        for config in bandit_configs:
            params = config['params'].copy()
            params['n_arms'] = self.n_arms
            if 'd' in config['class'].__init__.__code__.co_varnames:
                params['d'] = self.d
            if 'k' in config['class'].__init__.__code__.co_varnames:
                params['k'] = self.k
            class_instance = config['class'](**params)
            bandits.append({'name': config['name'], 'class_instance': class_instance})
        return bandits

    def _run_single_experiment(self, bandits, exp_num, num_rounds, 
                              cumulative_rewards_history, 
                              cumulative_regrets_history, 
                              ctr_history):
        """Run a single experiment"""
        bandit_instances = {bandit['name']: copy.deepcopy(bandit['class_instance']) 
                           for bandit in bandits}
        cumulative_rewards = {bandit['name']: 0 for bandit in bandits}
        correct_predictions = {bandit['name']: 0 for bandit in bandits}

        for round in range(num_rounds):
            # Select a random instance
            idx = np.random.randint(len(self.X))
            z = self.X[idx, :self.k]  # environment features (first k features)
            x_array = np.tile(self.X[idx], (self.n_arms, 1))  # shape [n_arms, d]
            true_class = self.y[idx]

            for bandit_info in bandits:
                bandit_name = bandit_info['name']
                bandit = bandit_instances[bandit_name]

                # Select an arm and update based on bandit type
                if isinstance(bandit, (EpsilonGreedy, UCB)):
                    chosen_arm = bandit.select_arm()
                    reward = 1 if (chosen_arm == true_class) else 0
                    bandit.update_arm(chosen_arm, reward)
                elif isinstance(bandit, LinUCBDisjoint):
                    chosen_arm = bandit.select_arm(x_array)
                    reward = 1 if (chosen_arm == true_class) else 0
                    bandit.update_arm(chosen_arm, reward, x_array)
                elif isinstance(bandit, LinUCBHybrid):
                    chosen_arm = bandit.select_arm(z, x_array)
                    reward = 1 if (chosen_arm == true_class) else 0
                    bandit.update_arm(chosen_arm, reward, z, x_array)
                elif isinstance(bandit, DLinUCB):
                    chosen_arm = bandit.select_arm(x_array)
                    reward = 1 if (chosen_arm == true_class) else 0
                    bandit.update_arm(chosen_arm, reward, x_array)

                # Update metrics
                cumulative_rewards[bandit_name] += reward
                correct_predictions[bandit_name] += reward

                cumulative_rewards_history[bandit_name][exp_num, round] = cumulative_rewards[bandit_name]
                cumulative_regrets_history[bandit_name][exp_num, round] = round + 1 - cumulative_rewards[bandit_name]
                ctr_history[bandit_name][exp_num, round] = correct_predictions[bandit_name] / (round + 1)
