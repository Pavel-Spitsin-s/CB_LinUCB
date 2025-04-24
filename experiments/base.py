import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from tqdm import tqdm

class BaseExperiment:
    def __init__(self):
        pass
    
    def run_experiments(self, bandit_configs, num_experiments=10, num_rounds=1000):
        """
        Runs experiments for multiple bandit configurations.
        
        Args:
            bandit_configs: List of dictionaries with bandit configurations
            num_experiments: Number of experiments to run
            num_rounds: Number of rounds per experiment
        """
        bandits = self._setup_bandits(bandit_configs)
        
        # Initialize history trackers
        cumulative_rewards_history = {bandit['name']: np.zeros((num_experiments, num_rounds)) 
                                      for bandit in bandits}
        cumulative_regrets_history = {bandit['name']: np.zeros((num_experiments, num_rounds)) 
                                      for bandit in bandits}
        ctr_history = {bandit['name']: np.zeros((num_experiments, num_rounds)) 
                       for bandit in bandits}
        
        start_time = time.time()
        
        for exp in tqdm(range(num_experiments)):
            self._run_single_experiment(
                bandits, exp, num_rounds, 
                cumulative_rewards_history, 
                cumulative_regrets_history, 
                ctr_history
            )
            
        end_time = time.time()
        duration = end_time - start_time
        mean_duration = duration / num_experiments
        print(f"Mean duration per experiment: {mean_duration:.2f} seconds")
        
        # Calculate mean values and visualize results
        self._visualize_results(
            cumulative_rewards_history, 
            cumulative_regrets_history, 
            ctr_history
        )
        
        return {
            'cumulative_rewards': cumulative_rewards_history,
            'cumulative_regrets': cumulative_regrets_history,
            'ctr': ctr_history
        }
    
    def _setup_bandits(self, bandit_configs):
        """Setup bandit instances from configurations"""
        raise NotImplementedError("Subclasses must implement _setup_bandits")
    
    def _run_single_experiment(self, bandits, exp_num, num_rounds, 
                              cumulative_rewards_history, 
                              cumulative_regrets_history, 
                              ctr_history):
        """Run a single experiment"""
        raise NotImplementedError("Subclasses must implement _run_single_experiment")
    
    def _visualize_results(self, cumulative_rewards_history, 
                          cumulative_regrets_history, ctr_history):
        """Visualize experiment results"""
        # Calculate mean values across experiments
        mean_cumulative_rewards = {name: np.mean(history, axis=0) 
                                  for name, history in cumulative_rewards_history.items()}
        mean_cumulative_regrets = {name: np.mean(history, axis=0) 
                                  for name, history in cumulative_regrets_history.items()}
        mean_ctr = {name: np.mean(history, axis=0) 
                   for name, history in ctr_history.items()}

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        for name, rewards in mean_cumulative_rewards.items():
            plt.plot(rewards, label=name)
        plt.grid(True)
        plt.xlabel('Round')
        plt.title('Cumulative Rewards over Rounds')
        plt.legend()

        plt.subplot(1, 3, 2)
        for name, regrets in mean_cumulative_regrets.items():
            plt.plot(regrets, label=name)
        plt.grid(True)
        plt.xlabel('Round')
        plt.title('Cumulative Regrets over Rounds')
        plt.legend()

        plt.subplot(1, 3, 3)
        for name, ctr in mean_ctr.items():
            plt.plot(ctr, label=name)
        plt.grid(True)
        plt.xlabel('Round')
        plt.title('CTR over Rounds')
        plt.legend()

        plt.tight_layout()
        plt.show()
