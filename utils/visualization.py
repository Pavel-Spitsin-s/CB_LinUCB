import matplotlib.pyplot as plt
import numpy as np

def plot_experiments_results(results, title=None):
    """
    Plot experiment results.
    
    Args:
        results: Dictionary with 'cumulative_rewards', 'cumulative_regrets', 'ctr'
        title: Optional title for the figure
    """
    # Calculate mean values across experiments
    mean_cumulative_rewards = {name: np.mean(history, axis=0) 
                              for name, history in results['cumulative_rewards'].items()}
    mean_cumulative_regrets = {name: np.mean(history, axis=0) 
                              for name, history in results['cumulative_regrets'].items()}
    mean_ctr = {name: np.mean(history, axis=0) 
               for name, history in results['ctr'].items()}

    fig = plt.figure(figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=16)

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

def compare_parameter_sensitivity(results_list, param_values, param_name, metric='ctr'):
    """
    Compare the sensitivity of a bandit algorithm to parameter changes.
    
    Args:
        results_list: List of experiment results for different parameter values
        param_values: List of parameter values corresponding to results_list
        param_name: Name of the parameter being varied
        metric: Metric to compare ('ctr', 'cumulative_rewards', or 'cumulative_regrets')
    """
    plt.figure(figsize=(10, 6))
    
    for i, results in enumerate(results_list):
        if metric == 'ctr':
            metric_data = results['ctr']
        elif metric == 'cumulative_rewards':
            metric_data = results['cumulative_rewards']
        elif metric == 'cumulative_regrets':
            metric_data = results['cumulative_regrets']
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        # Get the first bandit name (assuming only one bandit per experiment)
        name = list(metric_data.keys())[0]
        mean_values = np.mean(metric_data[name], axis=0)
        
        plt.plot(mean_values, label=f"{param_name}={param_values[i]}")
    
    plt.grid(True)
    plt.xlabel('Round')
    if metric == 'ctr':
        plt.title('CTR over Rounds')
        plt.ylabel('CTR')
    elif metric == 'cumulative_rewards':
        plt.title('Cumulative Rewards over Rounds')
        plt.ylabel('Cumulative Reward')
    else:
        plt.title('Cumulative Regrets over Rounds')
        plt.ylabel('Cumulative Regret')
        
    plt.legend()
    plt.show()
