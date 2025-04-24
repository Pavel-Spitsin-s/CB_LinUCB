import numpy as np

class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
    
    def select_arm(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement select_arm")
    
    def update_arm(self, chosen_arm, reward, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement update_arm")
