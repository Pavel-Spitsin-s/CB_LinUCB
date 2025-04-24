from sklearn.datasets import load_iris, fetch_openml
from experiments.classification import ClassificationExperiment

class Iris(ClassificationExperiment):
    """Iris flower classification experiment"""
    def load_data(self):
        iris = load_iris()
        return iris.data, iris.target

class Mushroom(ClassificationExperiment):
    """Mushroom classification experiment"""
    def load_data(self):
        mushroom = fetch_openml('mushroom', version=1, as_frame=False)
        return mushroom.data, mushroom.target
