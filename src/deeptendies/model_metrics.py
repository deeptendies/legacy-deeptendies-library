from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np

class ModelMetrics:
    """
    This helper class is for computing 
    evaluation metrics for developed models
    """

    def __init__(self, y_true, y_predicted):
        """
        Params:
        y_true: the true results (np array)
        y_predicted: the predicted results (np array)
        Returns:
        a ModelMetrics object
        """
        self.y_true = y_true
        self.y_predicted = y_predicted
        self.mean_percentage_error_val = np.mean((y_true-y_predicted)/y_true)

    def mean_directional_accuracy(self):
        """
        This function is used to compute the mean directional
        accuracy based on number of correct rise/fall predictions
        Returns:
        mean directional accuracy
        """
        return np.mean((np.sign(self.y_true[1:] - self.y_true[:-1]) == np.sign(self.y_predicted[1:] - self.y_predicted[:-1])).astype(int)) 
    
    def mean_percentage_error(self):
        """
        This function returns the mean percentage error
        """
        return self.mean_percentage_error_val

    @staticmethod
    def mean_prediction_accuracy(models):
        """
        This function is used to calculate mean 
        prediction accuracy based on multiple stocks
        Params: models: list of ModelMetrics objects for each stock's results
        Returns: mean prediction accuracy
        """
        number_of_models = len(models)
        sum_mean_percentage_error = 0
        for m in models:
            sum_mean_percentage_error = sum_mean_percentage_error + m.mean_percentage_error()
        return 1 - (sum_mean_percentage_error/number_of_models)

    @staticmethod
    def print_mean_prediction_accuracy(models):
        """
        Computes and prints mean prediction accuracy
        """
        print("Mean prediction accuracy:\t", ModelMetrics.mean_prediction_accuracy(models))

    def mean_absolute_percentage_error(self): 
        """
        Returns:
        Mean absolute percentage error regression metric
        """
        return np.mean(np.abs((self.y_true - self.y_predicted) / self.y_true))


    def mean_squared_error(self):
        """
        Returns:
        Mean squared error regression metric
        """
        return mean_squared_error(self.y_true, self.y_predicted)

    def root_mean_squared_error(self):
        """
        Returns:
        Root mean squared error regression metric
        """
        return sqrt(self.mean_squared_error())

    def mean_absolute_error(self):
        """
        Mean absolute error metric
        """
        return mean_absolute_error(self.y_true, self.y_predicted)

    def print_metrics(self):
        """
        This function is used to print out model metrics
        """
        print("Mean squared error:\t\t", self.mean_squared_error())
        print("Root mean squared error:\t", self.root_mean_squared_error())
        print("Mean absolute error:\t\t", self.mean_absolute_error())
        print("Mean absolute percentage error:\t", self.mean_absolute_percentage_error())
        print("Mean percentage error:\t\t", self.mean_percentage_error())
        print("Mean directional accuracy:\t", self.mean_directional_accuracy())