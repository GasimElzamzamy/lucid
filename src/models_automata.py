import numpy as np
from tslearn.piecewise import PiecewiseAggregateApproximation as PAA
from tslearn.piecewise import SymbolicAggregateApproximation as SAX
import Levenshtein

class ProbabilisticAutomata:
    """
    White-Box Anomaly Detection using PAA, SAX, and Markov Transition Probabilities.
    """
    def __init__(self, window_size, alphabet_size, laplace_smoothing=1e-5):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.smoothing = laplace_smoothing
        
        # These will be populated during the fit() method
        self.vocabulary = set()
        self.transition_matrix = {}
        self.sax_transformer = None

    def fit(self, X_train):
        """
        TODO (Teammate): 
        1. Fit PAA and SAX on X_train.
        2. Convert X_train into sequences of letters (e.g., 'aab', 'abc').
        3. Build self.vocabulary containing all unique seen patterns.
        4. Calculate transition probabilities and store in self.transition_matrix.
        """
        pass

    def _handle_unseen_pattern(self, unseen_pattern):
        """
        TODO (Teammate):
        1. If a pattern in test data is not in self.vocabulary, calculate the
           Levenshtein distance against all known patterns.
        2. Return the closest known pattern.
        """
        pass

    def calculate_path_probability(self, sequence):
        """
        TODO (Teammate):
        1. Iterate through the sequence of SAX patterns.
        2. Multiply the transition probabilities between states.
        3. Return the final probability score and the list of transitions.
        """
        pass

    def predict(self, X_test, anomaly_threshold):
        """
        TODO (Teammate):
        1. Transform X_test using the fitted SAX transformer.
        2. Calculate path probabilities for the sequences.
        3. Flag sequences with a probability below the anomaly_threshold as 1 (anomaly).
        """
        pass