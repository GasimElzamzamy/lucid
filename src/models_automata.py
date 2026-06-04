import math
from collections import defaultdict

import numpy as np
from tslearn.piecewise import PiecewiseAggregateApproximation as PAA
from tslearn.piecewise import SymbolicAggregateApproximation as SAX
import Levenshtein


class ProbabilisticAutomata:
    """
    Interpretable anomaly detector based on SAX patterns and probabilistic
    state transitions.

    The model converts a 1D time series into SAX symbols, groups consecutive
    symbols into fixed-size patterns, and learns transition probabilities
    between those patterns from the training data.
    """

    def __init__(self, window_size, alphabet_size, laplace_smoothing=1e-5):
        self.window_size = int(window_size)
        self.alphabet_size = int(alphabet_size)
        self.smoothing = float(laplace_smoothing)

        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.alphabet_size < 2 or self.alphabet_size > 26:
            raise ValueError("alphabet_size must be between 2 and 26")
        if self.smoothing < 0:
            raise ValueError("laplace_smoothing must be >= 0")

        self.vocabulary = set()
        self.transition_matrix = {}
        self.transition_counts = {}
        self.sax_transformer = None
        self.paa_transformer = None
        self.last_explanations = []

    def fit(self, X_train):
        """
        Learn the SAX vocabulary and transition probability matrix.

        Parameters
        ----------
        X_train : array-like
            One-dimensional time series, or a two-dimensional array with a
            single feature column after PCA/PC1 extraction.
        """
        symbols = self._series_to_symbols(X_train)
        patterns = self._symbols_to_patterns(symbols)

        self.vocabulary = set(patterns)
        self.transition_counts = self._build_transition_counts(patterns)
        self.transition_matrix = self._counts_to_probabilities(self.transition_counts)
        return self

    def _handle_unseen_pattern(self, unseen_pattern):
        """
        Map an unseen SAX pattern to the nearest known pattern by Levenshtein
        distance. Ties are resolved lexicographically for reproducibility.
        """
        if not self.vocabulary:
            raise ValueError("The automata vocabulary is empty. Call fit() first.")

        unseen_pattern = str(unseen_pattern)
        return min(
            self.vocabulary,
            key=lambda known: (Levenshtein.distance(unseen_pattern, known), known),
        )

    def calculate_path_probability(self, sequence):
        """
        Calculate the product of transition probabilities for a pattern path.

        Returns
        -------
        tuple
            (path_probability, transitions)
            transitions is a list of dictionaries containing from/to states,
            mapped states, unseen flags, and transition probabilities.
        """
        if not self.transition_matrix:
            raise ValueError("Transition matrix is empty. Call fit() first.")

        mapped_sequence = []
        pattern_info = []

        for pattern in sequence:
            pattern = str(pattern)
            if pattern in self.vocabulary:
                mapped = pattern
                status = "seen"
            else:
                mapped = self._handle_unseen_pattern(pattern)
                status = "unseen"

            mapped_sequence.append(mapped)
            pattern_info.append({
                "pattern": pattern,
                "mapped_to": mapped,
                "status": status,
            })

        transitions = []
        log_probability = 0.0

        for idx in range(len(mapped_sequence) - 1):
            from_state = mapped_sequence[idx]
            to_state = mapped_sequence[idx + 1]
            probability = self._transition_probability(from_state, to_state)

            log_probability += math.log(max(probability, 1e-300))
            transitions.append({
                "from_state": from_state,
                "to_state": to_state,
                "original_from_state": pattern_info[idx]["pattern"],
                "original_to_state": pattern_info[idx + 1]["pattern"],
                "from_status": pattern_info[idx]["status"],
                "to_status": pattern_info[idx + 1]["status"],
                "probability": probability,
            })

        path_probability = float(math.exp(log_probability)) if transitions else 1.0
        return path_probability, transitions

    def predict(self, X_test, anomaly_threshold):
        """
        Predict anomalies from transition probabilities.

        A transition is marked as anomaly when its probability is below the
        provided threshold. The first pattern has no previous state, so it is
        treated as normal unless it is unseen.
        """
        symbols = self._series_to_symbols(X_test)
        patterns = self._symbols_to_patterns(symbols)

        predictions = []
        explanations = []
        previous_mapped = None

        for time_step, pattern in enumerate(patterns):
            if pattern in self.vocabulary:
                mapped = pattern
                status = "seen"
            else:
                mapped = self._handle_unseen_pattern(pattern)
                status = "unseen"

            if previous_mapped is None:
                probability = 1.0
                transition = None
            else:
                probability = self._transition_probability(previous_mapped, mapped)
                transition = {
                    "from_state": previous_mapped,
                    "to_state": mapped,
                    "probability": probability,
                }

            decision = "anomaly" if probability < anomaly_threshold else "normal"
            predictions.append(1 if decision == "anomaly" else 0)

            explanations.append({
                "time_step": time_step,
                "state": previous_mapped,
                "pattern": pattern,
                "status": status,
                "mapped_to": None if status == "seen" else mapped,
                "transition": transition,
                "probability": float(probability),
                "decision": decision,
                "confidence_score": float(probability),
            })

            previous_mapped = mapped

        self.last_explanations = explanations
        return np.array(predictions, dtype=int)

    def transform_to_patterns(self, X):
        """Public helper for tests and explainability code."""
        return self._symbols_to_patterns(self._series_to_symbols(X))

    def _series_to_symbols(self, X, fit_transformer=False):
        series = self._as_1d_series(X)
        if len(series) < self.window_size:
            raise ValueError("Time series length must be at least window_size")

        ts = series.reshape(1, -1, 1)

        if fit_transformer or self.sax_transformer is None:
            self.paa_transformer = PAA(n_segments=len(series))
            _ = self.paa_transformer.fit_transform(ts)

            self.sax_transformer = SAX(
                n_segments=len(series),
                alphabet_size_avg=self.alphabet_size,
                scale=False,
            )
            sax_values = self.sax_transformer.fit_transform(ts).reshape(-1)
        else:
            sax_values = self.sax_transformer.transform(ts).reshape(-1)

        return "".join(self._symbol_to_char(value) for value in sax_values)

    def _symbols_to_patterns(self, symbols):
        return [
            symbols[i:i + self.window_size]
            for i in range(len(symbols) - self.window_size + 1)
        ]

    def _build_transition_counts(self, patterns):
        counts = defaultdict(lambda: defaultdict(int))
        for current_state, next_state in zip(patterns[:-1], patterns[1:]):
            counts[current_state][next_state] += 1
        return {state: dict(next_states) for state, next_states in counts.items()}

    def _counts_to_probabilities(self, transition_counts):
        matrix = {}
        vocabulary_size = max(len(self.vocabulary), 1)

        for state, next_counts in transition_counts.items():
            total = sum(next_counts.values())
            denominator = total + self.smoothing * vocabulary_size
            matrix[state] = {
                next_state: (count + self.smoothing) / denominator
                for next_state, count in next_counts.items()
            }

        return matrix

    def _transition_probability(self, from_state, to_state):
        vocabulary_size = max(len(self.vocabulary), 1)
        outgoing_counts = self.transition_counts.get(from_state, {})
        total = sum(outgoing_counts.values())

        if total == 0:
            return 1.0 / vocabulary_size

        count = outgoing_counts.get(to_state, 0)
        denominator = total + self.smoothing * vocabulary_size
        return float((count + self.smoothing) / denominator)

    def _as_1d_series(self, X):
        array = np.asarray(X, dtype=float)

        if array.ndim == 1:
            return array
        if array.ndim == 2 and 1 in array.shape:
            return array.reshape(-1)

        raise ValueError(
            "ProbabilisticAutomata expects a 1D series. "
            "For multivariate data, pass the PCA PC1 output from the data pipeline."
        )

    @staticmethod
    def _symbol_to_char(value):
        return chr(ord("a") + int(value))
